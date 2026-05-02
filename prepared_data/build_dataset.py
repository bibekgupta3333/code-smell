"""
Build line-annotated code-smell datasets from SmellyCodeDataset.

Outputs (under prepared_data/datasets/):
  annotated/   -> source from SmellyAnnotated/, with line-level smell ranges
                  extracted from inline `// SmellName` comments.
  unannotated/ -> source from SmellyUnannotated/ (no inline labels), paired
                  with method-level ground-truth from
                  Analysis/GroundTruthLevel/Cleaned_GroundTruth.csv for grading.

Per-language JSONs + combined `all.json` + stratified train/val/test
(60/20/20, seed=42, stratified by language).
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "datasets" / "SmellyCodeDataset"
OUT_ROOT = ROOT / "prepared_data" / "datasets"

# -------- canonical taxonomy (Fowler) --------
CATEGORY_OF: dict[str, str] = {
    # Bloaters
    "Long Method": "Bloaters",
    "Large Class": "Bloaters",
    "Primitive Obsession": "Bloaters",
    "Long Parameter List": "Bloaters",
    "Data Clumps": "Bloaters",
    # Object-Orientation Abusers
    "Switch Statements": "Object-Orientation Abusers",
    "Temporary Field": "Object-Orientation Abusers",
    "Refused Bequest": "Object-Orientation Abusers",
    "Alternative Classes with Different Interfaces": "Object-Orientation Abusers",
    "Parallel Inheritance Hierarchies": "Object-Orientation Abusers",
    # Change Preventers
    "Divergent Change": "Change Preventers",
    "Shotgun Surgery": "Change Preventers",
    # Dispensables
    "Comments": "Dispensables",
    "Duplicate Code": "Dispensables",
    "Lazy Class": "Dispensables",
    "Data Class": "Dispensables",
    "Dead Code": "Dispensables",
    "Speculative Generality": "Dispensables",
    # Couplers
    "Feature Envy": "Couplers",
    "Inappropriate Intimacy": "Couplers",
    "Message Chains": "Couplers",
    "Middle Man": "Couplers",
    "Control Coupling": "Couplers",
}

# Map alternate spellings/synonyms found in source comments -> canonical.
ALIASES: dict[str, str] = {
    "Message Chain": "Message Chains",
    "Switch Statement": "Switch Statements",
    "Temporary Fields": "Temporary Field",
    "Long Parameters List": "Long Parameter List",
    "Long Parameters": "Long Parameter List",
    "Parallel Inheritance Hierarchy": "Parallel Inheritance Hierarchies",
    "Parallel Inheritance": "Parallel Inheritance Hierarchies",
    "Middleman": "Middle Man",
    "Inappropriate Intimacies": "Inappropriate Intimacy",
    "Unnecessary Comments": "Comments",
    "Useless Comments": "Comments",
}

# Order matters: longer / more specific names first to win the regex race.
SMELL_NAMES: list[str] = sorted(
    set(list(CATEGORY_OF.keys()) + list(ALIASES.keys())),
    key=lambda s: (-len(s), s),
)

# Match smell name only when it appears as a comment payload (case-sensitive
# to avoid accidentally hitting identifiers like 'longMethod').
SMELL_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in SMELL_NAMES) + r")\b")

LANG_DIR = {
    "java": ("Java", ".java"),
    "python": ("Python", ".py"),
    "javascript": ("JavaScript", ".js"),
    "cpp": ("C++", (".cpp", ".h")),
}

COMMENT_PREFIXES = {
    "java": ("//", "/*", "*"),
    "javascript": ("//", "/*", "*"),
    "cpp": ("//", "/*", "*"),
    "python": ("#",),
}

# Filenames to skip even if present (compiled artifacts already excluded by ext).
SKIP_NAMES = {"Makefile"}


def is_comment_line(line: str, lang: str) -> bool:
    s = line.lstrip()
    return any(s.startswith(p) for p in COMMENT_PREFIXES[lang])


def comment_payload(line: str, lang: str) -> str:
    """Return text after the comment marker on this line, '' if no comment."""
    if lang == "python":
        i = line.find("#")
        return line[i + 1 :] if i >= 0 else ""
    # C-family: prefer //, else /* ... */
    i = line.find("//")
    if i >= 0:
        return line[i + 2 :]
    i = line.find("/*")
    if i >= 0:
        return line[i + 2 :].rstrip("*/").strip()
    s = line.lstrip()
    if s.startswith("*"):
        return s[1:]
    return ""


def canonical_smell(raw: str) -> str | None:
    raw = raw.strip()
    return ALIASES.get(raw, raw if raw in CATEGORY_OF else None)


def find_method_for_line(lines: list[str], lineno: int, lang: str) -> str:
    """Best-effort: walk back from `lineno` (1-based) to nearest method/def
    header; return method name or 'Entire Class'."""
    if lang == "python":
        pat = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(")
    elif lang in ("java", "javascript"):
        # crude: a line that looks like a method header
        pat = re.compile(
            r"^\s*(?:public|private|protected|static|async|function|constructor|\w[\w<>\[\]]*\s+)*"
            r"([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$"
        )
    else:  # cpp
        pat = re.compile(
            r"^\s*[\w:~<>*&\s]+?\s+([A-Za-z_]\w*)::([A-Za-z_]\w*)\s*\("
        )
    for i in range(lineno - 1, -1, -1):
        m = pat.match(lines[i])
        if m:
            # cpp uses group(2); others group(1)
            name = m.group(2) if (lang == "cpp" and m.lastindex and m.lastindex >= 2) else m.group(1)
            # filter common false positives
            if name and name not in {"if", "for", "while", "switch", "return", "catch"}:
                return name
    return "Entire Class"


def block_extent(lines: list[str], lineno: int, lang: str) -> int:
    """If the comment header introduces a multi-line block (e.g. consecutive
    field declarations), expand line_end to the last non-blank line of that
    block. Stops at the first blank line or next comment-only line."""
    n = len(lines)
    end = lineno
    for i in range(lineno, n):
        s = lines[i].strip()
        if not s:
            break
        if is_comment_line(lines[i], lang):
            break
        end = i + 1
    return max(end, lineno)


def method_brace_end(lines: list[str], header_lineno: int) -> int:
    """For a C-family method header on `header_lineno` (1-based), return the
    line number of the matching closing brace. Returns header_lineno if no
    body brace is found."""
    n = len(lines)
    # find first '{' on or after header_lineno
    i = header_lineno - 1
    depth = 0
    started = False
    while i < n:
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    return i + 1
        i += 1
    return header_lineno


def looks_like_method_header(line: str, lang: str) -> bool:
    # Strip trailing line comment so "(unused method)" inside `// ...` doesn't
    # fool the heuristic.
    code = line
    if lang == "python":
        i = code.find("#")
        if i >= 0:
            code = code[:i]
        return bool(re.match(r"^\s*def\s+\w+\s*\(", code))
    if lang in ("java", "javascript", "cpp"):
        i = code.find("//")
        if i >= 0:
            code = code[:i]
        s = code.strip()
        if not s or ";" in s:
            return False
        return bool(re.search(r"\)\s*\{?\s*$", s)) and "(" in s


def extract_annotations(text: str, lang: str) -> list[dict]:
    lines = text.splitlines()
    n = len(lines)
    anns: list[dict] = []
    for i, raw in enumerate(lines):
        payload = comment_payload(raw, lang)
        if not payload:
            continue
        for m in SMELL_RE.finditer(payload):
            canon = canonical_smell(m.group(1))
            if not canon:
                continue
            line_start = i + 1

            # decide kind + line_end
            stripped = raw.lstrip()
            comment_only = (
                stripped.startswith("//")
                or stripped.startswith("#")
                or stripped.startswith("/*")
                or stripped.startswith("*")
            )

            if comment_only:
                # Attached to the next code construct.
                # If the next non-blank, non-comment line is a method header,
                # span the whole method body. Else expand block.
                j = i + 1
                while j < n and (not lines[j].strip() or is_comment_line(lines[j], lang)):
                    j += 1
                if j < n and looks_like_method_header(lines[j], lang):
                    if lang == "python":
                        # python: span until dedent
                        header_indent = len(lines[j]) - len(lines[j].lstrip())
                        end = j + 1
                        for k in range(j + 1, n):
                            s = lines[k]
                            if s.strip() == "":
                                continue
                            ind = len(s) - len(s.lstrip())
                            if ind <= header_indent:
                                break
                            end = k + 1
                        line_end = end
                    else:
                        line_end = method_brace_end(lines, j + 1)
                    kind = "method-scope"
                    method_name = find_method_for_line(lines, j + 1, lang)
                else:
                    line_end = block_extent(lines, line_start, lang)
                    kind = "block"
                    method_name = find_method_for_line(lines, line_start, lang)
            else:
                # Inline trailing comment on a code line.
                line_end = line_start
                kind = "inline"
                # If the code on this line is itself a method header, span method.
                if looks_like_method_header(raw, lang):
                    if lang == "python":
                        header_indent = len(raw) - len(raw.lstrip())
                        end = i + 1
                        for k in range(i + 1, n):
                            s = lines[k]
                            if s.strip() == "":
                                continue
                            ind = len(s) - len(s.lstrip())
                            if ind <= header_indent:
                                break
                            end = k + 1
                        line_end = end
                    else:
                        line_end = method_brace_end(lines, line_start)
                    kind = "method-scope"
                method_name = find_method_for_line(lines, line_start, lang)

            anns.append(
                {
                    "smell_type": canon,
                    "category": CATEGORY_OF[canon],
                    "method": method_name,
                    "line_start": line_start,
                    "line_end": line_end,
                    "evidence": raw.rstrip(),
                    "annotation_kind": kind,
                }
            )
    # de-duplicate identical (smell_type, line_start, line_end)
    seen = set()
    uniq = []
    for a in anns:
        k = (a["smell_type"], a["line_start"], a["line_end"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(a)
    return uniq


def load_ground_truth_csv() -> dict[tuple[str, str], list[dict]]:
    """Returns (lang_norm, class_name) -> list of {smell_type, category, method, description}."""
    path = (
        DATASET_ROOT
        / "Analysis"
        / "GroundTruthLevel"
        / "Cleaned_GroundTruth.csv"
    )
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    lang_map = {"Java": "java", "Python": "python", "JavaScript": "javascript", "C++": "cpp"}
    with path.open() as f:
        for row in csv.DictReader(f):
            lang = lang_map.get(row["Language"])
            if not lang:
                continue
            smell = ALIASES.get(row["Code Smell"], row["Code Smell"])
            out[(lang, row["Class"])].append(
                {
                    "smell_type": smell,
                    "category": row["Category"],
                    "method": row["Method"],
                    "description": row.get("Type", ""),
                }
            )
    return out


def collect_files(lang: str, variant: str) -> list[Path]:
    """variant in {'SmellyAnnotated','SmellyUnannotated'}"""
    sub, ext = LANG_DIR[lang]
    folder = DATASET_ROOT / sub / variant
    out: list[Path] = []
    if isinstance(ext, str):
        out = sorted(folder.glob(f"*{ext}"))
    else:
        for e in ext:
            out.extend(sorted(folder.glob(f"*{e}")))
        out.sort()
    return [p for p in out if p.name not in SKIP_NAMES]


def class_name_from_file(p: Path, lang: str) -> str:
    stem = p.stem
    # main.java / Main.py / main.cpp -> 'main'
    return stem


def build_record_annotated(p: Path, lang: str) -> dict:
    text = p.read_text(encoding="utf-8", errors="replace")
    rel = str(p.relative_to(DATASET_ROOT))
    cls = class_name_from_file(p, lang)
    anns = extract_annotations(text, lang)
    loc = len(text.splitlines())
    return {
        "sample_id": f"{lang}_{cls}_{p.suffix.lstrip('.')}",
        "language": lang,
        "class_name": cls,
        "file_path": rel,
        "loc": loc,
        "source_code": text,
        "annotations": anns,
        "metadata": {
            "variant": "annotated",
            "num_annotations": len(anns),
            "num_lines": loc,
        },
    }


def build_record_unannotated(
    p: Path, lang: str, gt_index: dict[tuple[str, str], list[dict]]
) -> dict:
    text = p.read_text(encoding="utf-8", errors="replace")
    rel = str(p.relative_to(DATASET_ROOT))
    cls = class_name_from_file(p, lang)
    loc = len(text.splitlines())
    gt = gt_index.get((lang, cls), [])
    return {
        "sample_id": f"{lang}_{cls}_{p.suffix.lstrip('.')}",
        "language": lang,
        "class_name": cls,
        "file_path": rel,
        "loc": loc,
        "source_code": text,
        "annotations": [],  # blind
        "ground_truth": gt,  # method-level labels for grading
        "metadata": {
            "variant": "unannotated",
            "num_ground_truth": len(gt),
            "num_lines": loc,
        },
    }


def stratified_split(
    samples: list[dict], ratios=(0.6, 0.2, 0.2), seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_lang[s["language"]].append(s)
    train, val, test = [], [], []
    for lang, group in by_lang.items():
        g = group[:]
        rng.shuffle(g)
        n = len(g)
        n_tr = max(1, int(round(n * ratios[0])))
        n_val = max(1, int(round(n * ratios[1])))
        # ensure at least 1 in test
        if n_tr + n_val >= n:
            n_val = max(1, n - n_tr - 1)
        train += g[:n_tr]
        val += g[n_tr : n_tr + n_val]
        test += g[n_tr + n_val :]
    return train, val, test


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_occurrence_csvs(
    out_dir: Path, variant: str, per_lang: dict[str, list[dict]]
) -> None:
    """Write one row per smell occurrence.

    Columns: language, file_path, class_name, smell_type, category, method,
             line_start, line_end, annotation_kind, evidence

    For `unannotated` (method-level CSV ground truth has no line numbers),
    line_start/line_end/annotation_kind/evidence are emitted as empty.

    Produces:
      <out_dir>/occurrences/<lang>.csv   per language
      <out_dir>/occurrences/all.csv      combined
    """
    occ_dir = out_dir / "occurrences"
    occ_dir.mkdir(parents=True, exist_ok=True)
    header = [
        "language", "file_path", "class_name",
        "smell_type", "category", "method",
        "line_start", "line_end",
        "annotation_kind", "evidence",
    ]

    all_rows: list[list] = []
    for lang, recs in per_lang.items():
        rows: list[list] = []
        for r in recs:
            labels = (
                r.get("annotations")
                if variant == "annotated"
                else r.get("ground_truth", [])
            )
            for a in labels:
                rows.append([
                    r["language"],
                    r["file_path"],
                    r["class_name"],
                    a.get("smell_type", ""),
                    a.get("category", ""),
                    a.get("method", ""),
                    a.get("line_start", ""),
                    a.get("line_end", ""),
                    a.get("annotation_kind", ""),
                    a.get("evidence", a.get("description", "")),
                ])
        # per-language CSV
        with (occ_dir / f"{lang}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        all_rows.extend(rows)

    with (occ_dir / "all.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(all_rows)


def build_variant(variant: str, gt_index) -> dict:
    """variant in {'annotated','unannotated'}"""
    src_variant = "SmellyAnnotated" if variant == "annotated" else "SmellyUnannotated"
    out_dir = OUT_ROOT / variant
    per_lang: dict[str, list[dict]] = {}
    all_samples: list[dict] = []
    for lang in LANG_DIR:
        files = collect_files(lang, src_variant)
        recs = []
        for p in files:
            if variant == "annotated":
                recs.append(build_record_annotated(p, lang))
            else:
                recs.append(build_record_unannotated(p, lang, gt_index))
        per_lang[lang] = recs
        all_samples.extend(recs)
        write_json(out_dir / f"{lang}.json", recs)

    write_json(out_dir / "all.json", all_samples)

    train, val, test = stratified_split(all_samples)
    write_json(out_dir / "train.json", train)
    write_json(out_dir / "val.json", val)
    write_json(out_dir / "test.json", test)

    # Per-occurrence CSV (one row per smell instance) — used for P/R/F1.
    write_occurrence_csvs(out_dir, variant, per_lang)

    # summary
    smell_counts: Counter = Counter()
    cat_counts: Counter = Counter()
    per_lang_smell: dict[str, Counter] = defaultdict(Counter)
    for s in all_samples:
        labels = s.get("annotations") if variant == "annotated" else s.get("ground_truth", [])
        for a in labels:
            smell_counts[a["smell_type"]] += 1
            cat_counts[a["category"]] += 1
            per_lang_smell[s["language"]][a["smell_type"]] += 1

    return {
        "variant": variant,
        "num_files": len(all_samples),
        "files_per_language": {k: len(v) for k, v in per_lang.items()},
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "total_labels": sum(smell_counts.values()),
        "smell_counts": dict(smell_counts.most_common()),
        "category_counts": dict(cat_counts.most_common()),
        "per_language_smell_counts": {
            k: dict(v.most_common()) for k, v in per_lang_smell.items()
        },
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    gt_index = load_ground_truth_csv()
    summary = {
        "annotated": build_variant("annotated", gt_index),
        "unannotated": build_variant("unannotated", gt_index),
        "smell_taxonomy": {
            "categories": sorted(set(CATEGORY_OF.values())),
            "smells": sorted(CATEGORY_OF.keys()),
            "aliases": ALIASES,
        },
    }
    write_json(OUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
