"""JSON extraction + precision/recall/F1 over (method, smell_type) pairs."""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from typing import Iterable


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict | None:
    """Extract the first JSON object from a model response.

    Handles three formats:
      1. <answer>{...}</answer>           (P4)
      2. ```json {...} ```                (any prompt that ignored 'no fences')
      3. raw {...} starting somewhere     (P1, P2, P3, P5)
    """
    if not text:
        return None

    # 1. <answer>...</answer>
    m = _ANSWER_RE.search(text)
    if m:
        text = m.group(1)

    # 2. fenced block
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3. greedy: from first { to last }
    start = text.find("{")
    if start < 0:
        return None
    end = text.rfind("}")
    if end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        # 3b. balance-trim
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None


# ---------------------------------------------------------------------------
# Predictions → keys
# ---------------------------------------------------------------------------

def prediction_keys(record: dict, parsed: dict | None) -> set[tuple[str, str, str, str]]:
    """Convert a parsed model output into the same key tuple as ground_truth.

    Applies canonical-name normalisation (variant spellings) and drops
    predictions whose `smell_type` is outside the 23-leaf vocabulary
    (e.g. category leaks like "Bloaters"). Method names are canonicalised
    by stripping trailing `(...)` and `Class.` prefix.
    """
    if not parsed:
        return set()
    findings = parsed.get("findings", []) or []
    keys: set[tuple[str, str, str, str]] = set()
    for f in findings:
        if not isinstance(f, dict):
            continue
        st = canonicalise_smell(f.get("smell_type") or "")
        if st is None:
            continue   # silently drop invalid predictions; counted separately
        m = canonicalise_method(f.get("method") or "Entire Class")
        keys.add((record["file_path"], record["class_name"], m, st))
    return keys


def invalid_prediction_count(parsed: dict | None) -> tuple[int, int]:
    """Return (n_invalid_smell, n_total_findings) — diagnostic, not used in P/R/F1."""
    if not parsed:
        return 0, 0
    findings = parsed.get("findings", []) or []
    n = len(findings)
    invalid = sum(
        1 for f in findings
        if isinstance(f, dict) and canonicalise_smell(f.get("smell_type") or "") is None
    )
    return invalid, n


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# Canonical 23-smell vocabulary (kept in sync with prompts/_taxonomy.md).
SMELL_VOCAB: tuple[str, ...] = (
    "Long Method", "Large Class", "Primitive Obsession", "Long Parameter List",
    "Data Clumps",
    "Switch Statements", "Temporary Field", "Refused Bequest",
    "Alternative Classes with Different Interfaces",
    "Parallel Inheritance Hierarchies",
    "Divergent Change", "Shotgun Surgery",
    "Comments", "Duplicate Code", "Lazy Class", "Data Class",
    "Dead Code", "Speculative Generality",
    "Feature Envy", "Inappropriate Intimacy", "Message Chains",
    "Middle Man", "Control Coupling",
)

# Normalize common variant spellings to canonical names. Mirrors the
# "Wrong / Correct" table in prompts/_taxonomy.md so the grader honours the
# contract advertised to the model.
SMELL_ALIASES: dict[str, str] = {
    "message chain":                       "Message Chains",
    "switch statement":                    "Switch Statements",
    "temporary fields":                    "Temporary Field",
    "long parameters list":                "Long Parameter List",
    "long parameters":                     "Long Parameter List",
    "parallel inheritance hierarchy":      "Parallel Inheritance Hierarchies",
    "parallel inheritance":                "Parallel Inheritance Hierarchies",
    "middleman":                           "Middle Man",
    "middle-man":                          "Middle Man",
    "inappropriate intimacies":            "Inappropriate Intimacy",
    "unnecessary comments":                "Comments",
    "useless comments":                    "Comments",
    "duplicated code":                     "Duplicate Code",
    "god class":                           "Large Class",
    "feature envies":                      "Feature Envy",
}

# Categories that small models sometimes emit as `smell_type` by mistake.
_CATEGORY_LEAKS: frozenset[str] = frozenset({
    "bloaters", "object-orientation abusers", "oo abusers",
    "change preventers", "dispensables", "couplers",
})

_VOCAB_CI: dict[str, str] = {s.lower(): s for s in SMELL_VOCAB}


def canonicalise_smell(name: str) -> str | None:
    """Return canonical smell name, or None if it's invalid (category leak,
    unknown string, empty)."""
    if not name:
        return None
    key = name.strip().lower()
    if not key or key in _CATEGORY_LEAKS:
        return None
    if key in _VOCAB_CI:
        return _VOCAB_CI[key]
    if key in SMELL_ALIASES:
        return SMELL_ALIASES[key]
    return None


_METHOD_PARENS_RE = re.compile(r"\(.*?\)\s*$")


def canonicalise_method(name: str) -> str:
    """Strip trailing `(...)`, `Class.` qualifier, surrounding whitespace."""
    if not name:
        return "Entire Class"
    m = name.strip()
    m = _METHOD_PARENS_RE.sub("", m)        # foo(a, b) -> foo
    if "." in m and m.count(".") == 1:      # Cashier.apply_discount -> apply_discount
        m = m.split(".", 1)[1]
    return m or "Entire Class"


def evaluate(per_record: Iterable[dict]) -> dict:
    """Compute overall + per-language + per-smell P/R/F1 + confusion matrix.

    `per_record` items must contain:
        language, file_path, class_name, gold (set), pred (set), parse_error (bool)
    """
    items = list(per_record)
    n = len(items)

    # --- overall + per-language + per-smell tallies ---
    tp = fp = fn = 0
    parse_errors = 0
    truncated_responses = 0
    invalid_findings_total = 0
    total_findings = 0
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    by_smell: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    # per-language × per-smell occurrence-level tallies
    by_lang_smell: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0, 0])
    )
    # per-language record counts (denominator for per-language record-level views)
    lang_n: dict[str, int] = defaultdict(int)

    for it in items:
        if it.get("parse_error"):
            parse_errors += 1
        if it.get("truncated"):
            truncated_responses += 1
        invalid_findings_total += int(it.get("invalid_findings") or 0)
        total_findings         += int(it.get("total_findings") or 0)
        gold, pred = it["gold"], it["pred"]
        ttp = len(gold & pred)
        tfp = len(pred - gold)
        tfn = len(gold - pred)
        tp += ttp; fp += tfp; fn += tfn

        lang = it["language"]
        lang_n[lang] += 1
        by_lang[lang][0] += ttp
        by_lang[lang][1] += tfp
        by_lang[lang][2] += tfn

        for k in (gold & pred):
            by_smell[k[3]][0] += 1
            by_lang_smell[lang][k[3]][0] += 1
        for k in (pred - gold):
            by_smell[k[3]][1] += 1
            by_lang_smell[lang][k[3]][1] += 1
        for k in (gold - pred):
            by_smell[k[3]][2] += 1
            by_lang_smell[lang][k[3]][2] += 1

    # --- per-smell confusion matrix: BOTH occurrence-level + record-level ---
    confusion: dict[str, dict[str, int]] = {}
    for smell in SMELL_VOCAB:
        # occurrence-level: counts of (file,class,method,smell) tuples
        occ_tp, occ_fp, occ_fn = by_smell[smell]
        occ_p, occ_r, occ_f = prf(occ_tp, occ_fp, occ_fn)
        occ_support = occ_tp + occ_fn  # actual GT occurrences

        # record-level: for each file, was the smell present?
        rec_tp = rec_fp = rec_fn = rec_tn = 0
        for it in items:
            in_gold = any(k[3] == smell for k in it["gold"])
            in_pred = any(k[3] == smell for k in it["pred"])
            if in_gold and in_pred:
                rec_tp += 1
            elif in_pred and not in_gold:
                rec_fp += 1
            elif in_gold and not in_pred:
                rec_fn += 1
            else:
                rec_tn += 1
        rec_p, rec_r, rec_f = prf(rec_tp, rec_fp, rec_fn)
        rec_acc = (rec_tp + rec_tn) / n if n else 0.0
        rec_spec = rec_tn / (rec_tn + rec_fp) if (rec_tn + rec_fp) else 0.0
        rec_support = rec_tp + rec_fn  # number of files containing this smell

        confusion[smell] = {
            # occurrence-level (matches the per-smell table from the dataset)
            "occ_tp":        occ_tp,
            "occ_fp":        occ_fp,
            "occ_fn":        occ_fn,
            "occ_precision": round(occ_p, 4),
            "occ_recall":    round(occ_r, 4),
            "occ_f1":        round(occ_f, 4),
            "occ_support":   occ_support,
            # record-level binary (yes/no per file)
            "rec_tp":          rec_tp,
            "rec_fp":          rec_fp,
            "rec_fn":          rec_fn,
            "rec_tn":          rec_tn,
            "rec_precision":   round(rec_p, 4),
            "rec_recall":      round(rec_r, 4),
            "rec_f1":          round(rec_f, 4),
            "rec_specificity": round(rec_spec, 4),
            "rec_accuracy":    round(rec_acc, 4),
            "rec_support":     rec_support,
            # back-compat: old keys mirror record-level view
            "tp": rec_tp, "fp": rec_fp, "fn": rec_fn, "tn": rec_tn,
            "precision":   round(rec_p, 4),
            "recall":      round(rec_r, 4),
            "f1":          round(rec_f, 4),
            "specificity": round(rec_spec, 4),
            "accuracy":    round(rec_acc, 4),
            "support":     rec_support,
        }

    # --- macro / weighted averages over the 23 smells (occurrence-level) ---
    present = [c for c in confusion.values() if c["occ_support"] > 0]
    if present:
        macro_p = sum(c["occ_precision"] for c in present) / len(present)
        macro_r = sum(c["occ_recall"]    for c in present) / len(present)
        macro_f = sum(c["occ_f1"]        for c in present) / len(present)
        total_support = sum(c["occ_support"] for c in present)
        weighted_p = sum(c["occ_precision"] * c["occ_support"] for c in present) / total_support
        weighted_r = sum(c["occ_recall"]    * c["occ_support"] for c in present) / total_support
        weighted_f = sum(c["occ_f1"]        * c["occ_support"] for c in present) / total_support
    else:
        macro_p = macro_r = macro_f = 0.0
        weighted_p = weighted_r = weighted_f = 0.0

    micro_p, micro_r, micro_f = prf(tp, fp, fn)

    # --- per-language × per-smell + per-language summary -----------------
    per_language_per_smell: dict[str, dict[str, dict]] = {}
    per_language_summary:   dict[str, dict] = {}
    for lang in sorted(by_lang.keys()):
        smell_table: dict[str, dict] = {}
        for smell in SMELL_VOCAB:
            s_tp, s_fp, s_fn = by_lang_smell[lang].get(smell, [0, 0, 0])
            sp, sr, sf = prf(s_tp, s_fp, s_fn)
            smell_table[smell] = {
                "tp": s_tp, "fp": s_fp, "fn": s_fn,
                "support": s_tp + s_fn,
                "precision": round(sp, 4),
                "recall":    round(sr, 4),
                "f1":        round(sf, 4),
            }
        per_language_per_smell[lang] = smell_table

        # micro = pooled across all smells for this language
        l_tp, l_fp, l_fn = by_lang[lang]
        lp, lr, lf = prf(l_tp, l_fp, l_fn)

        # macro / weighted across smells PRESENT in this language's GT
        present_l = [c for c in smell_table.values() if c["support"] > 0]
        if present_l:
            l_macro_p = sum(c["precision"] for c in present_l) / len(present_l)
            l_macro_r = sum(c["recall"]    for c in present_l) / len(present_l)
            l_macro_f = sum(c["f1"]        for c in present_l) / len(present_l)
            sup_total = sum(c["support"] for c in present_l)
            l_w_p = sum(c["precision"] * c["support"] for c in present_l) / sup_total
            l_w_r = sum(c["recall"]    * c["support"] for c in present_l) / sup_total
            l_w_f = sum(c["f1"]        * c["support"] for c in present_l) / sup_total
        else:
            l_macro_p = l_macro_r = l_macro_f = 0.0
            l_w_p = l_w_r = l_w_f = 0.0
            sup_total = 0

        per_language_summary[lang] = {
            "n_records": lang_n[lang],
            "tp": l_tp, "fp": l_fp, "fn": l_fn,
            "total_gt_occurrences":   l_tp + l_fn,
            "total_pred_occurrences": l_tp + l_fp,
            "smells_present_in_gt":   len(present_l),
            "micro_precision": round(lp, 4),
            "micro_recall":    round(lr, 4),
            "micro_f1":        round(lf, 4),
            "macro_precision": round(l_macro_p, 4),
            "macro_recall":    round(l_macro_r, 4),
            "macro_f1":        round(l_macro_f, 4),
            "weighted_precision": round(l_w_p, 4),
            "weighted_recall":    round(l_w_r, 4),
            "weighted_f1":        round(l_w_f, 4),
        }

    return {
        "overall": {
            # micro = pooled-key P/R/F1 over (file, class, method, smell)
            "precision":  round(micro_p, 4),
            "recall":     round(micro_r, 4),
            "f1":         round(micro_f, 4),
            "micro_precision": round(micro_p, 4),
            "micro_recall":    round(micro_r, 4),
            "micro_f1":        round(micro_f, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall":    round(macro_r, 4),
            "macro_f1":        round(macro_f, 4),
            "weighted_precision": round(weighted_p, 4),
            "weighted_recall":    round(weighted_r, 4),
            "weighted_f1":        round(weighted_f, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "n_records":           n,
            "parse_errors":        parse_errors,
            "truncated_responses": truncated_responses,
            "invalid_findings":    invalid_findings_total,
            "total_findings":      total_findings,
        },
        "per_language": {
            lang: dict(zip(("precision", "recall", "f1"),
                            (round(x, 4) for x in prf(*counts))),
                       **{"tp": counts[0], "fp": counts[1], "fn": counts[2]})
            for lang, counts in sorted(by_lang.items())
        },
        "per_smell": {
            smell: dict(zip(("precision", "recall", "f1"),
                             (round(x, 4) for x in prf(*counts))),
                        **{"tp": counts[0], "fp": counts[1], "fn": counts[2]})
            for smell, counts in sorted(by_smell.items())
        },
        "per_language_per_smell": per_language_per_smell,
        "per_language_summary":   per_language_summary,
        "confusion_matrix": confusion,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (record-level resampling)
# ---------------------------------------------------------------------------

def _micro_prf_from_items(items: list[dict]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for it in items:
        gold, pred = it["gold"], it["pred"]
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return prf(tp, fp, fn)


def bootstrap_ci(
    items: list[dict],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI over micro P/R/F1 by resampling records with
    replacement. Standard for set-prediction tasks; see Berg-Kirkpatrick 2012,
    Dror et al. 2018 (ACL Hitchhiker's Guide).

    Returns
    -------
    {
      "n_resamples", "alpha",
      "micro_precision":  {"point": .., "lo": .., "hi": .., "stderr": ..},
      "micro_recall":     {...},
      "micro_f1":         {...},
    }
    """
    items = list(items)
    n = len(items)
    if n == 0:
        empty = {"point": 0.0, "lo": 0.0, "hi": 0.0, "stderr": 0.0}
        return {
            "n_resamples": 0, "alpha": alpha,
            "micro_precision": empty, "micro_recall": empty, "micro_f1": empty,
        }

    rng = random.Random(seed)
    p_samples: list[float] = []
    r_samples: list[float] = []
    f_samples: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        sample = [items[i] for i in idx]
        sp, sr, sf = _micro_prf_from_items(sample)
        p_samples.append(sp); r_samples.append(sr); f_samples.append(sf)

    point_p, point_r, point_f = _micro_prf_from_items(items)

    def _summary(samples: list[float], point: float) -> dict:
        s = sorted(samples)
        lo = s[int(alpha / 2 * n_resamples)]
        hi = s[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
        mean = sum(s) / len(s)
        var = sum((x - mean) ** 2 for x in s) / max(1, len(s) - 1)
        return {
            "point":  round(point, 4),
            "lo":     round(lo, 4),
            "hi":     round(hi, 4),
            "stderr": round(var ** 0.5, 4),
        }

    return {
        "n_resamples": n_resamples,
        "alpha":       alpha,
        "micro_precision": _summary(p_samples, point_p),
        "micro_recall":    _summary(r_samples, point_r),
        "micro_f1":        _summary(f_samples, point_f),
    }
