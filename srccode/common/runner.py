"""Shared run loop used by all 5 prompt scripts.

Each `run_pN_*.py` file just imports `run(prompt_name)` and calls it.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from . import config, dataset, evaluator, llm_client, prompt_loader


def _print_summary(metrics: dict, verbose: bool = False) -> None:
    o = metrics["overall"]
    print(f"\n{'=' * 72}")
    print(f"Records: {o['n_records']}   Parse errors: {o['parse_errors']}")
    print(f"Pooled keys: TP={o['tp']}  FP={o['fp']}  FN={o['fn']}")

    tu = metrics.get("token_usage")
    if tu:
        print(f"Tokens   : in={tu['total_input_tokens']}  out={tu['total_output_tokens']}  "
              f"total={tu['total_tokens']}   "
              f"avg in/out={tu['avg_input_tokens']:.0f}/{tu['avg_output_tokens']:.0f}   "
              f"max in/out={tu['max_input_tokens']}/{tu['max_output_tokens']}")
    print(f"\n{'Metric':22s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print(f"{'-' * 56}")
    print(f"{'Micro (pooled keys)':22s} {o['micro_precision']:>10.4f} {o['micro_recall']:>10.4f} {o['micro_f1']:>10.4f}")
    print(f"{'Macro (avg over smells)':22s} {o['macro_precision']:>10.4f} {o['macro_recall']:>10.4f} {o['macro_f1']:>10.4f}")
    print(f"{'Weighted (by support)':22s} {o['weighted_precision']:>10.4f} {o['weighted_recall']:>10.4f} {o['weighted_f1']:>10.4f}")

    bs = metrics.get("bootstrap_ci")
    if bs:
        ci = int((1 - bs["alpha"]) * 100)
        print(f"\nBootstrap {ci}% CI  ({bs['n_resamples']} resamples, percentile method):")
        for k, label in (("micro_precision", "Micro Precision"),
                         ("micro_recall",    "Micro Recall   "),
                         ("micro_f1",        "Micro F1       ")):
            s = bs[k]
            print(f"  {label}  {s['point']:.4f}  [{s['lo']:.4f}, {s['hi']:.4f}]   SE={s['stderr']:.4f}")

    print(f"\n{'=' * 72}\nPer-language summary (occurrence-level, micro/macro/weighted):")
    print(f"  {'lang':<10s} {'N':>3s} {'GT':>4s} {'Pred':>4s} | "
          f"{'micP':>5s} {'micR':>5s} {'micF1':>6s} | "
          f"{'macP':>5s} {'macR':>5s} {'macF1':>6s} | "
          f"{'wP':>5s} {'wR':>5s} {'wF1':>6s}")
    print("  " + "-" * 88)
    for lang, s in metrics.get("per_language_summary", {}).items():
        print(f"  {lang:<10s} {s['n_records']:>3d} {s['total_gt_occurrences']:>4d} {s['total_pred_occurrences']:>4d} | "
              f"{s['micro_precision']:>5.2f} {s['micro_recall']:>5.2f} {s['micro_f1']:>6.3f} | "
              f"{s['macro_precision']:>5.2f} {s['macro_recall']:>5.2f} {s['macro_f1']:>6.3f} | "
              f"{s['weighted_precision']:>5.2f} {s['weighted_recall']:>5.2f} {s['weighted_f1']:>6.3f}")

    if not verbose:
        return

    # Per-language × per-smell tables (occurrence-level: actual vs predicted)
    plps = metrics.get("per_language_per_smell", {})
    for lang in plps:
        present = [(s, c) for s, c in plps[lang].items() if c["support"] > 0 or c["fp"] > 0]
        if not present:
            continue
        present.sort(key=lambda kv: (-kv[1]["support"], -kv[1]["fp"], kv[0]))
        gt_total = sum(c["support"] for _, c in present)
        pred_total = sum(c["tp"] + c["fp"] for _, c in present)
        print(f"\n{'-' * 84}\n"
              f"[{lang}] per-smell — actual vs predicted (occurrence-level)   "
              f"GT={gt_total}  Pred={pred_total}")
        print(f"  {'Smell':<42s} {'Act':>4s} {'Pred':>5s} {'TP':>3s} {'FP':>3s} {'FN':>3s} "
              f"{'P':>5s} {'R':>5s} {'F1':>5s}")
        print("  " + "-" * 82)
        for smell, c in present:
            print(f"  {smell:<42s} {c['support']:>4d} {c['tp']+c['fp']:>5d} "
                  f"{c['tp']:>3d} {c['fp']:>3d} {c['fn']:>3d} "
                  f"{c['precision']:>5.2f} {c['recall']:>5.2f} {c['f1']:>5.2f}")

    print(f"\n{'=' * 100}\nConfusion matrix per smell (ALL languages combined) — occurrence-level + record-level:")
    print(f"{'Smell':<48s} | "
          f"{'oTP':>4s} {'oFP':>4s} {'oFN':>4s} {'oP':>5s} {'oR':>5s} {'oF1':>5s} {'oSup':>4s} | "
          f"{'rTP':>3s} {'rFP':>3s} {'rFN':>3s} {'rTN':>3s} {'rF1':>5s} {'rSup':>4s}")
    print("-" * 120)
    cm = metrics["confusion_matrix"]
    rows = sorted(cm.items(), key=lambda kv: (-kv[1]["occ_support"], kv[0]))
    for smell, c in rows:
        print(f"{smell:<48s} | "
              f"{c['occ_tp']:>4d} {c['occ_fp']:>4d} {c['occ_fn']:>4d} "
              f"{c['occ_precision']:>5.2f} {c['occ_recall']:>5.2f} {c['occ_f1']:>5.2f} "
              f"{c['occ_support']:>4d} | "
              f"{c['rec_tp']:>3d} {c['rec_fp']:>3d} {c['rec_fn']:>3d} {c['rec_tn']:>3d} "
              f"{c['rec_f1']:>5.2f} {c['rec_support']:>4d}")
    print("\nLegend: Act = actual GT occurrences, Pred = total predicted, "
          "o* = occurrence-level, r* = record-level binary (per file)")


def run(prompt_name: str) -> int:
    """Entry point shared by all 5 scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=f"Run {prompt_name}")
    parser.add_argument("--provider", choices=["auto", "local", "cloud", "bedrock"],
                        default="auto",
                        help="Which LLM backend to use. "
                             "'auto' = detect from OLLAMA_HOST env var (default). "
                             "'local' = Ollama at http://localhost:11434. "
                             "'cloud' = Ollama Cloud at https://ollama.com (needs OLLAMA_API_KEY). "
                             "'bedrock' = AWS Bedrock via boto3 converse API "
                             "(needs AWS credentials + region; see config.AWS_REGION).")
    parser.add_argument("--model", default=None,
                        help="Model id. For Ollama: tag like qwen2.5-coder:7b. "
                             "For Bedrock: full model id like "
                             "anthropic.claude-3-5-sonnet-20240620-v1:0. "
                             "Default depends on --provider.")
    parser.add_argument("--language", choices=["java", "python", "javascript", "cpp"],
                        default=None, help="Filter to one language")
    parser.add_argument("--dataset", choices=["unannotated", "annotated"],
                        default="unannotated",
                        help="Which annotation flavour to evaluate against")
    parser.add_argument("--split",
                        choices=["all", "train", "val", "test",
                                 "java", "python", "javascript", "cpp"],
                        default="test",
                        help="Which file under prepared_data/datasets/<dataset>/ to load. "
                             "'all'=34 records, 'train'=20, 'val'=6, 'test'=8, "
                             "or a language name to load that language's full set.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of records (debug)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-ctx", type=int, default=None,
                        help="Input context window in tokens (Ollama only). "
                             "Default: 8192 for p1/p3/p4, 16384 for p2/p5. "
                             "Ignored on Bedrock (context is fixed per model).")
    parser.add_argument("--num-predict", type=int, default=4096,
                        help="Max output tokens (default 4096). "
                             "Maps to Ollama num_predict / Bedrock maxTokens.")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for micro P/R/F1 95%% CIs. "
                             "Set 0 to disable. Default 1000.")
    parser.add_argument("--no-csv", action="store_true",
                        help="Skip per-record CSV export.")
    parser.add_argument("--rag-mode", choices=["dense", "random"], default="dense",
                        help="P5 retrieval mode: 'dense' (default) uses "
                             "sentence-transformers/all-MiniLM-L6-v2 cosine "
                             "similarity over the train split; 'random' is the "
                             "null-retriever ablation baseline.")
    parser.add_argument("--rag-k", type=int, default=2,
                        help="Number of exemplars retrieved for P5 (default 2).")
    parser.add_argument("--output-dir", default=str(config.RESULTS_ROOT))
    parser.add_argument("--verbose", action="store_true",
                        help="Log full system and user prompts before each invocation")
    args = parser.parse_args()

    # Apply --provider override BEFORE the health check / default-model lookup.
    if args.provider == "local":
        config.PROVIDER = "local"
        config.OLLAMA_HOST = "http://localhost:11434"
    elif args.provider == "cloud":
        config.PROVIDER = "cloud"
        config.OLLAMA_HOST = "https://ollama.com"
        if not config.OLLAMA_API_KEY:
            print("WARNING: --provider=cloud but OLLAMA_API_KEY is not set.")
    elif args.provider == "bedrock":
        config.PROVIDER = "bedrock"
    else:  # auto
        config.PROVIDER = "cloud" if "ollama.com" in config.OLLAMA_HOST else "local"

    if args.model is None:
        args.model = config.default_model()

    if args.num_ctx is None:
        # Long-prompt strategies need more context to avoid output truncation.
        args.num_ctx = 16384 if prompt_name in ("p2_few_shot", "p5_rag") else 8192

    mode = config.PROVIDER
    host = (f"AWS Bedrock {config.AWS_REGION or 'cli-default'}") if config.is_bedrock() else config.OLLAMA_HOST
    print(f"Provider: {mode}  (host={host})")
    print(llm_client.health_check())
    print(f"Model: {args.model}")
    print(f"Prompt: {prompt_name}  num_ctx={args.num_ctx}  num_predict={args.num_predict}")
    print(f"Dataset: {args.dataset}/{args.split}.json")

    records = dataset.load_test(args.language, dataset=args.dataset, split=args.split)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("No test records found.")
        return 1
    print(f"Records: {len(records)}\n")

    per_record_results: list[dict] = []
    raw_responses: list[dict] = []

    t0 = time.time()
    for i, rec in enumerate(records, 1):
        sys_text, user_text = prompt_loader.render(
            rec["language"], prompt_name, rec,
            rag_mode=args.rag_mode, rag_k=args.rag_k,
        )
        logging.info(f"[{i}/{len(records)}] Processing {rec['sample_id']} ({rec['language']}/{rec['class_name']}) — system_bytes={len(sys_text)} user_bytes={len(user_text)}")
        if args.verbose:
            logging.info(f"SYSTEM PROMPT:\n{sys_text}\n\nUSER PROMPT:\n{user_text}")
        try:
            response, usage = llm_client.chat(
                args.model, sys_text, user_text,
                temperature=args.temperature, seed=args.seed,
                num_ctx=args.num_ctx, num_predict=args.num_predict,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(records)}] {rec['sample_id']}: REQUEST FAILED: {e}")
            response = ""
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        parsed = evaluator.extract_json(response)
        gold = dataset.ground_truth_keys(rec)
        pred = evaluator.prediction_keys(rec, parsed)
        parse_error = parsed is None
        invalid_findings, total_findings = evaluator.invalid_prediction_count(parsed)
        # Heuristic: response was non-empty but JSON failed to balance,
        # OR response is non-empty and does not contain a closing brace.
        truncated = bool(response) and (
            parse_error or response.rstrip().endswith((",", '"', ":")) or
            response.count("{") > response.count("}")
        )

        if args.verbose:
            logging.info(f"LLM RESPONSE ({len(response)} chars):\n{response}")
            logging.info(
                f"PARSED FINDINGS: {total_findings if parsed else 'PARSE_ERROR'}  "
                f"invalid={invalid_findings}  truncated={truncated}"
            )

        item = {
            "language":    rec["language"],
            "file_path":   rec["file_path"],
            "class_name":  rec["class_name"],
            "gold":        gold,
            "pred":        pred,
            "parse_error": parse_error,
            "truncated":   truncated,
            "invalid_findings": invalid_findings,
            "total_findings":   total_findings,
            "input_tokens":  usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens":  usage["total_tokens"],
        }
        per_record_results.append(item)
        raw_responses.append({
            "sample_id":   rec["sample_id"],
            "language":    rec["language"],
            "file_path":   rec["file_path"],
            "parsed":      parsed,
            "gold_keys":   sorted(list(gold)),
            "pred_keys":   sorted(list(pred)),
            "parse_error": parse_error,
            "truncated":   truncated,
            "invalid_findings": invalid_findings,
            "total_findings":   total_findings,
            "input_tokens":  usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens":  usage["total_tokens"],
            "raw":         response,
        })

        tp = len(gold & pred); fp = len(pred - gold); fn = len(gold - pred)
        if parse_error:
            flag = "PARSE_ERR" + ("+TRUNC" if truncated else "")
        else:
            flag = f"tp={tp} fp={fp} fn={fn}"
            if invalid_findings:
                flag += f" inv={invalid_findings}"
            if truncated:
                flag += " TRUNC"
        tok_str = f" tok={usage['input_tokens']}→{usage['output_tokens']}"
        print(f"  [{i}/{len(records)}] {rec['sample_id']:35s} {flag}{tok_str}")

    elapsed = time.time() - t0
    metrics = evaluator.evaluate(per_record_results)
    if args.bootstrap > 0:
        metrics["bootstrap_ci"] = evaluator.bootstrap_ci(
            per_record_results,
            n_resamples=args.bootstrap,
            alpha=0.05,
            seed=args.seed,
        )
    metrics["meta"] = {
        "prompt":       prompt_name,
        "model":        args.model,
        "provider":     config.PROVIDER,
        "host":         host,
        "is_cloud":     config.is_cloud(),
        "is_bedrock":   config.is_bedrock(),
        "dataset":      args.dataset,
        "split":        args.split,
        "language":     args.language or "all",
        "limit":        args.limit,
        "temperature":  args.temperature,
        "seed":         args.seed,
        "num_ctx":      args.num_ctx,
        "num_predict":  args.num_predict,
        "bootstrap":    args.bootstrap,
        "rag_mode":     args.rag_mode if prompt_name == "p5_rag" else None,
        "rag_k":        args.rag_k    if prompt_name == "p5_rag" else None,
        "elapsed_sec":  round(elapsed, 1),
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
    }

    # Aggregate token usage
    tok_in  = sum(r["input_tokens"]  for r in per_record_results)
    tok_out = sum(r["output_tokens"] for r in per_record_results)
    n = max(len(per_record_results), 1)
    metrics["token_usage"] = {
        "total_input_tokens":  tok_in,
        "total_output_tokens": tok_out,
        "total_tokens":        tok_in + tok_out,
        "avg_input_tokens":    round(tok_in  / n, 1),
        "avg_output_tokens":   round(tok_out / n, 1),
        "max_input_tokens":    max((r["input_tokens"]  for r in per_record_results), default=0),
        "max_output_tokens":   max((r["output_tokens"] for r in per_record_results), default=0),
    }

    _print_summary(metrics, verbose=args.verbose)

    # ---- persist results ----
    out_dir = Path(args.output_dir) / prompt_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "_").replace(":", "_")
    base = f"{prompt_name}__{safe_model}__{args.dataset}__{args.split}__{args.language or 'all'}__{stamp}"

    (out_dir / f"{base}.metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (out_dir / f"{base}.predictions.json").write_text(
        json.dumps(raw_responses, indent=2, default=str), encoding="utf-8"
    )

    if not args.no_csv:
        csv_path = out_dir / f"{base}.per_record.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "sample_id", "language", "file_path", "class_name",
                "gold_count", "pred_count",
                "tp", "fp", "fn",
                "precision", "recall", "f1",
                "parse_error", "truncated",
                "invalid_findings", "total_findings",
                "input_tokens", "output_tokens", "total_tokens",
            ])
            for raw, item in zip(raw_responses, per_record_results):
                gold = item["gold"]; pred = item["pred"]
                tp_i = len(gold & pred)
                fp_i = len(pred - gold)
                fn_i = len(gold - pred)
                p_i, r_i, f_i = evaluator.prf(tp_i, fp_i, fn_i)
                w.writerow([
                    raw["sample_id"], item["language"],
                    item["file_path"], item["class_name"],
                    len(gold), len(pred),
                    tp_i, fp_i, fn_i,
                    f"{p_i:.4f}", f"{r_i:.4f}", f"{f_i:.4f}",
                    int(item["parse_error"]), int(item["truncated"]),
                    item["invalid_findings"], item["total_findings"],
                    item["input_tokens"], item["output_tokens"], item["total_tokens"],
                ])
        print(f"Saved → {csv_path}")

    print(f"\nSaved → {out_dir / (base + '.metrics.json')}")
    return 0
