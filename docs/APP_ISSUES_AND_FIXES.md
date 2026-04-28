# Application Issues & Fix Guide

**Generated:** April 21, 2026  
**Scope:** Full-stack analysis of Code Smell Detection app (LangGraph workflow, LLM/RAG agents, API, Database, Frontend)

---

## 📋 Summary

| Severity | Count | Focus Areas |
|---|---|---|
| 🔴 Critical | 2 | Missing module, global mutable state |
| 🟠 High | 7 | Memory leaks, silent failures, async/sync mismatches, validation |
| 🟡 Medium | 6 | Progress tracking, retries, cache correctness |
| 🟢 Low | 5 | HTTP semantics, error details, minor polish |

---

## 🔴 CRITICAL

### C1. Missing `RAGManager` class → ChromaDB always reports unhealthy

- **File:** [src/api/routes/health.py](src/api/routes/health.py) lines 36–49
- **Symptom:** `/api/v1/status` returns `"chromadb": {"status": "unhealthy", "message": "Unavailable"}` even when ChromaDB works fine.
- **Root cause:** `from src.rag.rag_manager import RAGManager` — the file `src/rag/rag_manager.py` does **not** exist. The `ImportError` is swallowed and converted into a health-check failure.
- **Fix:**
  - Either create `src/rag/rag_manager.py` exposing a `RAGManager` thin wrapper, **or**
  - Replace the check in `_check_chromadb_availability()` with a real one against the existing modules:
    ```python
    def _check_chromadb_availability() -> bool:
        try:
            from src.rag.vector_store import VectorStore
            vs = VectorStore()
            vs.get_collection_count()  # any lightweight op
            return True
        except Exception as e:
            logger.debug(f"ChromaDB health check failed: {e}")
            return False
    ```

---

### C2. Unbounded global `analysis_state` dict → memory leak & race conditions

- **File:** [src/api/routes/analysis.py](src/api/routes/analysis.py) line 47
- **Symptom:** Long-running server grows memory indefinitely; two background tasks can race on the same key.
- **Fix:**
  - Add `created_at`/`expires_at` fields on insert.
  - Add a periodic cleanup task in `lifespan`:
    ```python
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(3600)
            now = datetime.utcnow()
            expired = [k for k, v in analysis_state.items()
                       if v.get("expires_at", now) < now]
            for k in expired:
                analysis_state.pop(k, None)

    asyncio.create_task(_cleanup_loop())
    ```
  - For concurrency, wrap writes in an `asyncio.Lock` or move state to Redis/DB.

---

## 🟠 HIGH

### H1. Progress endpoint fakes stages from elapsed time

- **File:** [src/api/routes/analysis.py](src/api/routes/analysis.py) lines 460–475
- **Symptom:** Frontend progress bar jumps/sticks; stages like `rag_retrieval` are skipped for fast runs.
- **Fix:** Have each workflow node update real state:
  ```python
  # In src/workflow/workflow_graph.py
  async def parse_code_node(state, analysis_id):
      analysis_state[analysis_id]["workflow_step"] = "parsing"
      ...
  ```
  Then return that value directly from `/progress/{id}`.

---

### H2. LLM response parser accepts findings with missing fields

- **File:** [src/llm/response_parser.py](src/llm/response_parser.py) lines 225–285
- **Symptom:** Findings with `location=null`/`severity=null` flow to the UI and crash rendering.
- **Fix:**
  ```python
  REQUIRED = ("smell_type", "location", "severity")
  cleaned = []
  for item in code_smells_data:
      if not isinstance(item, dict):
          continue
      if not all(k in item and item[k] is not None for k in REQUIRED):
          logger.warning(f"Dropping malformed finding: {item}")
          continue
      cleaned.append(item)
  code_smells_data = cleaned
  ```

---

### H3. Frontend crashes on missing/undefined findings

- **File:** [src/static/app.js](src/static/app.js) lines 450–480
- **Fix:** Guard before iterating:
  ```js
  const findings = Array.isArray(result.findings) ? result.findings : [];
  findings
    .filter(f => f && f.smell_type && f.location)
    .forEach(f => createFindingItem(f));
  ```

---

### H4. `cleanup_dependencies()` is sync but called inside async `lifespan`

- **File:** [src/api_server.py](src/api_server.py) line 62
- **Fix:**
  ```python
  import asyncio
  await asyncio.to_thread(cleanup_dependencies)
  ```

---

### H5. Missing ground-truth file silently returns `{}`

- **File:** [src/api/detection_integration.py](src/api/detection_integration.py) ~lines 120–135
- **Symptom:** F1 score is quietly `None` forever; users don't know why.
- **Fix:** Log a single clear error at startup and surface it in the status response; do not treat FileNotFoundError the same as "no ground truth available".

---

### H6. RAG cache key collisions

- **File:** [src/rag/rag_retriever.py](src/rag/rag_retriever.py) lines 68–75
- **Root cause:** `cache_key = f"{code[:50]}_{smell_type}_{top_k}"` — any two files whose first 50 chars match share cached results.
- **Fix:**
  ```python
  import hashlib
  code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
  cache_key = f"{code_hash}_{smell_type}_{top_k}"
  ```

---

### H7. Background task loses traceback on failure

- **File:** [src/api/routes/analysis.py](src/api/routes/analysis.py) `run_analysis_task` except block
- **Fix:**
  ```python
  import traceback
  analysis_state[analysis_id]["status"] = "failed"
  analysis_state[analysis_id]["error"] = str(e)
  analysis_state[analysis_id]["traceback"] = traceback.format_exc()
  logger.error("Analysis failed", exc_info=True)
  ```

---

## 🟡 MEDIUM

### M1. DB session generator doesn't guard `None`

- **File:** [src/api/dependencies.py](src/api/dependencies.py) ~lines 195–201
- **Fix:**
  ```python
  def get_database_session():
      session = None
      try:
          session = get_database_manager().get_session()
          yield session
      finally:
          if session is not None:
              try: session.close()
              except Exception as e: logger.warning(f"session close failed: {e}")
  ```

---

### M2. Ollama connection verification does not retry

- **File:** [src/llm/llm_client.py](src/llm/llm_client.py) `verify_connection` (lines 116–139)
- **Symptom:** API boots faster than Ollama; first health check fails.
- **Fix:** Retry with backoff (3× × 1s) before marking unhealthy.

---

### M3. Smell-type known-set can drift

- **File:** [src/analysis/quality_validator.py](src/analysis/quality_validator.py) ~line 145
- **Fix:** Use only `CANONICAL_SMELLS` from `src/utils/smell_catalog.py` as the source of truth; remove the hardcoded `validation_rules` union.

---

### M4. Workflow nodes don't publish progress

- **File:** [src/workflow/workflow_graph.py](src/workflow/workflow_graph.py) lines ~100–200
- **Fix:** Paired with H1 — each node writes `analysis_state[id]["workflow_step"] = "<node>"` at entry.

---

### M5. Response parser trusts bare JSON lists

- **File:** [src/llm/response_parser.py](src/llm/response_parser.py) lines 225–235
- **Fix:** Apply the same per-item validation as H2; don't set `is_valid_code = True` without checking.

---

### M6. Comparison route has same unbounded cache

- **File:** [src/api/routes/comparison.py](src/api/routes/comparison.py) line 23
- **Fix:** Reuse the TTL cleanup from C2.

---

## 🟢 LOW

### L1. `ChatOllama` fallback import path untested

- **File:** [src/analysis/code_smell_detector.py](src/analysis/code_smell_detector.py) lines 22–24
- **Fix:** Add a CI job with `pip uninstall langchain_ollama` to validate the fallback; pin one path as primary.

### L2. `/results/{id}` returns 202 on "still processing"

- **File:** [src/api/routes/analysis.py](src/api/routes/analysis.py) lines 313–322
- **Fix:** Return `200` with `status: "processing"` in payload, or use `425 Too Early`. `202` is only valid on the POST that accepted the job.

### L3. Stale results never expire

- **File:** [src/api/routes/analysis.py](src/api/routes/analysis.py) lines 480–495
- **Fix:** Reject results older than 24h; return `410 Gone`.

### L4. Frontend hides model reasoning when model is user-picked

- **File:** [src/static/app.js](src/static/app.js) lines 480–495
- **Fix:** Always show `model_reasoning` when present, regardless of whether the model was auto-selected.

### L5. LLM findings may arrive with aliases (`smell_type` vs `type`, dict vs int `location`)

- **File:** [src/llm/response_parser.py](src/llm/response_parser.py)
- **Status:** Already handled per repo memory, but add a unit test to lock the behavior (`tests/test_response_parser.py`).

---

## ✅ Recommended Fix Order

1. **C1 (RAGManager)** — small, removes a misleading health status immediately.
2. **C2 (state TTL + lock)** — required before any production/long-running use.
3. **H2 + H3** — unlock reliable end-to-end rendering.
4. **H4 + H1/M4** — async correctness and real progress tracking.
5. **H5 + H6 + H7** — observability and cache correctness.
6. Remaining M/L items — polish and hardening.

---

## 🧪 Verification Checklist

After fixes, confirm:

- [ ] `curl /api/v1/status` reports `chromadb: healthy` when ChromaDB is up.
- [ ] Submit 50 analyses; memory returns to baseline after TTL cleanup.
- [ ] Submit malformed LLM response (mock) → UI shows "no findings", no JS errors.
- [ ] `/progress/{id}` reflects real workflow nodes (`parsing → rag_retrieval → inference → validation`).
- [ ] `kill -9` the Ollama container; API health flips to `unhealthy` within 3s; after restart, flips back.
- [ ] Two concurrent analyses with identical code prefixes return different findings (cache collision fixed).
- [ ] Failed analysis exposes traceback via `/results/{id}`.
