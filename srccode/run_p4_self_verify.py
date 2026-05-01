"""P4 — Self-Verification (Critique-Refine, novel).

Uses <analysis>/<answer> sentinel tags; the runner extracts only <answer>.
"""
from srccode.common.runner import run

if __name__ == "__main__":
    raise SystemExit(run("p4_self_verify"))
