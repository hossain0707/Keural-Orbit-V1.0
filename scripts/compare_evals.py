#!/usr/bin/env python
"""Stage 6/9 — diff two eval summaries (baseline vs candidate) with regression gates.

Reads the summary.json files written by run_eval.py, aligns tasks/metrics, and emits
comparison.md + comparison.json. General-suite regressions beyond the threshold flag
the run (exit code 2) — the guide (§1) calls the general suite "your regression
tripwire".

Usage:
  python scripts/compare_evals.py --baseline base --candidate keural
  python scripts/compare_evals.py --baseline results/base/<ts>/summary.json \
                                  --candidate results/keural/<ts>/summary.json
Tags resolve to the NEWEST run under results/<tag>/. Exit codes: 0 ok, 1 error,
2 regression(s) found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Headline metric per task, tried in order; falls back to first shared numeric metric.
PRIMARY_METRIC_PRIORITY = [
    "acc,none", "exact_match,none", "exact_match,strict-match",
    "exact_match,flexible-extract", "pass@1,create_test", "pass@1,none",
    "acc_norm,none", "f1,none",
]
# Absolute drop in the primary metric (0-1 scale) that counts as a regression.
REGRESSION_THRESHOLD = 0.01  # 1.0 point


def resolve_summary(ref: str) -> Path:
    """A tag ('base') -> newest results/<tag>/*/summary.json; a path -> itself."""
    p = Path(ref)
    if not p.is_absolute():
        p = common.project_root() / p
    if p.is_file():
        return p
    tag_dir = common.project_root() / "results" / ref
    candidates = sorted(tag_dir.glob("*/summary.json"))
    if not candidates:
        raise common.PipelineError(
            f"{ref!r} is neither a summary.json path nor a tag with runs under "
            f"{tag_dir} — run scripts/run_eval.py first."
        )
    return candidates[-1]  # utc_stamp dirs sort lexicographically = chronologically


def primary_metric(metrics_a: dict, metrics_b: dict) -> str | None:
    shared = [m for m in metrics_a if m in metrics_b]
    for m in PRIMARY_METRIC_PRIORITY:
        if m in shared:
            return m
    numeric = [m for m in shared if "stderr" not in m]
    return numeric[0] if numeric else None


def compare(base: dict, cand: dict) -> dict:
    """Align tasks and compute deltas. Only numeric benchmark tasks participate;
    the 'domain' entry has its own shape and is surfaced separately."""
    b_suites, c_suites = base.get("suites", {}), cand.get("suites", {})
    task_suite = {**base.get("task_suites", {}), **cand.get("task_suites", {})}
    rows, regressions, improvements = [], [], []

    tasks = sorted((set(b_suites) | set(c_suites)) - {"domain"})
    for task in tasks:
        if task == "domain":
            continue
        b, c = b_suites.get(task), c_suites.get(task)
        suite = task_suite.get(task, "?")
        if not isinstance(b, dict) or not isinstance(c, dict) or not b or not c:
            rows.append({"task": task, "suite": suite, "metric": "n/a",
                         "baseline": None, "candidate": None, "delta": None,
                         "flag": "missing in " + ("baseline" if not b else "candidate")})
            continue
        m = primary_metric(b, c)
        if m is None:
            rows.append({"task": task, "suite": suite, "metric": "n/a",
                         "baseline": None, "candidate": None, "delta": None,
                         "flag": "no shared numeric metric"})
            continue
        delta = c[m] - b[m]
        flag = ""
        if suite == "general" and delta < -REGRESSION_THRESHOLD:
            flag = "REGRESSION"
            regressions.append(task)
        elif delta > REGRESSION_THRESHOLD:
            flag = "improved"
            improvements.append(task)
        rows.append({"task": task, "suite": suite, "metric": m,
                     "baseline": b[m], "candidate": c[m],
                     "delta": delta, "flag": flag})

    domain = {}
    for label, s in (("baseline", b_suites), ("candidate", c_suites)):
        d = s.get("domain")
        domain[label] = d if d else "absent"

    return {
        "baseline": {"tag": base.get("tag"), "model": base.get("model"),
                     "run_dir": base.get("run_dir"), "smoke": base.get("smoke")},
        "candidate": {"tag": cand.get("tag"), "model": cand.get("model"),
                      "run_dir": cand.get("run_dir"), "smoke": cand.get("smoke")},
        "regression_threshold": REGRESSION_THRESHOLD,
        "rows": rows,
        "regressions": regressions,
        "improvements": improvements,
        "domain": domain,
        "verdict": ("REGRESSIONS FOUND — do not ship without investigating"
                    if regressions else "no general-suite regressions"),
    }


def to_markdown(cmp: dict) -> str:
    b, c = cmp["baseline"], cmp["candidate"]
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
    lines = [
        "# Eval comparison",
        "",
        f"- **Baseline:** `{b['tag']}` ({b['model']}) — {b['run_dir']}"
        + (" **[SMOKE — not comparable at full fidelity]**" if b.get("smoke") else ""),
        f"- **Candidate:** `{c['tag']}` ({c['model']}) — {c['run_dir']}"
        + (" **[SMOKE — not comparable at full fidelity]**" if c.get("smoke") else ""),
        f"- **Verdict:** {cmp['verdict']}",
        "",
        "| task | suite | metric | baseline | candidate | Δ | |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in sorted(cmp["rows"], key=lambda r: (r["suite"], r["task"])):
        delta = "n/a" if r["delta"] is None else f"{r['delta']:+.4f}"
        lines.append(f"| {r['task']} | {r['suite']} | {r['metric']} | "
                     f"{fmt(r['baseline'])} | {fmt(r['candidate'])} | {delta} | "
                     f"{r['flag']} |")
    lines += ["", "## Domain eval", ""]
    for label in ("baseline", "candidate"):
        d = cmp["domain"][label]
        if isinstance(d, dict):
            lines.append(f"- {label}: n={d.get('n_examples')}, "
                         f"exact={d.get('exact_match_rate')}, "
                         f"contains={d.get('contains_rate')} "
                         f"(heuristic only — needs human/judge grading)")
        else:
            lines.append(f"- {label}: {d}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="tag or path to summary.json")
    ap.add_argument("--candidate", required=True, help="tag or path to summary.json")
    ap.add_argument("--out", default=None,
                    help="output dir (default results/comparisons/<stamp>)")
    args = ap.parse_args()

    cfg = common.load_config(None)
    logger = common.setup_logging("compare", cfg)
    common.stage_banner(logger, "compare_evals", None)

    try:
        b_path, c_path = resolve_summary(args.baseline), resolve_summary(args.candidate)
        logger.info("baseline : %s", b_path)
        logger.info("candidate: %s", c_path)
        base, cand = common.read_json(b_path), common.read_json(c_path)
        if base.get("smoke") != cand.get("smoke"):
            logger.warning("comparing a smoke run against a full run — deltas are NOT "
                           "meaningful; rerun both at the same fidelity")

        cmp = compare(base, cand)
        out_dir = Path(args.out) if args.out else (
            common.project_root() / "results" / "comparisons" / common.utc_stamp())
        out_dir.mkdir(parents=True, exist_ok=True)
        common.write_json_atomic(out_dir / "comparison.json", cmp)
        md = to_markdown(cmp)
        (out_dir / "comparison.md").write_text(md, encoding="utf-8")
        logger.info("written: %s", out_dir)
        print(md)

        if cmp["regressions"]:
            logger.error("general-suite regressions: %s", cmp["regressions"])
            return 2
        return 0
    except common.PipelineError as e:
        logger.error("FATAL: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
