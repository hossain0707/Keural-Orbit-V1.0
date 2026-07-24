#!/usr/bin/env python
"""Stage 4/6 — launch ms-swift LoRA SFT (pilot or full) with hard safety gates.

Design (see VERIFIED-FACTS.md §5):
- Refuses to launch unless verify_lora_targets.py --pre passes for the SAME config
  (require_target_verification; deliberately not skippable from the CLI).
- Validates every swift_args key against ms-swift's real SftArguments dataclass —
  a renamed/dropped argument aborts BEFORE burning GPU hours (this exact drift
  happened: train_type -> tuner_type in ms-swift 4.4.0).
- Resumable: --resume finds the newest valid checkpoint (trainer_state.json present)
  and reuses its run dir with --resume_from_checkpoint.
- After a successful run, re-verifies the saved adapter (verify_lora_targets.py
  --checkpoint) so a silently-wrong adapter never proceeds to merge/publish.

Usage:
  python scripts/train.py --config configs/train_pilot.yaml [--dry-run] [--resume]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


# ------------------------------------------------------------------ arg building


def validate_swift_keys(swift_args: dict, logger) -> None:
    """Every key must be a real ms-swift SftArguments field. Abort otherwise."""
    import dataclasses

    from swift.arguments import SftArguments  # heavy import, but CPU-only

    fields = {f.name for f in dataclasses.fields(SftArguments)}
    unknown = sorted(k for k in swift_args if k not in fields)
    if unknown:
        raise common.PipelineError(
            f"swift_args key(s) {unknown} are not valid ms-swift 4.4.0 SftArguments "
            "fields — the installed ms-swift may have renamed them (this pipeline "
            "already hit train_type->tuner_type). Fix the config; do not guess."
        )
    logger.info("all %d swift_args keys validated against SftArguments", len(swift_args))


def to_cli_value(v) -> list[str]:
    if isinstance(v, bool):
        return ["true" if v else "false"]
    if isinstance(v, (list, tuple)):
        out: list[str] = []
        for item in v:
            out.extend(to_cli_value(item))
        return out
    return [str(v)]


PATH_KEYS = {"dataset", "val_dataset", "resume_from_checkpoint", "output_dir"}


def build_command(cfg: dict, run_dir: Path, resume_ckpt: Path | None, logger) -> list[str]:
    # Invoked as `venv_python() -m swift.cli.main sft ...`, NOT venv/bin/swift —
    # that console-script wrapper's shebang hardcodes the real (space-containing)
    # venv path. swift.cli.main.cli_main() re-execs the actual sft.py using
    # sys.executable, so launching it via the space-free symlinked python keeps
    # that fix intact through to the real training subprocess. See
    # common.venv_python() for the full explanation.
    swift_args = dict(cfg["swift_args"])
    swift_args["output_dir"] = str(run_dir)
    swift_args["add_version"] = False  # we version run dirs ourselves
    if resume_ckpt is not None:
        swift_args["resume_from_checkpoint"] = str(resume_ckpt)

    validate_swift_keys(swift_args, logger)

    cmd = [str(common.venv_python()), "-m", "swift.cli.main", "sft"]
    for key, val in swift_args.items():
        if val is None:
            continue
        vals = to_cli_value(val)
        if key in PATH_KEYS:
            vals = [str(common.resolve_path(cfg, v)) for v in vals]
        cmd.append(f"--{key}")
        cmd.extend(vals)
    return cmd


# ------------------------------------------------------------------ verification


def run_pre_verification(train_config: str, logger) -> None:
    script = common.project_root() / "scripts" / "verify_lora_targets.py"
    rc = subprocess.run(
        [sys.executable, str(script), "--pre", "--train-config", train_config],
        env=common.child_env(), cwd=common.subprocess_cwd(),
    ).returncode
    if rc != 0:
        raise common.PipelineError(
            "pre-training LoRA target verification FAILED — refusing to train. "
            "See logs/verify/ for the report. This gate exists because MoE LoRA "
            "mis-targeting fails silently (router trained / experts skipped)."
        )
    logger.info("pre-training target verification passed")


def find_latest_checkpoint(base_output_dir: Path) -> tuple[Path, Path] | None:
    """Newest (run_dir, checkpoint) that has a trainer_state.json — else None."""
    best: tuple[float, Path, Path] | None = None
    for ckpt in base_output_dir.glob("*/checkpoint-*"):
        state = ckpt / "trainer_state.json"
        if not state.exists():
            continue  # partial/corrupt checkpoint — never resume from it
        mtime = state.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, ckpt.parent, ckpt)
    return (best[1], best[2]) if best else None


def run_post_verification(run_dir: Path, train_config: str, logger) -> dict:
    found = find_latest_checkpoint(run_dir.parent)
    if not found or found[0] != run_dir:
        # look inside THIS run dir only
        ckpts = sorted(run_dir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        ckpts = [c for c in ckpts if (c / "trainer_state.json").exists()]
        if not ckpts:
            raise common.PipelineError(
                f"training exited 0 but no valid checkpoint found under {run_dir} — "
                "checkpointing is broken; do NOT proceed to merge."
            )
        last = ckpts[-1]
    else:
        last = found[1]
    script = common.project_root() / "scripts" / "verify_lora_targets.py"
    report_path = run_dir / "post_train_verification.json"
    rc = subprocess.run(
        [sys.executable, str(script), "--checkpoint", str(last),
         "--train-config", train_config, "--json-out", str(report_path)],
        env=common.child_env(), cwd=common.subprocess_cwd(),
    ).returncode
    if rc != 0:
        raise common.PipelineError(
            f"POST-TRAIN verification FAILED for {last} — the saved adapter does not "
            f"match the intended targeting. Report: {report_path}. The checkpoint is "
            "suspect; do NOT merge/publish it."
        )
    logger.info("post-train verification passed for %s", last)
    return {"checkpoint": str(last), "report": str(report_path), "passed": True}


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="configs/train_pilot.yaml or configs/train_full.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the newest valid checkpoint under output_dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the full command + env and exit (still runs the "
                         "CPU-side pre-verification)")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("train", cfg)
    common.stage_banner(logger, f"train ({cfg.get('run_name', '?')})", args.config)

    try:
        common.guard_environment(logger)

        base_out = common.resolve_path(cfg, cfg["output_dir"])
        base_out.mkdir(parents=True, exist_ok=True)

        # --- resume / run dir ---
        resume_ckpt: Path | None = None
        if args.resume:
            found = find_latest_checkpoint(base_out)
            if not found:
                raise common.PipelineError(
                    f"--resume given but no valid checkpoint (with trainer_state.json) "
                    f"under {base_out} — run without --resume for a fresh start."
                )
            run_dir, resume_ckpt = found
            logger.info("resuming run %s from %s", run_dir.name, resume_ckpt.name)
        else:
            run_dir = base_out / common.utc_stamp()
            if run_dir.exists():
                raise common.PipelineError(f"run dir collision: {run_dir}")

        # --- gates (order: cheap+safety first, GPU checks last) ---
        if cfg.get("require_target_verification", True):
            run_pre_verification(args.config, logger)
        else:
            logger.warning("require_target_verification=false — the MoE targeting "
                           "gate is DISABLED by config. You are on your own.")

        dataset_manifest = common.project_root() / "data/processed/dataset_manifest.json"
        manifest_fp = None
        if dataset_manifest.exists():
            manifest_fp = common.config_fingerprint(common.read_json(dataset_manifest))
        else:
            logger.warning("dataset manifest missing (%s) — run "
                           "scripts/build_dataset.py first; training would fail on "
                           "missing dataset files", dataset_manifest)

        cmd = build_command(cfg, run_dir, resume_ckpt, logger)
        env_extra = {
            "NPROC_PER_NODE": cfg.get("nproc_per_node", 2),
            "CUDA_VISIBLE_DEVICES": cfg.get("cuda_visible_devices", "0,1"),
        }

        if args.dry_run:
            print("[dry-run] run dir:", run_dir)
            print("[dry-run] env:", json.dumps(env_extra))
            print("[dry-run] command:")
            print("  " + " ".join(shlex.quote(c) for c in cmd))
            print("[dry-run] nothing launched. (pre-verification DID run and passed.)")
            return 0

        common.assert_free_disk(cfg, "train", logger)
        common.assert_gpus_free(cfg, logger)

        # --- record metadata BEFORE launch (crash forensics) ---
        run_dir.mkdir(parents=True, exist_ok=True)
        pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
            env=common.child_env(),
        ).stdout
        meta = {
            "run_name": cfg.get("run_name"),
            "started_utc": common.utc_stamp(),
            "config_file": args.config,
            "config_snapshot": cfg,
            "dataset_manifest_fingerprint": manifest_fp,
            "pip_freeze_sha256": hashlib.sha256(pip_freeze.encode()).hexdigest(),
            "model_revision": cfg["pipeline"]["model"]["revision"],
            "gpus_at_start": common.gpu_status(),
            "argv": cmd,
            "env": {k: str(v) for k, v in env_extra.items()},
            "resumed_from": str(resume_ckpt) if resume_ckpt else None,
        }
        common.write_json_atomic(run_dir / "run_metadata.json", meta)
        (run_dir / "pip_freeze.txt").write_text(pip_freeze)

        # --- launch ---
        rc = common.run_logged(cmd, logger, run_dir / "train.log",
                               env=common.child_env(env_extra))
        if rc != 0:
            raise common.PipelineError(
                f"swift sft exited {rc}. Full log: {run_dir / 'train.log'}. "
                f"Fix the cause, then re-run with --resume to continue from the last "
                "checkpoint."
            )

        # --- post-train verification (mandatory) ---
        post = run_post_verification(run_dir, args.config, logger)
        meta["post_train_verification"] = post
        meta["finished_utc"] = common.utc_stamp()
        common.write_json_atomic(run_dir / "run_metadata.json", meta)

        logger.info("training complete + verified: %s", run_dir)
        logger.info("next: eval the checkpoint (scripts/run_eval.py) and/or merge "
                    "(scripts/merge_lora.py --checkpoint %s)", post["checkpoint"])
        return 0

    except common.PipelineError as e:
        logger.error("FATAL: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
