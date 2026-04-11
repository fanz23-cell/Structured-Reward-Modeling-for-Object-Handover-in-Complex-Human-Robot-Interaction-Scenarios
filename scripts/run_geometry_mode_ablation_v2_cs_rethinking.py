import argparse
import json
import pickle
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


MODE_SPECS = {
    "fixed": {
        "geometry_mode": "fixed",
        "geometry_parameterization_mode": "global",
    },
    "global": {
        "geometry_mode": "trainable",
        "geometry_parameterization_mode": "global",
    },
    "contextual": {
        "geometry_mode": "trainable",
        "geometry_parameterization_mode": "contextual",
    },
}

SUMMARY_METRICS = [
    ("eval_preference_accuracy", "val_preference_accuracy"),
    ("eval_structured_preference_accuracy_geom", "val_structured_pref_acc_geom"),
    ("eval_structured_preference_accuracy_final", "val_structured_pref_acc_final"),
    ("eval_comfort_better_accuracy", "val_comfort_better_accuracy"),
    ("eval_safety_better_accuracy", "val_safety_better_accuracy"),
    ("eval_comfort_score_mae", "val_comfort_score_mae"),
    ("eval_safety_score_mae", "val_safety_score_mae"),
    ("eval_score_alignment_mae", "val_score_alignment_mae"),
    ("eval_score_alignment_rmse", "val_score_alignment_rmse"),
    ("eval_geometry_prior_loss", "val_geometry_prior_loss"),
    ("eval_safety_subreason_accuracy", "val_safety_subreason_accuracy"),
    ("eval_comfort_subreason_accuracy", "val_comfort_subreason_accuracy"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fixed/global/contextual geometry ablations for V2 structured reward."
    )
    parser.add_argument("--data_dir", default="data/synthetic_v2_cs_rethinking")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--disable_jit", action="store_true")
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--output_root", default="outputs/structured_pref_v2_cs_rethinking_ablation")
    parser.add_argument("--summary_json", default="outputs/structured_pref_v2_cs_rethinking_ablation/summary.json")
    parser.add_argument("--summary_md", default="outputs/structured_pref_v2_cs_rethinking_ablation/summary.md")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--modes", nargs="+", choices=sorted(MODE_SPECS.keys()), default=["fixed", "global", "contextual"])
    return parser.parse_args()


def load_checkpoint_metrics(path: Path):
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    return payload["metrics"]


def aggregate(values):
    if not values:
        return {"mean": None, "std": None}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std}


def run_single(args, mode_name: str, seed: int):
    spec = MODE_SPECS[mode_name]
    run_dir = Path(args.output_root) / f"{mode_name}_seed{seed}"
    cmd = [
        args.python_exe,
        "scripts/train_structured_pref_v2_cs_rethinking.py",
        "--data_dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--max_seq_len",
        str(args.max_seq_len),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--geometry_mode",
        spec["geometry_mode"],
        "--geometry_parameterization_mode",
        spec["geometry_parameterization_mode"],
        "--output_dir",
        str(run_dir),
    ]
    if args.disable_jit:
        cmd.append("--disable_jit")

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    latest_ckpt = run_dir / "latest.pkl"
    if result.returncode != 0:
        return {
            "ok": False,
            "mode": mode_name,
            "seed": seed,
            "cmd": cmd,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    metrics = load_checkpoint_metrics(latest_ckpt)
    return {
        "ok": True,
        "mode": mode_name,
        "seed": seed,
        "cmd": cmd,
        "stdout_tail": result.stdout.splitlines()[-20:],
        "metrics": metrics,
        "checkpoint": str(latest_ckpt),
    }


def build_summary(results):
    grouped = {mode: [] for mode in MODE_SPECS}
    for item in results:
        if item["ok"]:
            grouped[item["mode"]].append(item)

    summary = {"runs": results, "aggregate": {}}
    for mode_name, runs in grouped.items():
        metric_summary = {}
        for raw_key, alias in SUMMARY_METRICS:
            values = [float(run["metrics"]["val"].get(raw_key, 0.0)) for run in runs]
            metric_summary[alias] = aggregate(values)
        summary["aggregate"][mode_name] = {
            "num_runs": len(runs),
            "metrics": metric_summary,
        }
    return summary


def render_markdown(summary):
    lines = [
        "# V2 Geometry Mode Ablation",
        "",
        "| Mode | Runs | Val Pref Acc | Val Structured Geom Acc | Val Structured Final Acc | Val Comfort Better Acc | Val Safety Better Acc | Val Comfort MAE | Val Safety MAE | Val Score Align MAE | Val Score Align RMSE | Val Safety Subreason Acc | Val Comfort Subreason Acc | Val Geometry Prior |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode_name in ["fixed", "global", "contextual"]:
        info = summary["aggregate"].get(mode_name, {})
        metrics = info.get("metrics", {})

        def fmt(name):
            stat = metrics.get(name, {})
            mean = stat.get("mean")
            std = stat.get("std")
            if mean is None:
                return "n/a"
            return f"{mean:.3f} +/- {std:.3f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    mode_name,
                    str(info.get("num_runs", 0)),
                    fmt("val_preference_accuracy"),
                    fmt("val_structured_pref_acc_geom"),
                    fmt("val_structured_pref_acc_final"),
                    fmt("val_comfort_better_accuracy"),
                    fmt("val_safety_better_accuracy"),
                    fmt("val_comfort_score_mae"),
                    fmt("val_safety_score_mae"),
                    fmt("val_score_alignment_mae"),
                    fmt("val_score_alignment_rmse"),
                    fmt("val_safety_subreason_accuracy"),
                    fmt("val_comfort_subreason_accuracy"),
                    fmt("val_geometry_prior_loss"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `fixed` = manually fixed geometry ruler baseline.",
            "- `global` = one shared learned comfort/safety ruler for all samples.",
            "- `contextual` = learned base ruler plus per-sample context-conditioned adjustment.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    print("Running geometry mode ablation")
    print(f"  data_dir: {args.data_dir}")
    print(f"  seeds: {args.seeds}")
    print(f"  modes: {args.modes}")
    for mode_name in args.modes:
        spec = MODE_SPECS[mode_name]
        print(
            f"\nmode={mode_name} geometry_mode={spec['geometry_mode']} "
            f"geometry_parameterization_mode={spec['geometry_parameterization_mode']}"
        )
        for seed in args.seeds:
            print(f"  seed={seed} running...")
            result = run_single(args, mode_name, seed)
            results.append(result)
            if result["ok"]:
                val_metrics = result["metrics"]["val"]
                print(
                    "   "
                    + str(
                        {
                            "val_preference_accuracy": round(float(val_metrics.get("eval_preference_accuracy", 0.0)), 6),
                            "val_structured_pref_acc_geom": round(
                                float(val_metrics.get("eval_structured_preference_accuracy_geom", 0.0)), 6
                            ),
                            "val_structured_pref_acc_final": round(
                                float(val_metrics.get("eval_structured_preference_accuracy_final", 0.0)), 6
                            ),
                            "val_comfort_score_mae": round(
                                float(val_metrics.get("eval_comfort_score_mae", 0.0)), 6
                            ),
                            "val_safety_score_mae": round(
                                float(val_metrics.get("eval_safety_score_mae", 0.0)), 6
                            ),
                        }
                    )
                )
            else:
                print("   failed")
                print(result["stderr"].strip())

    summary = build_summary(results)
    summary_json_path = Path(args.summary_json)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    summary_md_path = Path(args.summary_md)
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.write_text(render_markdown(summary), encoding="utf-8")

    print(f"\nSummary JSON: {summary_json_path}")
    print(f"Summary MD: {summary_md_path}")
    print(summary_md_path.read_text(encoding='utf-8'))


if __name__ == "__main__":
    main()
