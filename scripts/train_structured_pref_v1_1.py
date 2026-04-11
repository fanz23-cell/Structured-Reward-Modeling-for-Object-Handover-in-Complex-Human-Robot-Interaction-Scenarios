import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "PrefMMT") not in sys.path:
    sys.path.insert(0, str(ROOT / "PrefMMT"))

PREFERENCE_ID_TO_LABEL = {0: "A", 1: "B", 2: "tie"}
DIMENSION_ID_TO_LABEL = {0: "safety", 1: "comfort", 2: "social", 3: "efficiency", 4: "unclear"}
REACTION_ID_TO_LABEL = {
    0: "none",
    1: "hesitation",
    2: "avoidance",
    3: "interruption",
    4: "overreach_or_reposition",
    5: "unnatural_posture",
    6: "unclear",
}
DECOMP_BETTER_ID_TO_LABEL = {0: "A", 1: "B", 2: "tie", 3: "unclear"}
SECONDARY_ID_TO_LABEL = {
    0: "none",
    1: "safety",
    2: "comfort",
    3: "social",
    4: "efficiency",
    5: "unclear",
}
DIMENSION_ORDER = ["safety", "comfort", "social", "efficiency"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the minimal structured preference reward model on synthetic v1.1 data."
    )
    parser.add_argument("--data_dir", default="data/synthetic_v1_1")
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--val_file", default="val.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_reason", type=float, default=0.5)
    parser.add_argument("--lambda_reaction", type=float, default=0.5)
    parser.add_argument("--lambda_secondary", type=float, default=0.3)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--embd_dim", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--debug_batches", type=int, default=1)
    parser.add_argument("--disable_jit", action="store_true")
    parser.add_argument("--lambda_dominant", type=float, default=0.5)
    parser.add_argument("--lambda_decomp", type=float, default=0.5)
    parser.add_argument("--lambda_decomp_scores", type=float, default=0.3)
    parser.add_argument("--lambda_dom_sec_consistency", type=float, default=0.2)
    parser.add_argument("--lambda_decomp_consistency", type=float, default=0.2)
    parser.add_argument("--lambda_priority_veto", type=float, default=0.2)
    parser.add_argument("--output_dir", default="outputs/structured_pref_v1_1")
    return parser.parse_args()


def ensure_runtime_dependencies():
    missing = []
    for module_name in ["jax", "flax", "optax", "transformers", "ml_collections"]:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Missing runtime dependencies for structured PrefMMT training: "
            f"{joined}. Activate an environment with the PrefMMT training stack installed."
        )


def summarize_dataset(dataset, split_name):
    preference_counter = Counter()
    reason_counter = Counter()
    reaction_counter = Counter()
    dominant_counter = Counter()
    secondary_counter = Counter()
    scene_counter = Counter()
    source_counter = Counter()
    for item in dataset.samples:
        metadata = item["metadata"]
        preference_counter[PREFERENCE_ID_TO_LABEL[int(item["preference_label"])]] += 1
        reason_counter[DIMENSION_ID_TO_LABEL[int(item["reason_label"])]] += 1
        reaction_counter[REACTION_ID_TO_LABEL[int(item["reaction_label"])]] += 1
        dominant_counter[DIMENSION_ID_TO_LABEL[int(item["dominant_dimension_label"])]] += 1
        secondary_counter[SECONDARY_ID_TO_LABEL[int(item["secondary_dimension_label"])]] += 1
        scene_counter[metadata["scene_type"]] += 1
        source_counter[metadata["candidate_source_a"]] += 1
        source_counter[metadata["candidate_source_b"]] += 1

    print(f"\n{split_name} label summary")
    print(f"  preference_label: {dict(sorted(preference_counter.items()))}")
    print(f"  reason_label: {dict(sorted(reason_counter.items()))}")
    print(f"  reaction_label: {dict(sorted(reaction_counter.items()))}")
    print(f"  dominant_dimension: {dict(sorted(dominant_counter.items()))}")
    print(f"  secondary_dimension: {dict(sorted(secondary_counter.items()))}")
    print(f"  scene_type: {dict(sorted(scene_counter.items()))}")
    print(f"  candidate_source: {dict(sorted(source_counter.items()))}")


def split_numeric_and_meta(batch):
    numeric_batch = {}
    meta_batch = {}
    for key, value in batch.items():
        if key in {"sample_id", "metadata"}:
            meta_batch[key] = value
        else:
            numeric_batch[key] = value
    return numeric_batch, meta_batch


def build_batch(dataset, indices):
    batch = dataset.get_batch(indices)
    numeric_batch, meta_batch = split_numeric_and_meta(batch)
    return numeric_batch, meta_batch


def iter_minibatches(dataset, batch_size, shuffle, rng):
    indices = np.arange(len(dataset))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def decode_dimension_better(values):
    return {name: DECOMP_BETTER_ID_TO_LABEL[int(values[idx])] for idx, name in enumerate(DIMENSION_ORDER)}


def structured_preference_from_scores(score_preds, safety_veto_margin=0.5, safety_veto_scale=2.0):
    weights = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    weights = weights / weights.sum()
    diffs = score_preds[:, 0] - score_preds[:, 1]
    weighted = float(np.sum(diffs * weights))
    safety_diff = float(diffs[0])
    veto_term = safety_veto_scale * (
        max(0.0, safety_diff - safety_veto_margin) - max(0.0, -safety_diff - safety_veto_margin)
    )
    structured_logit = weighted + veto_term
    if structured_logit > 0:
        return "A", round(structured_logit, 4)
    if structured_logit < 0:
        return "B", round(structured_logit, 4)
    return "tie", round(structured_logit, 4)


def decode_prediction_bundle(predictions, sample_index):
    score_diff = (
        predictions["decomp_score_preds"][sample_index, :, 0]
        - predictions["decomp_score_preds"][sample_index, :, 1]
    )
    score_order_pred = np.where(np.abs(score_diff) <= 0.1, 2, np.where(score_diff > 0, 0, 1))
    structured_pref_pred, structured_pref_logit = structured_preference_from_scores(
        predictions["decomp_score_preds"][sample_index]
    )
    return {
        "preference_pred": PREFERENCE_ID_TO_LABEL[int(np.argmax(predictions["preference_logits"][sample_index]))],
        "reason_pred": DIMENSION_ID_TO_LABEL[int(np.argmax(predictions["reason_logits"][sample_index]))],
        "reaction_pred": REACTION_ID_TO_LABEL[int(np.argmax(predictions["reaction_logits"][sample_index]))],
        "dominant_dimension_pred": DIMENSION_ID_TO_LABEL[
            int(np.argmax(predictions["dominant_dimension_logits"][sample_index]))
        ],
        "secondary_dimension_pred": SECONDARY_ID_TO_LABEL[
            int(np.argmax(predictions["secondary_dimension_logits"][sample_index]))
        ],
        "decomp_pred": decode_dimension_better(
            np.argmax(predictions["decomp_better_logits"][sample_index], axis=-1)
        ),
        "decomp_score_order_pred": decode_dimension_better(score_order_pred),
        "decomp_score_pred": {
            name: np.asarray(predictions["decomp_score_preds"][sample_index][idx]).round(3).tolist()
            for idx, name in enumerate(DIMENSION_ORDER)
        },
        "structured_pref_pred": structured_pref_pred,
        "structured_pref_logit": structured_pref_logit,
    }


def print_debug_batch(meta_batch, numeric_batch, predictions, limit=1):
    count = min(limit, len(meta_batch["sample_id"]))
    for idx in range(count):
        metadata = meta_batch["metadata"][idx]
        decoded_pred = decode_prediction_bundle(predictions, idx)
        print(
            "debug sample",
            {
                "sample_id": meta_batch["sample_id"][idx],
                "scene_type": metadata["scene_type"],
                "candidate_source_a": metadata["candidate_source_a"],
                "candidate_source_b": metadata["candidate_source_b"],
                "preference_gt": PREFERENCE_ID_TO_LABEL[int(numeric_batch["preference_label"][idx])],
                "preference_pred": decoded_pred["preference_pred"],
                "structured_pref_pred": decoded_pred["structured_pref_pred"],
                "structured_pref_logit": decoded_pred["structured_pref_logit"],
                "reason_gt": DIMENSION_ID_TO_LABEL[int(numeric_batch["reason_label"][idx])],
                "reason_pred": decoded_pred["reason_pred"],
                "reaction_gt": REACTION_ID_TO_LABEL[int(numeric_batch["reaction_label"][idx])],
                "reaction_pred": decoded_pred["reaction_pred"],
                "dominant_dimension_gt": DIMENSION_ID_TO_LABEL[
                    int(numeric_batch["dominant_dimension_label"][idx])
                ],
                "dominant_dimension_pred": decoded_pred["dominant_dimension_pred"],
                "secondary_dimension_gt": SECONDARY_ID_TO_LABEL[
                    int(numeric_batch["secondary_dimension_label"][idx])
                ],
                "secondary_dimension_pred": decoded_pred["secondary_dimension_pred"],
                "decomp_gt": decode_dimension_better(numeric_batch["decomp_better_labels"][idx]),
                "decomp_pred": decoded_pred["decomp_pred"],
                "decomp_score_order_pred": decoded_pred["decomp_score_order_pred"],
                "decomp_score_gt": {
                    name: np.asarray(numeric_batch["decomp_score_targets"][idx][dim_idx]).round(3).tolist()
                    for dim_idx, name in enumerate(DIMENSION_ORDER)
                },
                "decomp_score_pred": decoded_pred["decomp_score_pred"],
            },
        )


def summarize_metrics(metrics):
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def save_checkpoint(path, model, epoch, metrics, config):
    import jax

    payload = {
        "epoch": epoch,
        "metrics": metrics,
        "config": dict(config),
        "train_params": jax.device_get(model.train_params),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(payload, fh)


def main():
    args = parse_args()
    if args.device == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.disable_jit:
        os.environ["JAX_DISABLE_JIT"] = "1"

    ensure_runtime_dependencies()

    import transformers
    from ml_collections import ConfigDict

    from JaxPref.PrefMMT import PrefMMT
    from JaxPref.data.structured_pref_dataset import StructuredPrefDataset
    from JaxPref.jax_utils import batch_to_jax, init_rng
    from flaxmodels.flaxmodels.gpt2.trajectory_gpt2 import TransRewardModel

    train_path = Path(args.data_dir) / args.train_file
    val_path = Path(args.data_dir) / args.val_file
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val file not found: {val_path}")

    train_dataset = StructuredPrefDataset(str(train_path), max_seq_len=args.max_seq_len)
    val_dataset = StructuredPrefDataset(str(val_path), max_seq_len=args.max_seq_len)

    print(f"Loaded train dataset: {len(train_dataset)} samples")
    print(f"Loaded val dataset: {len(val_dataset)} samples")
    print(f"Observation dim: {train_dataset.observation_dim}")
    print(f"Action dim: {train_dataset.action_dim}")
    summarize_dataset(train_dataset, "train")
    summarize_dataset(val_dataset, "val")

    np.random.seed(args.seed)
    init_rng(args.seed)
    rng = np.random.default_rng(args.seed)

    config = ConfigDict(PrefMMT.get_default_config())
    config.trans_lr = args.lr
    config.lambda_reason = args.lambda_reason
    config.lambda_reaction = args.lambda_reaction
    config.lambda_dominant = args.lambda_dominant
    config.lambda_secondary = args.lambda_secondary
    config.lambda_decomp = args.lambda_decomp
    config.lambda_decomp_scores = args.lambda_decomp_scores
    config.lambda_dom_sec_consistency = args.lambda_dom_sec_consistency
    config.lambda_decomp_consistency = args.lambda_decomp_consistency
    config.lambda_priority_veto = args.lambda_priority_veto
    config.n_layer = args.n_layer
    config.embd_dim = args.embd_dim
    config.n_embd = args.embd_dim
    config.n_head = args.n_head
    config.warmup_steps = 1
    steps_per_epoch = max(1, int(np.ceil(len(train_dataset) / args.batch_size)))
    config.total_steps = max(1, args.epochs * steps_per_epoch)

    hf_config = transformers.GPT2Config(**config)
    trans = TransRewardModel(
        config=hf_config,
        observation_dim=train_dataset.observation_dim,
        action_dim=train_dataset.action_dim,
        activation="relu",
        activation_final="none",
    )
    model = PrefMMT(config, trans)
    output_dir = Path(args.output_dir)
    latest_ckpt = output_dir / "latest.pkl"
    best_ckpt = output_dir / "best.pkl"
    best_pref_acc = -np.inf
    best_val_loss = np.inf

    for epoch in range(1, args.epochs + 1):
        train_metrics = defaultdict(list)
        for batch_indices in iter_minibatches(train_dataset, args.batch_size, shuffle=True, rng=rng):
            numeric_batch, meta_batch = build_batch(train_dataset, batch_indices)
            jax_batch = batch_to_jax(numeric_batch)
            metrics = model.train(jax_batch)
            for key, value in metrics.items():
                train_metrics[key].append(float(value))

        val_metrics = defaultdict(list)
        first_val_batch = None
        for batch_indices in iter_minibatches(val_dataset, args.batch_size, shuffle=False, rng=rng):
            numeric_batch, meta_batch = build_batch(val_dataset, batch_indices)
            if first_val_batch is None:
                first_val_batch = (numeric_batch, meta_batch)
            jax_batch = batch_to_jax(numeric_batch)
            metrics = model.evaluation(jax_batch)
            for key, value in metrics.items():
                val_metrics[key].append(float(value))

        train_summary = summarize_metrics(train_metrics)
        val_summary = summarize_metrics(val_metrics)
        if first_val_batch is not None:
            debug_numeric_batch, debug_meta_batch = first_val_batch
            debug_predictions = model.predict(batch_to_jax(debug_numeric_batch))
            debug_predictions = {key: np.asarray(value) for key, value in debug_predictions.items()}
            print_debug_batch(debug_meta_batch, debug_numeric_batch, debug_predictions, limit=args.debug_batches)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(
            "train",
            {
                "preference_loss": round(train_summary.get("trans_loss", 0.0), 6),
                "reason_loss": round(train_summary.get("reason_loss", 0.0), 6),
                "reaction_loss": round(train_summary.get("reaction_loss", 0.0), 6),
                "dominant_loss": round(train_summary.get("dominant_loss", 0.0), 6),
                "secondary_loss": round(train_summary.get("secondary_loss", 0.0), 6),
                "decomp_loss": round(train_summary.get("decomp_loss", 0.0), 6),
                "decomp_score_loss": round(train_summary.get("decomp_score_loss", 0.0), 6),
                "dom_sec_consistency_loss": round(
                    train_summary.get("dominant_secondary_consistency_loss", 0.0), 6
                ),
                "decomp_consistency_loss": round(
                    train_summary.get("decomp_consistency_loss", 0.0), 6
                ),
                "priority_veto_loss": round(train_summary.get("priority_veto_loss", 0.0), 6),
                "preference_accuracy": round(train_summary.get("preference_accuracy", 0.0), 6),
                "reason_accuracy": round(train_summary.get("reason_accuracy", 0.0), 6),
                "reaction_accuracy": round(train_summary.get("reaction_accuracy", 0.0), 6),
                "dominant_accuracy": round(train_summary.get("dominant_accuracy", 0.0), 6),
                "secondary_accuracy": round(train_summary.get("secondary_accuracy", 0.0), 6),
                "decomp_accuracy": round(train_summary.get("decomp_accuracy", 0.0), 6),
                "decomp_score_mae": round(train_summary.get("decomp_score_mae", 0.0), 6),
                "decomp_score_rmse": round(train_summary.get("decomp_score_rmse", 0.0), 6),
                "decomp_score_order_accuracy": round(
                    train_summary.get("decomp_score_order_accuracy", 0.0), 6
                ),
                "priority_veto_accuracy": round(
                    train_summary.get("priority_veto_accuracy", 0.0), 6
                ),
                "safety_accuracy": round(train_summary.get("safety_accuracy", 0.0), 6),
                "comfort_accuracy": round(train_summary.get("comfort_accuracy", 0.0), 6),
                "social_accuracy": round(train_summary.get("social_accuracy", 0.0), 6),
                "efficiency_accuracy": round(train_summary.get("efficiency_accuracy", 0.0), 6),
                "safety_score_mae": round(train_summary.get("safety_score_mae", 0.0), 6),
                "comfort_score_mae": round(train_summary.get("comfort_score_mae", 0.0), 6),
                "social_score_mae": round(train_summary.get("social_score_mae", 0.0), 6),
                "efficiency_score_mae": round(train_summary.get("efficiency_score_mae", 0.0), 6),
            },
        )
        print(
            "val",
            {
                "preference_loss": round(val_summary.get("eval_trans_loss", 0.0), 6),
                "reason_loss": round(val_summary.get("eval_reason_loss", 0.0), 6),
                "reaction_loss": round(val_summary.get("eval_reaction_loss", 0.0), 6),
                "dominant_loss": round(val_summary.get("eval_dominant_loss", 0.0), 6),
                "secondary_loss": round(val_summary.get("eval_secondary_loss", 0.0), 6),
                "decomp_loss": round(val_summary.get("eval_decomp_loss", 0.0), 6),
                "decomp_score_loss": round(val_summary.get("eval_decomp_score_loss", 0.0), 6),
                "dom_sec_consistency_loss": round(
                    val_summary.get("eval_dominant_secondary_consistency_loss", 0.0), 6
                ),
                "decomp_consistency_loss": round(
                    val_summary.get("eval_decomp_consistency_loss", 0.0), 6
                ),
                "priority_veto_loss": round(val_summary.get("eval_priority_veto_loss", 0.0), 6),
                "preference_accuracy": round(val_summary.get("eval_preference_accuracy", 0.0), 6),
                "reason_accuracy": round(val_summary.get("eval_reason_accuracy", 0.0), 6),
                "reaction_accuracy": round(val_summary.get("eval_reaction_accuracy", 0.0), 6),
                "dominant_accuracy": round(val_summary.get("eval_dominant_accuracy", 0.0), 6),
                "secondary_accuracy": round(val_summary.get("eval_secondary_accuracy", 0.0), 6),
                "decomp_accuracy": round(val_summary.get("eval_decomp_accuracy", 0.0), 6),
                "decomp_score_mae": round(val_summary.get("eval_decomp_score_mae", 0.0), 6),
                "decomp_score_rmse": round(val_summary.get("eval_decomp_score_rmse", 0.0), 6),
                "decomp_score_order_accuracy": round(
                    val_summary.get("eval_decomp_score_order_accuracy", 0.0), 6
                ),
                "priority_veto_accuracy": round(
                    val_summary.get("eval_priority_veto_accuracy", 0.0), 6
                ),
                "safety_accuracy": round(val_summary.get("eval_safety_accuracy", 0.0), 6),
                "comfort_accuracy": round(val_summary.get("eval_comfort_accuracy", 0.0), 6),
                "social_accuracy": round(val_summary.get("eval_social_accuracy", 0.0), 6),
                "efficiency_accuracy": round(val_summary.get("eval_efficiency_accuracy", 0.0), 6),
                "safety_score_mae": round(val_summary.get("eval_safety_score_mae", 0.0), 6),
                "comfort_score_mae": round(val_summary.get("eval_comfort_score_mae", 0.0), 6),
                "social_score_mae": round(val_summary.get("eval_social_score_mae", 0.0), 6),
                "efficiency_score_mae": round(val_summary.get("eval_efficiency_score_mae", 0.0), 6),
            },
        )
        save_checkpoint(latest_ckpt, model, epoch, {"train": train_summary, "val": val_summary}, config)
        print(f"latest checkpoint: {latest_ckpt}")
        val_pref_acc = val_summary.get("eval_preference_accuracy", 0.0)
        val_total_loss = val_summary.get("eval_trans_loss", np.inf)
        if (val_pref_acc > best_pref_acc) or (
            np.isclose(val_pref_acc, best_pref_acc) and val_total_loss < best_val_loss
        ):
            best_pref_acc = val_pref_acc
            best_val_loss = val_total_loss
            save_checkpoint(best_ckpt, model, epoch, {"train": train_summary, "val": val_summary}, config)
            print(f"best checkpoint updated: {best_ckpt}")
        else:
            print(f"best checkpoint: {best_ckpt}")

    print("\nStructured preference training smoke test completed.")


if __name__ == "__main__":
    main()
