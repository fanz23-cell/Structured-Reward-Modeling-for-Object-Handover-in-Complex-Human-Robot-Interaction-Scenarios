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

PREFERENCE_ID_TO_LABEL = {0: "A", 1: "B", 2: "tie", 3: "unclear"}
REASON_ID_TO_LABEL = {0: "safety", 1: "comfort", 2: "mixed", 3: "unclear"}
REACTION_ID_TO_LABEL = {
    0: "none",
    1: "hesitation",
    2: "avoidance",
    3: "interruption",
    4: "overreach_or_reposition",
    5: "unnatural_posture",
    6: "unclear",
}
BETTER_ID_TO_LABEL = {0: "A", 1: "B", 2: "tie", 3: "unclear"}
SAFETY_SUBREASON_ID_TO_LABEL = {
    0: "distance_intrusion",
    1: "speed_risk",
    2: "timing_risk",
    3: "mixed",
    4: "unclear",
}
COMFORT_SUBREASON_ID_TO_LABEL = {
    0: "reachability",
    1: "height_alignment",
    2: "front_sector_alignment",
    3: "posture_cost",
    4: "mixed",
    5: "unclear",
}
GEOMETRY_MODE_ID_TO_LABEL = {0: "fixed", 1: "trainable"}
GEOMETRY_PARAM_MODE_ID_TO_LABEL = {0: "fixed", 1: "global", 2: "contextual"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the V2 comfort/safety structured reward reranker prototype."
    )
    parser.add_argument("--data_dir", default="data/synthetic_v2_cs_rethinking")
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--val_file", default="val.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--embd_dim", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--geometry_mode", choices=["fixed", "trainable"], default="trainable")
    parser.add_argument(
        "--geometry_parameterization_mode",
        choices=["global", "contextual"],
        default="contextual",
    )
    parser.add_argument("--debug_batches", type=int, default=1)
    parser.add_argument("--disable_jit", action="store_true")
    parser.add_argument("--lambda_reason", type=float, default=0.5)
    parser.add_argument("--lambda_reaction", type=float, default=0.5)
    parser.add_argument("--lambda_comfort_better", type=float, default=0.5)
    parser.add_argument("--lambda_safety_better", type=float, default=0.5)
    parser.add_argument("--lambda_comfort_score", type=float, default=0.3)
    parser.add_argument("--lambda_safety_score", type=float, default=0.3)
    parser.add_argument("--lambda_score_alignment", type=float, default=0.1)
    parser.add_argument("--lambda_geometry_prior", type=float, default=0.02)
    parser.add_argument("--lambda_safety_subreason", type=float, default=0.2)
    parser.add_argument("--lambda_comfort_subreason", type=float, default=0.2)
    parser.add_argument("--lambda_structured_preference_consistency", type=float, default=0.2)
    parser.add_argument("--lambda_learned_to_geom_consistency", type=float, default=0.2)
    parser.add_argument("--structured_tau_safe", type=float, default=0.5)
    parser.add_argument("--structured_lambda_veto", type=float, default=4.0)
    parser.add_argument("--alpha_geom", type=float, default=0.8)
    parser.add_argument("--alpha_learned", type=float, default=0.2)
    parser.add_argument("--output_dir", default="outputs/structured_pref_v2_cs_rethinking")
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
            "Missing runtime dependencies for structured PrefMMT V2 training: "
            f"{joined}. Activate an environment with the PrefMMT training stack installed."
        )


def summarize_dataset(dataset, split_name):
    preference_counter = Counter()
    reason_counter = Counter()
    reaction_counter = Counter()
    comfort_better_counter = Counter()
    safety_better_counter = Counter()
    scene_counter = Counter()
    source_counter = Counter()
    for item in dataset.samples:
        metadata = item["metadata"]
        preference_counter[PREFERENCE_ID_TO_LABEL[int(item["preference_label"])]] += 1
        reason_counter[REASON_ID_TO_LABEL[int(item["reason_label"])]] += 1
        reaction_counter[REACTION_ID_TO_LABEL[int(item["reaction_label"])]] += 1
        comfort_better_counter[BETTER_ID_TO_LABEL[int(item["comfort_better_label"])]] += 1
        safety_better_counter[BETTER_ID_TO_LABEL[int(item["safety_better_label"])]] += 1
        scene_counter[metadata["scene_type"]] += 1
        source_counter[metadata["candidate_source_a"]] += 1
        source_counter[metadata["candidate_source_b"]] += 1

    print(f"\n{split_name} label summary")
    print(f"  preference_label: {dict(sorted(preference_counter.items()))}")
    print(f"  reason_label: {dict(sorted(reason_counter.items()))}")
    print(f"  reaction_label: {dict(sorted(reaction_counter.items()))}")
    print(f"  comfort_better_label: {dict(sorted(comfort_better_counter.items()))}")
    print(f"  safety_better_label: {dict(sorted(safety_better_counter.items()))}")
    print(f"  scene_type: {dict(sorted(scene_counter.items()))}")
    print(f"  candidate_source: {dict(sorted(source_counter.items()))}")


def split_numeric_and_meta(batch):
    numeric_batch = {}
    meta_batch = {}
    for key, value in batch.items():
        if key in {"sample_id", "metadata", "context_id", "pair_id"}:
            meta_batch[key] = value
        else:
            numeric_batch[key] = value
    return numeric_batch, meta_batch


def build_batch(dataset, indices):
    batch = dataset.get_batch(indices)
    return split_numeric_and_meta(batch)


def iter_minibatches(dataset, batch_size, shuffle, rng):
    indices = np.arange(len(dataset))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def summarize_metrics(metrics):
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def tree_to_numpy(value):
    if isinstance(value, dict):
        return {key: tree_to_numpy(item) for key, item in value.items()}
    return np.asarray(value)


def _param_value_to_python(value):
    array = np.asarray(value)
    if array.ndim > 1 and array.shape[0] >= 1:
        array = array[0]
    if array.ndim == 0:
        return round(float(array), 3)
    return np.round(array, 3).tolist()


def decode_prediction_bundle(predictions, sample_index):
    comfort_scores = np.asarray(predictions["comfort_score_preds"][sample_index]).round(3).tolist()
    safety_scores = np.asarray(predictions["safety_score_preds"][sample_index]).round(3).tolist()
    comfort_geom = np.asarray(predictions["comfort_score_geom"][sample_index]).round(3).tolist()
    safety_geom = np.asarray(predictions["safety_score_geom"][sample_index]).round(3).tolist()
    structured_geom = np.asarray(predictions["segment_score_structured_geom"][sample_index]).round(3).tolist()
    structured_learned = np.asarray(predictions["segment_score_structured_learned"][sample_index]).round(3).tolist()
    structured_final = np.asarray(predictions["segment_score_structured_final"][sample_index]).round(3).tolist()
    loser_indicator = int(np.asarray(predictions["loser_indicator"][sample_index]))
    subreason_active = bool(np.asarray(predictions["predicted_subreason_active"][sample_index]) > 0.5)
    geometry_mode_summary = predictions.get("geometry_mode_summary", {})
    geometry_mode = GEOMETRY_MODE_ID_TO_LABEL.get(
        int(np.asarray(geometry_mode_summary.get("geometry_mode_id", 0))),
        "unknown",
    )
    geometry_param_mode = GEOMETRY_PARAM_MODE_ID_TO_LABEL.get(
        int(np.asarray(geometry_mode_summary.get("geometry_parameterization_mode_id", 0))),
        "unknown",
    )
    return {
        "preference_pred": PREFERENCE_ID_TO_LABEL[int(np.argmax(predictions["overall_preference_logits"][sample_index]))],
        "structured_preference_pred_geom": PREFERENCE_ID_TO_LABEL[
            int(np.argmax(predictions["structured_preference_logits_geom"][sample_index]))
        ],
        "structured_preference_pred_learned": PREFERENCE_ID_TO_LABEL[
            int(np.argmax(predictions["structured_preference_logits_learned"][sample_index]))
        ],
        "structured_preference_pred_final": PREFERENCE_ID_TO_LABEL[
            int(np.argmax(predictions["structured_preference_logits_final"][sample_index]))
        ],
        "reason_pred": REASON_ID_TO_LABEL[int(np.argmax(predictions["reason_logits"][sample_index]))],
        "reaction_pred": REACTION_ID_TO_LABEL[int(np.argmax(predictions["reaction_logits"][sample_index]))],
        "comfort_better_pred": BETTER_ID_TO_LABEL[
            int(np.argmax(predictions["comfort_better_logits"][sample_index]))
        ],
        "safety_better_pred": BETTER_ID_TO_LABEL[
            int(np.argmax(predictions["safety_better_logits"][sample_index]))
        ],
        "safety_subreason_pred": (
            SAFETY_SUBREASON_ID_TO_LABEL[int(np.argmax(predictions["safety_subreason_logits"][sample_index]))]
            if subreason_active else None
        ),
        "comfort_subreason_pred": (
            COMFORT_SUBREASON_ID_TO_LABEL[int(np.argmax(predictions["comfort_subreason_logits"][sample_index]))]
            if subreason_active else None
        ),
        "subreason_active_pred": subreason_active,
        "loser_id_pred": ("A" if loser_indicator == 0 else "B") if subreason_active else None,
        "comfort_score_geom": {"A": comfort_geom[0], "B": comfort_geom[1]},
        "comfort_score_pred": {"A": comfort_scores[0], "B": comfort_scores[1]},
        "safety_score_geom": {"A": safety_geom[0], "B": safety_geom[1]},
        "safety_score_pred": {"A": safety_scores[0], "B": safety_scores[1]},
        "structured_segment_score_geom": {"A": structured_geom[0], "B": structured_geom[1]},
        "structured_segment_score_learned": {"A": structured_learned[0], "B": structured_learned[1]},
        "structured_segment_score_final": {"A": structured_final[0], "B": structured_final[1]},
        "comfort_zone_params": {
            key: _param_value_to_python(value[sample_index] if np.asarray(value).ndim > 0 else value)
            for key, value in predictions["comfort_zone_params"].items()
        },
        "safety_zone_params": {
            key: _param_value_to_python(value[sample_index] if np.asarray(value).ndim > 0 else value)
            for key, value in predictions["safety_zone_params"].items()
        },
        "comfort_adapter_summary": {
            key: _param_value_to_python(value[sample_index] if np.asarray(value).ndim > 0 else value)
            for key, value in predictions["comfort_adapter_summary"].items()
        },
        "safety_adapter_summary": {
            key: _param_value_to_python(value[sample_index] if np.asarray(value).ndim > 0 else value)
            for key, value in predictions["safety_adapter_summary"].items()
        },
        "shared_context_summary": {
            key: _param_value_to_python(value[sample_index] if np.asarray(value).ndim > 0 else value)
            for key, value in predictions["shared_context_summary"].items()
        },
        "geometry_mode_summary": {
            "geometry_mode": geometry_mode,
            "geometry_parameterization_mode": geometry_param_mode,
        },
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
                "overall_preference_gt": PREFERENCE_ID_TO_LABEL[int(numeric_batch["preference_label"][idx])],
                "overall_preference_pred": decoded_pred["preference_pred"],
                "structured_preference_pred_geom": decoded_pred["structured_preference_pred_geom"],
                "structured_preference_pred_learned": decoded_pred["structured_preference_pred_learned"],
                "structured_preference_pred_final": decoded_pred["structured_preference_pred_final"],
                "loser_id_gt": "A" if int(numeric_batch["preference_label"][idx]) == 1 else "B",
                "loser_id_pred": decoded_pred["loser_id_pred"],
                "subreason_active_pred": decoded_pred["subreason_active_pred"],
                "reason_gt": REASON_ID_TO_LABEL[int(numeric_batch["reason_label"][idx])],
                "reason_pred": decoded_pred["reason_pred"],
                "reaction_gt": REACTION_ID_TO_LABEL[int(numeric_batch["reaction_label"][idx])],
                "reaction_pred": decoded_pred["reaction_pred"],
                "comfort_better_gt": BETTER_ID_TO_LABEL[int(numeric_batch["comfort_better_label"][idx])],
                "comfort_better_pred": decoded_pred["comfort_better_pred"],
                "safety_better_gt": BETTER_ID_TO_LABEL[int(numeric_batch["safety_better_label"][idx])],
                "safety_better_pred": decoded_pred["safety_better_pred"],
                "comfort_score_gt": (
                    {
                        "A": round(float(numeric_batch["comfort_score_targets"][idx][0]), 3),
                        "B": round(float(numeric_batch["comfort_score_targets"][idx][1]), 3),
                    }
                    if float(np.max(numeric_batch["comfort_score_masks"][idx])) > 0
                    else None
                ),
                "comfort_score_geom": decoded_pred["comfort_score_geom"],
                "comfort_score_pred": decoded_pred["comfort_score_pred"],
                "safety_score_gt": (
                    {
                        "A": round(float(numeric_batch["safety_score_targets"][idx][0]), 3),
                        "B": round(float(numeric_batch["safety_score_targets"][idx][1]), 3),
                    }
                    if float(np.max(numeric_batch["safety_score_masks"][idx])) > 0
                    else None
                ),
                "safety_score_geom": decoded_pred["safety_score_geom"],
                "safety_score_pred": decoded_pred["safety_score_pred"],
                "structured_segment_score_geom": decoded_pred["structured_segment_score_geom"],
                "structured_segment_score_learned": decoded_pred["structured_segment_score_learned"],
                "structured_segment_score_final": decoded_pred["structured_segment_score_final"],
                "safety_subreason_gt": SAFETY_SUBREASON_ID_TO_LABEL.get(
                    int(numeric_batch["safety_subreason_label"][idx]), "unclear"
                ) if float(numeric_batch["safety_subreason_mask"][idx]) > 0 else None,
                "comfort_subreason_gt": COMFORT_SUBREASON_ID_TO_LABEL.get(
                    int(numeric_batch["comfort_subreason_label"][idx]), "unclear"
                ) if float(numeric_batch["comfort_subreason_mask"][idx]) > 0 else None,
                "safety_subreason_pred": decoded_pred["safety_subreason_pred"],
                "comfort_subreason_pred": decoded_pred["comfort_subreason_pred"],
                "comfort_zone_params": decoded_pred["comfort_zone_params"],
                "safety_zone_params": decoded_pred["safety_zone_params"],
                "comfort_adapter_summary": decoded_pred["comfort_adapter_summary"],
                "safety_adapter_summary": decoded_pred["safety_adapter_summary"],
                "shared_context_summary": decoded_pred["shared_context_summary"],
                "geometry_mode_summary": decoded_pred["geometry_mode_summary"],
            },
        )


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
    from JaxPref.data.structured_pref_dataset_v2_cs_rethinking import StructuredPrefDatasetV2CSRethinking
    from JaxPref.jax_utils import batch_to_jax, init_rng
    from flaxmodels.flaxmodels.gpt2.trajectory_gpt2 import TransRewardModel

    train_path = Path(args.data_dir) / args.train_file
    val_path = Path(args.data_dir) / args.val_file
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val file not found: {val_path}")

    train_dataset = StructuredPrefDatasetV2CSRethinking(str(train_path), max_seq_len=args.max_seq_len)
    val_dataset = StructuredPrefDatasetV2CSRethinking(str(val_path), max_seq_len=args.max_seq_len)

    print(f"Loaded train dataset: {len(train_dataset)} pairs")
    print(f"Loaded val dataset: {len(val_dataset)} pairs")
    print(f"Observation dim: {train_dataset.observation_dim}")
    print(f"Action dim: {train_dataset.action_dim}")
    print(f"Context dim: {train_dataset.context_dim}")
    print(f"Candidate aux dim: {train_dataset.candidate_aux_dim}")
    print(f"Geometry dim: {train_dataset.geometry_dim}")
    print(f"Geometry raw dim: {train_dataset.geometry_raw_dim}")
    print(f"Geometry mode: {args.geometry_mode}")
    print(f"Geometry parameterization mode: {args.geometry_parameterization_mode}")
    summarize_dataset(train_dataset, "train")
    summarize_dataset(val_dataset, "val")

    np.random.seed(args.seed)
    init_rng(args.seed)
    rng = np.random.default_rng(args.seed)

    config = ConfigDict(PrefMMT.get_default_config_v2())
    config.geometry_mode = args.geometry_mode
    config.geometry_parameterization_mode = args.geometry_parameterization_mode
    config.trans_lr = args.lr
    config.lambda_reason = args.lambda_reason
    config.lambda_reaction = args.lambda_reaction
    config.lambda_comfort_better = args.lambda_comfort_better
    config.lambda_safety_better = args.lambda_safety_better
    config.lambda_comfort_score = args.lambda_comfort_score
    config.lambda_safety_score = args.lambda_safety_score
    config.lambda_score_alignment = args.lambda_score_alignment
    config.lambda_geometry_prior = args.lambda_geometry_prior
    config.lambda_safety_subreason = args.lambda_safety_subreason
    config.lambda_comfort_subreason = args.lambda_comfort_subreason
    config.lambda_structured_preference_consistency = args.lambda_structured_preference_consistency
    config.lambda_learned_to_geom_consistency = args.lambda_learned_to_geom_consistency
    config.structured_tau_safe = args.structured_tau_safe
    config.structured_lambda_veto = args.structured_lambda_veto
    config.alpha_geom = args.alpha_geom
    config.alpha_learned = args.alpha_learned
    config.n_layer = args.n_layer
    config.embd_dim = args.embd_dim
    config.n_embd = args.embd_dim
    config.pref_attn_embd_dim = args.embd_dim
    config.n_head = args.n_head
    config.context_feature_dim = train_dataset.context_dim
    config.candidate_aux_dim = train_dataset.candidate_aux_dim
    config.geometry_feature_dim = train_dataset.geometry_dim
    config.geometry_raw_feature_dim = train_dataset.geometry_raw_dim
    config.warmup_steps = 1
    steps_per_epoch = max(1, int(np.ceil(len(train_dataset) / args.batch_size)))
    config.total_steps = max(1, args.epochs * steps_per_epoch)

    hf_config = transformers.GPT2Config(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_positions=config.n_positions,
        resid_pdrop=config.resid_pdrop,
        attn_pdrop=config.attn_pdrop,
    )
    for key in [
        "pref_attn_embd_dim",
        "use_weighted_sum",
        "num_reason_classes",
        "num_reaction_classes",
        "num_dominant_classes",
        "num_secondary_classes",
        "num_decomp_classes",
        "num_decomp_dimensions",
    ]:
        setattr(hf_config, key, getattr(config, key))

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
    best_pref_acc = (-np.inf, -np.inf)

    for epoch in range(1, args.epochs + 1):
        train_metrics = defaultdict(list)
        for batch_indices in iter_minibatches(train_dataset, args.batch_size, shuffle=True, rng=rng):
            numeric_batch, _ = build_batch(train_dataset, batch_indices)
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
            debug_predictions = tree_to_numpy(debug_predictions)
            print_debug_batch(debug_meta_batch, debug_numeric_batch, debug_predictions, limit=args.debug_batches)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(
            "train",
            {
                "trans_loss": round(train_summary.get("trans_loss", 0.0), 6),
                "preference_loss": round(train_summary.get("preference_loss", 0.0), 6),
                "structured_pref_consistency_overall": round(
                    train_summary.get("structured_preference_consistency_overall_loss", 0.0), 6
                ),
                "learned_to_geom_consistency_loss": round(
                    train_summary.get("learned_to_geom_consistency_loss", 0.0), 6
                ),
                "preference_accuracy": round(train_summary.get("preference_accuracy", 0.0), 6),
                "structured_preference_accuracy_geom": round(
                    train_summary.get("structured_preference_accuracy_geom", 0.0), 6
                ),
                "structured_preference_accuracy_learned": round(
                    train_summary.get("structured_preference_accuracy_learned", 0.0), 6
                ),
                "structured_preference_accuracy_final": round(
                    train_summary.get("structured_preference_accuracy_final", 0.0), 6
                ),
                "reason_accuracy": round(train_summary.get("reason_accuracy", 0.0), 6),
                "reaction_accuracy": round(train_summary.get("reaction_accuracy", 0.0), 6),
                "comfort_better_accuracy": round(
                    train_summary.get("comfort_better_accuracy", 0.0), 6
                ),
                "safety_better_accuracy": round(
                    train_summary.get("safety_better_accuracy", 0.0), 6
                ),
                "comfort_score_mae": round(train_summary.get("comfort_score_mae", 0.0), 6),
                "safety_score_mae": round(train_summary.get("safety_score_mae", 0.0), 6),
                "geometry_prior_loss": round(train_summary.get("geometry_prior_loss", 0.0), 6),
                "score_alignment_loss": round(train_summary.get("score_alignment_loss", 0.0), 6),
                "score_alignment_mae": round(train_summary.get("score_alignment_mae", 0.0), 6),
                "score_alignment_rmse": round(train_summary.get("score_alignment_rmse", 0.0), 6),
            },
        )
        print(
            "val",
            {
                "trans_loss": round(val_summary.get("eval_trans_loss", 0.0), 6),
                "preference_loss": round(val_summary.get("eval_preference_loss", 0.0), 6),
                "structured_pref_consistency_overall": round(
                    val_summary.get("eval_structured_preference_consistency_overall_loss", 0.0), 6
                ),
                "learned_to_geom_consistency_loss": round(
                    val_summary.get("eval_structured_preference_consistency_learned_to_geom_loss", 0.0), 6
                ),
                "preference_accuracy": round(val_summary.get("eval_preference_accuracy", 0.0), 6),
                "structured_preference_accuracy_geom": round(
                    val_summary.get("eval_structured_preference_accuracy_geom", 0.0), 6
                ),
                "structured_preference_accuracy_learned": round(
                    val_summary.get("eval_structured_preference_accuracy_learned", 0.0), 6
                ),
                "structured_preference_accuracy_final": round(
                    val_summary.get("eval_structured_preference_accuracy_final", 0.0), 6
                ),
                "reason_accuracy": round(val_summary.get("eval_reason_accuracy", 0.0), 6),
                "reaction_accuracy": round(val_summary.get("eval_reaction_accuracy", 0.0), 6),
                "comfort_better_accuracy": round(
                    val_summary.get("eval_comfort_better_accuracy", 0.0), 6
                ),
                "safety_better_accuracy": round(
                    val_summary.get("eval_safety_better_accuracy", 0.0), 6
                ),
                "comfort_score_mae": round(val_summary.get("eval_comfort_score_mae", 0.0), 6),
                "safety_score_mae": round(val_summary.get("eval_safety_score_mae", 0.0), 6),
                "geometry_prior_loss": round(val_summary.get("eval_geometry_prior_loss", 0.0), 6),
                "score_alignment_loss": round(val_summary.get("eval_score_alignment_loss", 0.0), 6),
                "score_alignment_mae": round(val_summary.get("eval_score_alignment_mae", 0.0), 6),
                "score_alignment_rmse": round(val_summary.get("eval_score_alignment_rmse", 0.0), 6),
            },
        )

        save_checkpoint(latest_ckpt, model, epoch, {"train": train_summary, "val": val_summary}, config)
        val_pref_acc = (
            val_summary.get("eval_preference_accuracy", 0.0),
            val_summary.get("eval_structured_preference_accuracy_final", 0.0),
        )
        if val_pref_acc >= best_pref_acc:
            best_pref_acc = val_pref_acc
            save_checkpoint(best_ckpt, model, epoch, {"train": train_summary, "val": val_summary}, config)

    print(f"\nTraining finished. Latest checkpoint: {latest_ckpt}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
