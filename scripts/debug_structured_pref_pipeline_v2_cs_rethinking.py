import argparse
import importlib
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "PrefMMT") not in sys.path:
    sys.path.insert(0, str(ROOT / "PrefMMT"))


BETTER_INDEX_TO_LABEL = {0: "A", 1: "B", 2: "tie", 3: "unclear"}
PREFERENCE_INDEX_TO_LABEL = BETTER_INDEX_TO_LABEL
REASON_INDEX_TO_LABEL = {0: "safety", 1: "comfort", 2: "mixed", 3: "unclear"}
SAFETY_SUBREASON_INDEX_TO_LABEL = {
    0: "distance_intrusion",
    1: "speed_risk",
    2: "timing_risk",
    3: "mixed",
    4: "unclear",
}
COMFORT_SUBREASON_INDEX_TO_LABEL = {
    0: "reachability",
    1: "height_alignment",
    2: "front_sector_alignment",
    3: "posture_cost",
    4: "mixed",
    5: "unclear",
}
GEOMETRY_MODE_ID_TO_LABEL = {0: "fixed", 1: "trainable"}
GEOMETRY_PARAM_MODE_ID_TO_LABEL = {0: "fixed", 1: "global", 2: "contextual"}


def structured_segment_score(comfort_score, safety_score, tau_safe=0.5, lambda_veto=4.0):
    penalty = max(tau_safe - safety_score, 0.0)
    return comfort_score - lambda_veto * penalty * penalty


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug the V2 comfort/safety structured preference pipeline."
    )
    parser.add_argument("--data_dir", default="data/synthetic_v2_cs_rethinking")
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--val_file", default="val.jsonl")
    parser.add_argument("--max_seq_len", type=int, default=8)
    parser.add_argument("--preview_count", type=int, default=3)
    parser.add_argument("--checkpoint", default="outputs/structured_pref_v2_cs_rethinking/latest.pkl")
    return parser.parse_args()


def summarize_sources(records):
    counter = Counter()
    for item in records:
        counter[item["metadata"]["candidate_source_a"]] += 1
        counter[item["metadata"]["candidate_source_b"]] += 1
    return counter


def load_context_records(path):
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check_modules(module_names):
    status = {}
    for name in module_names:
        try:
            importlib.import_module(name)
            status[name] = "ok"
        except Exception as exc:
            status[name] = f"missing: {type(exc).__name__}: {exc}"
    return status


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


def split_numeric_and_meta(batch):
    numeric_batch = {}
    meta_batch = {}
    for key, value in batch.items():
        if key in {"sample_id", "metadata", "context_id", "pair_id"}:
            meta_batch[key] = value
        else:
            numeric_batch[key] = value
    return numeric_batch, meta_batch


def print_model_preview(meta_batch, numeric_batch, predictions, limit):
    count = min(limit, len(meta_batch["sample_id"]))
    for idx in range(count):
        metadata = meta_batch["metadata"][idx]
        pref_label = int(numeric_batch["preference_label"][idx])
        loser_gt = "A" if pref_label == 1 else "B"
        subreason_active_pred = bool(float(predictions["predicted_subreason_active"][idx]) > 0.5)
        loser_indicator = int(np.asarray(predictions["loser_indicator"][idx]))
        geometry_mode_summary = predictions.get("geometry_mode_summary", {})
        geometry_mode = GEOMETRY_MODE_ID_TO_LABEL.get(
            int(np.asarray(geometry_mode_summary.get("geometry_mode_id", 0))),
            "unknown",
        )
        geometry_param_mode = GEOMETRY_PARAM_MODE_ID_TO_LABEL.get(
            int(np.asarray(geometry_mode_summary.get("geometry_parameterization_mode_id", 0))),
            "unknown",
        )
        print(
            "  model_pair",
            {
                "sample_id": meta_batch["sample_id"][idx],
                "scene_type": metadata["scene_type"],
                "comfort_score_geom_a": round(float(predictions["comfort_score_geom"][idx][0]), 3),
                "safety_score_geom_a": round(float(predictions["safety_score_geom"][idx][0]), 3),
                "comfort_score_pred_a": round(float(predictions["comfort_score_preds"][idx][0]), 3),
                "safety_score_pred_a": round(float(predictions["safety_score_preds"][idx][0]), 3),
                "comfort_score_geom_b": round(float(predictions["comfort_score_geom"][idx][1]), 3),
                "safety_score_geom_b": round(float(predictions["safety_score_geom"][idx][1]), 3),
                "comfort_score_pred_b": round(float(predictions["comfort_score_preds"][idx][1]), 3),
                "safety_score_pred_b": round(float(predictions["safety_score_preds"][idx][1]), 3),
                "segment_score_structured_geom_a": round(float(predictions["segment_score_structured_geom"][idx][0]), 3),
                "segment_score_structured_geom_b": round(float(predictions["segment_score_structured_geom"][idx][1]), 3),
                "segment_score_structured_learned_a": round(float(predictions["segment_score_structured_learned"][idx][0]), 3),
                "segment_score_structured_learned_b": round(float(predictions["segment_score_structured_learned"][idx][1]), 3),
                "segment_score_structured_final_a": round(float(predictions["segment_score_structured_final"][idx][0]), 3),
                "segment_score_structured_final_b": round(float(predictions["segment_score_structured_final"][idx][1]), 3),
                "overall_preference_gt": PREFERENCE_INDEX_TO_LABEL[pref_label],
                "preference_pred": PREFERENCE_INDEX_TO_LABEL[int(np.argmax(predictions["overall_preference_logits"][idx]))],
                "structured_preference_pred_geom": PREFERENCE_INDEX_TO_LABEL[int(np.argmax(predictions["structured_preference_logits_geom"][idx]))],
                "structured_preference_pred_learned": PREFERENCE_INDEX_TO_LABEL[int(np.argmax(predictions["structured_preference_logits_learned"][idx]))],
                "structured_preference_pred_final": PREFERENCE_INDEX_TO_LABEL[int(np.argmax(predictions["structured_preference_logits_final"][idx]))],
                "loser_id_gt": loser_gt,
                "loser_id_pred": ("A" if loser_indicator == 0 else "B") if subreason_active_pred else None,
                "subreason_active_pred": subreason_active_pred,
                "reason_gt": REASON_INDEX_TO_LABEL[int(numeric_batch["reason_label"][idx])],
                "safety_subreason_gt": (
                    SAFETY_SUBREASON_INDEX_TO_LABEL[int(numeric_batch["safety_subreason_label"][idx])]
                    if float(numeric_batch["safety_subreason_mask"][idx]) > 0
                    else None
                ),
                "comfort_subreason_gt": (
                    COMFORT_SUBREASON_INDEX_TO_LABEL[int(numeric_batch["comfort_subreason_label"][idx])]
                    if float(numeric_batch["comfort_subreason_mask"][idx]) > 0
                    else None
                ),
                "safety_subreason_pred": (
                    SAFETY_SUBREASON_INDEX_TO_LABEL[int(np.argmax(predictions["safety_subreason_logits"][idx]))]
                    if subreason_active_pred
                    else None
                ),
                "comfort_subreason_pred": (
                    COMFORT_SUBREASON_INDEX_TO_LABEL[int(np.argmax(predictions["comfort_subreason_logits"][idx]))]
                    if subreason_active_pred
                    else None
                ),
                "comfort_zone_params": {
                    key: _param_value_to_python(value[idx] if np.asarray(value).ndim > 0 else value)
                    for key, value in predictions["comfort_zone_params"].items()
                },
                "safety_zone_params": {
                    key: _param_value_to_python(value[idx] if np.asarray(value).ndim > 0 else value)
                    for key, value in predictions["safety_zone_params"].items()
                },
                "comfort_adapter_summary": {
                    key: _param_value_to_python(value[idx] if np.asarray(value).ndim > 0 else value)
                    for key, value in predictions["comfort_adapter_summary"].items()
                },
                "safety_adapter_summary": {
                    key: _param_value_to_python(value[idx] if np.asarray(value).ndim > 0 else value)
                    for key, value in predictions["safety_adapter_summary"].items()
                },
                "shared_context_summary": {
                    key: _param_value_to_python(value[idx] if np.asarray(value).ndim > 0 else value)
                    for key, value in predictions["shared_context_summary"].items()
                },
                "geometry_mode_summary": {
                    "geometry_mode": geometry_mode,
                    "geometry_parameterization_mode": geometry_param_mode,
                },
            },
        )


def main():
    args = parse_args()

    print("\nRuntime dependency check")
    for name, status in check_modules(
        ["jax", "flax", "optax", "transformers", "ml_collections"]
    ).items():
        print(f"  {name}: {status}")

    from JaxPref.data.structured_pref_dataset_v2_cs_rethinking import (
        StructuredPrefDatasetV2CSRethinking,
        load_structured_pref_batch_v2_cs_rethinking,
    )

    train_path = Path(args.data_dir) / args.train_file
    val_path = Path(args.data_dir) / args.val_file
    print(f"train path: {train_path}")
    print(f"val path: {val_path}")
    print(f"train exists: {train_path.exists()}")
    print(f"val exists: {val_path.exists()}")

    train_dataset = StructuredPrefDatasetV2CSRethinking(str(train_path), max_seq_len=args.max_seq_len)
    val_dataset = StructuredPrefDatasetV2CSRethinking(str(val_path), max_seq_len=args.max_seq_len)

    print("\nLoader check")
    print(f"  train pairs: {len(train_dataset)}")
    print(f"  val pairs: {len(val_dataset)}")
    print(f"  observation_dim: {train_dataset.observation_dim}")
    print(f"  action_dim: {train_dataset.action_dim}")
    print(f"  context_dim: {train_dataset.context_dim}")
    print(f"  candidate_aux_dim: {train_dataset.candidate_aux_dim}")
    print(f"  geometry_dim: {train_dataset.geometry_dim}")
    print(f"  geometry_raw_dim: {train_dataset.geometry_raw_dim}")

    preview = train_dataset.get_batch(range(min(args.preview_count, len(train_dataset))))
    print("\nBatch preview")
    print("  observations:", preview["observations"].shape)
    print("  actions:", preview["actions"].shape)
    print("  observations_2:", preview["observations_2"].shape)
    print("  labels:", preview["labels"].shape)
    print("  context_features:", preview["context_features"].shape)
    print("  candidate_a_aux_features:", preview["candidate_a_aux_features"].shape)
    print("  candidate_a_geometry_features:", preview["candidate_a_geometry_features"].shape)
    print("  candidate_a_geometry_raw_features:", preview["candidate_a_geometry_raw_features"].shape)
    print("  preference_label:", preview["preference_label"].tolist())
    print("  reason_label:", preview["reason_label"].tolist())
    print("  reaction_label:", preview["reaction_label"].tolist())
    print("  comfort_better_label:", preview["comfort_better_label"].tolist())
    print("  safety_better_label:", preview["safety_better_label"].tolist())
    print("  comfort_score_masks:", preview["comfort_score_masks"].tolist())
    print("  safety_score_masks:", preview["safety_score_masks"].tolist())
    print("  safety_subreason_mask:", preview["safety_subreason_mask"].tolist())
    print("  comfort_subreason_mask:", preview["comfort_subreason_mask"].tolist())
    print("  sample_ids:", preview["sample_id"].tolist())

    print("\nCandidate source distribution")
    for label, count in sorted(summarize_sources(train_dataset.samples).items()):
        print(f"  {label}: {count}")

    prefmmt_batch = load_structured_pref_batch_v2_cs_rethinking(
        str(val_path),
        max_seq_len=args.max_seq_len,
    )
    print("\nPrefMMT-compatible V2 batch check")
    print("  observations:", prefmmt_batch["observations"].shape)
    print("  actions:", prefmmt_batch["actions"].shape)
    print("  observations_2:", prefmmt_batch["observations_2"].shape)
    print("  actions_2:", prefmmt_batch["actions_2"].shape)
    print("  context_features:", prefmmt_batch["context_features"].shape)
    print("  candidate_b_aux_features:", prefmmt_batch["candidate_b_aux_features"].shape)
    print("  candidate_b_geometry_features:", prefmmt_batch["candidate_b_geometry_features"].shape)
    print("  comfort_score_targets:", prefmmt_batch["comfort_score_targets"].shape)
    print("  safety_score_targets:", prefmmt_batch["safety_score_targets"].shape)

    print("\nCandidate set summary")
    raw_records = load_context_records(train_path)
    for record in raw_records[: args.preview_count]:
        print(
            f"  {record['context_id']}: scene={record['context']['scene_type']} "
            f"candidates={len(record['candidate_set'])} pairs={len(record['training_pairs'])}"
        )
        for candidate in record["candidate_set"]:
            item = next(
                sample
                for sample in train_dataset.samples
                if sample["metadata"]["context_id"] == record["context_id"]
                and (
                    sample["metadata"]["candidate_a_id"] == candidate["candidate_id"]
                    or sample["metadata"]["candidate_b_id"] == candidate["candidate_id"]
                )
            )
            geometry = (
                item["metadata"]["candidate_a_geometry"]
                if item["metadata"]["candidate_a_id"] == candidate["candidate_id"]
                else item["metadata"]["candidate_b_geometry"]
            )
            comfort = geometry["comfort"]
            safety = geometry["safety"]
            print(
                f"    {candidate['candidate_id']} source={candidate['candidate_source']} "
                f"comfort_score={comfort['comfort_score']:.3f} safety_score={safety['safety_score']:.3f}"
            )
            print(
                f"      comfort_params={comfort['comfort_zone_params']} "
                f"safety_params={safety['safety_zone_params']}"
            )
            print(
                f"      comfort_intermediates={{'distance': {comfort['comfort_intermediates']['relative_distance']:.3f}, "
                f"'h_angle_deg': {comfort['comfort_intermediates']['horizontal_angle_deg']:.2f}, "
                f"'v_angle_deg': {comfort['comfort_intermediates']['vertical_angle_deg']:.2f}}}"
            )
            print(
                f"      safety_intermediates={{'ee_radius': {safety['safety_intermediates']['ee_ellipsoid_radius']:.3f}, "
                f"'base_radius': {safety['safety_intermediates']['base_ellipsoid_radius']:.3f}, "
                f"'max_speed': {safety['safety_intermediates']['max_approach_speed']:.3f}, "
                f"'timing_speed': {safety['safety_intermediates']['timing_window_speed']:.3f}}}"
            )

    print("\nPair preview")
    for item in train_dataset.samples[: args.preview_count]:
        metadata = item["metadata"]
        comfort_a = metadata["candidate_a_geometry"]["comfort"]["comfort_score"]
        comfort_b = metadata["candidate_b_geometry"]["comfort"]["comfort_score"]
        safety_a = metadata["candidate_a_geometry"]["safety"]["safety_score"]
        safety_b = metadata["candidate_b_geometry"]["safety"]["safety_score"]
        structured_a = structured_segment_score(comfort_a, safety_a)
        structured_b = structured_segment_score(comfort_b, safety_b)
        target_structured_a = None
        target_structured_b = None
        if max(item["comfort_score_masks"]) > 0 and max(item["safety_score_masks"]) > 0:
            target_structured_a = float(item["comfort_score_targets"][0]) - 4.0 * max(0.5 - float(item["safety_score_targets"][0]), 0.0) ** 2
            target_structured_b = float(item["comfort_score_targets"][1]) - 4.0 * max(0.5 - float(item["safety_score_targets"][1]), 0.0) ** 2
        print(
            f"  {metadata['pair_id']}: {metadata['candidate_source_a']} vs {metadata['candidate_source_b']} "
            f"scene={metadata['scene_type']}"
        )
        print(
            f"    comfort_gt={BETTER_INDEX_TO_LABEL[int(item['comfort_better_label'])]} "
            f"safety_gt={BETTER_INDEX_TO_LABEL[int(item['safety_better_label'])]} "
            f"comfort_geom=({comfort_a:.3f}, {comfort_b:.3f}) safety_geom=({safety_a:.3f}, {safety_b:.3f})"
        )
        print(
            f"    structured_score_geom=({structured_a:.3f}, {structured_b:.3f}) "
            f"structured_score_target=({target_structured_a}, {target_structured_b}) "
            f"overall_gt={BETTER_INDEX_TO_LABEL[int(item['preference_label'])]}"
        )
        print(
            f"    comfort_mask={item['comfort_score_masks'].tolist()} "
            f"safety_mask={item['safety_score_masks'].tolist()}"
        )

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists() and all(
        status == "ok" for status in check_modules(["jax", "flax", "optax", "transformers", "ml_collections"]).values()
    ):
        print("\nModel branch preview")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        import transformers
        from ml_collections import ConfigDict

        from JaxPref.PrefMMT import PrefMMT
        from JaxPref.jax_utils import batch_to_jax, init_rng
        from flaxmodels.flaxmodels.gpt2.trajectory_gpt2 import TransRewardModel

        with checkpoint_path.open("rb") as fh:
            payload = pickle.load(fh)
        config = ConfigDict(payload["config"])
        config.context_feature_dim = train_dataset.context_dim
        config.candidate_aux_dim = train_dataset.candidate_aux_dim
        config.geometry_feature_dim = train_dataset.geometry_dim
        config.geometry_raw_feature_dim = train_dataset.geometry_raw_dim
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
        init_rng(0)
        model = PrefMMT(config, trans)
        for key, params in payload["train_params"].items():
            model._train_states[key] = model._train_states[key].replace(params=params)
        preview_batch = train_dataset.get_batch(range(min(args.preview_count, len(train_dataset))))
        numeric_batch, meta_batch = split_numeric_and_meta(preview_batch)
        predictions = model.predict(batch_to_jax(numeric_batch))
        predictions = tree_to_numpy(predictions)
        print_model_preview(meta_batch, numeric_batch, predictions, args.preview_count)
    else:
        print("\nModel branch preview skipped")
        print(f"  checkpoint exists: {checkpoint_path.exists()}")
        print("  reason: missing checkpoint or runtime dependencies")

    print("\nV2 structured preference pipeline debug finished.")


if __name__ == "__main__":
    main()
