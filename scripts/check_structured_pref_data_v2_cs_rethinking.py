import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_TOP_LEVEL = [
    "context_id",
    "context",
    "candidate_set",
    "training_pairs",
    "annotator_id",
    "notes",
]
REQUIRED_CONTEXT = [
    "scene_type",
    "human_position",
    "human_orientation",
    "human_hand_position",
    "human_posture_features",
    "robot_initial_pose",
    "robot_initial_velocity",
    "environment_map_id",
    "local_costmap_embedding",
    "obstacle_proximity_features",
    "reachable_region_features",
    "task_goal",
]
REQUIRED_CANDIDATE = [
    "candidate_id",
    "candidate_source",
    "candidate_generator_detail",
    "planner_family",
    "hybrid_role",
    "source_description",
    "trajectory",
    "handover_pose",
    "handover_timing",
    "handover_time_context",
    "feasibility_flags",
]
REQUIRED_TRAJECTORY = [
    "time_seq",
    "robot_state_seq",
    "robot_action_seq",
    "robot_base_pose_seq",
    "robot_ee_pose_seq",
    "robot_velocity_seq",
    "gripper_state_seq",
    "object_pose_seq",
    "human_pose_seq",
    "human_hand_pose_seq",
    "relative_position_seq",
    "relative_distance_seq",
    "relative_orientation_seq",
]
REQUIRED_PAIR = [
    "pair_id",
    "candidate_a_id",
    "candidate_b_id",
    "pair_metadata",
    "labels",
]
REQUIRED_LABELS = [
    "overall_preference",
    "reason_label",
    "reaction_label",
    "comfort_better_label",
    "safety_better_label",
]

CANDIDATE_SOURCE_ENUM = {"classical", "learned", "hybrid", "safe_fallback"}
PLANNER_FAMILY_ENUM = {"geometric", "learning_based", "hybrid"}
HYBRID_ROLE_ENUM = {"none", "geometric_backbone_learning_refinement", "learning_backbone_geometric_guard", "switching_controller"}
PREFERENCE_ENUM = {"A", "B", "tie", "unclear"}
BETTER_ENUM = {"A", "B", "tie", "unclear"}
REASON_ENUM = {"safety", "comfort", "mixed", "unclear"}
REACTION_ENUM = {
    "none",
    "hesitation",
    "avoidance",
    "interruption",
    "overreach_or_reposition",
    "unnatural_posture",
    "unclear",
}
SAFETY_SUBREASON_ENUM = {"distance_intrusion", "speed_risk", "timing_risk", "mixed", "unclear"}
COMFORT_SUBREASON_ENUM = {"reachability", "height_alignment", "front_sector_alignment", "posture_cost", "mixed", "unclear"}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate V2 comfort/safety structured preference data.")
    parser.add_argument("--data", required=True, help="JSONL file or directory with train.jsonl/val.jsonl")
    parser.add_argument("--preview-count", type=int, default=3)
    return parser.parse_args()


def load_records(path):
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def ensure_fields(obj, fields, prefix, errors):
    for field in fields:
        if field not in obj:
            errors.append(f"{prefix}: missing field `{field}`")


def validate_enum(value, allowed, prefix, field, errors):
    if value not in allowed:
        errors.append(f"{prefix}: invalid `{field}`={value!r}")


def validate_trajectory(prefix, trajectory, errors):
    ensure_fields(trajectory, REQUIRED_TRAJECTORY, prefix, errors)
    time_seq = trajectory.get("time_seq", [])
    if not isinstance(time_seq, list) or not time_seq:
        errors.append(f"{prefix}.time_seq must be a non-empty list")
        return
    target_len = len(time_seq)
    for key in REQUIRED_TRAJECTORY[1:]:
        seq = trajectory.get(key)
        if not isinstance(seq, list):
            errors.append(f"{prefix}.{key} must be a list")
            continue
        if len(seq) != target_len:
            errors.append(f"{prefix}.{key} length {len(seq)} != time_seq length {target_len}")


def validate_score_target(prefix, field, value, errors):
    if value is None:
        return
    if not isinstance(value, dict):
        errors.append(f"{prefix}.{field} must be null or a dict with A/B keys")
        return
    if set(value.keys()) != {"A", "B"}:
        errors.append(f"{prefix}.{field} must contain exactly A/B keys")
        return
    for side, number in value.items():
        if not isinstance(number, (int, float)):
            errors.append(f"{prefix}.{field}.{side} must be numeric")


def validate_sample(sample, errors):
    context_id = sample.get("context_id", "<missing_context_id>")
    ensure_fields(sample, REQUIRED_TOP_LEVEL, context_id, errors)
    context = sample.get("context", {})
    ensure_fields(context, REQUIRED_CONTEXT, f"{context_id}.context", errors)

    candidate_set = sample.get("candidate_set", [])
    if not isinstance(candidate_set, list) or not candidate_set:
        errors.append(f"{context_id}.candidate_set must be a non-empty list")
        candidate_lookup = {}
    else:
        candidate_lookup = {}
        for candidate in candidate_set:
            candidate_id = candidate.get("candidate_id", "<missing_candidate_id>")
            ensure_fields(candidate, REQUIRED_CANDIDATE, f"{context_id}.candidate_set[{candidate_id}]", errors)
            if candidate_id in candidate_lookup:
                errors.append(f"{context_id}.candidate_set duplicate candidate_id `{candidate_id}`")
            candidate_lookup[candidate_id] = candidate
            validate_enum(candidate.get("candidate_source"), CANDIDATE_SOURCE_ENUM, f"{context_id}.candidate_set[{candidate_id}]", "candidate_source", errors)
            validate_enum(candidate.get("planner_family"), PLANNER_FAMILY_ENUM, f"{context_id}.candidate_set[{candidate_id}]", "planner_family", errors)
            validate_enum(candidate.get("hybrid_role"), HYBRID_ROLE_ENUM, f"{context_id}.candidate_set[{candidate_id}]", "hybrid_role", errors)
            validate_trajectory(f"{context_id}.candidate_set[{candidate_id}].trajectory", candidate.get("trajectory", {}), errors)

    training_pairs = sample.get("training_pairs", [])
    if not isinstance(training_pairs, list) or not training_pairs:
        errors.append(f"{context_id}.training_pairs must be a non-empty list")
        return

    for pair in training_pairs:
        pair_id = pair.get("pair_id", "<missing_pair_id>")
        ensure_fields(pair, REQUIRED_PAIR, f"{context_id}.training_pairs[{pair_id}]", errors)
        a_id = pair.get("candidate_a_id")
        b_id = pair.get("candidate_b_id")
        if a_id not in candidate_lookup:
            errors.append(f"{context_id}.training_pairs[{pair_id}]: unknown candidate_a_id `{a_id}`")
        if b_id not in candidate_lookup:
            errors.append(f"{context_id}.training_pairs[{pair_id}]: unknown candidate_b_id `{b_id}`")
        labels = pair.get("labels", {})
        ensure_fields(labels, REQUIRED_LABELS, f"{context_id}.training_pairs[{pair_id}].labels", errors)
        validate_enum(labels.get("overall_preference"), PREFERENCE_ENUM, f"{context_id}.training_pairs[{pair_id}].labels", "overall_preference", errors)
        validate_enum(labels.get("reason_label"), REASON_ENUM, f"{context_id}.training_pairs[{pair_id}].labels", "reason_label", errors)
        validate_enum(labels.get("reaction_label"), REACTION_ENUM, f"{context_id}.training_pairs[{pair_id}].labels", "reaction_label", errors)
        validate_enum(labels.get("comfort_better_label"), BETTER_ENUM, f"{context_id}.training_pairs[{pair_id}].labels", "comfort_better_label", errors)
        validate_enum(labels.get("safety_better_label"), BETTER_ENUM, f"{context_id}.training_pairs[{pair_id}].labels", "safety_better_label", errors)
        validate_score_target(f"{context_id}.training_pairs[{pair_id}].labels", "comfort_score_target", labels.get("comfort_score_target"), errors)
        validate_score_target(f"{context_id}.training_pairs[{pair_id}].labels", "safety_score_target", labels.get("safety_score_target"), errors)

        safety_subreason = labels.get("safety_subreason_label")
        comfort_subreason = labels.get("comfort_subreason_label")
        if safety_subreason is not None:
            validate_enum(safety_subreason, SAFETY_SUBREASON_ENUM, f"{context_id}.training_pairs[{pair_id}].labels", "safety_subreason_label", errors)
        if comfort_subreason is not None:
            validate_enum(comfort_subreason, COMFORT_SUBREASON_ENUM, f"{context_id}.training_pairs[{pair_id}].labels", "comfort_subreason_label", errors)


def summarize_records(records):
    stats = {
        "scene_type": Counter(),
        "candidate_source": Counter(),
        "overall_preference": Counter(),
        "reason_label": Counter(),
        "reaction_label": Counter(),
        "comfort_better_label": Counter(),
        "safety_better_label": Counter(),
    }
    for sample in records:
        stats["scene_type"][sample["context"]["scene_type"]] += 1
        for candidate in sample["candidate_set"]:
            stats["candidate_source"][candidate["candidate_source"]] += 1
        for pair in sample["training_pairs"]:
            labels = pair["labels"]
            stats["overall_preference"][labels["overall_preference"]] += 1
            stats["reason_label"][labels["reason_label"]] += 1
            stats["reaction_label"][labels["reaction_label"]] += 1
            stats["comfort_better_label"][labels["comfort_better_label"]] += 1
            stats["safety_better_label"][labels["safety_better_label"]] += 1
    return stats


def print_stats(title, stats):
    print(f"\n{title}")
    for key, counter in stats.items():
        print(f"  {key}:")
        for label, count in sorted(counter.items()):
            print(f"    {label}: {count}")


def print_preview(records, limit):
    print(f"\nPreviewing {min(limit, len(records))} context records")
    for record in records[:limit]:
        print(f"- {record['context_id']}")
        print(f"  scene={record['context']['scene_type']} candidates={len(record['candidate_set'])} pairs={len(record['training_pairs'])}")
        print(f"  sources={[candidate['candidate_source'] for candidate in record['candidate_set']]}")
        first_pair = record["training_pairs"][0]
        labels = first_pair["labels"]
        print(
            f"  first_pair={first_pair['pair_id']} pref={labels['overall_preference']} "
            f"reason={labels['reason_label']} comfort_better={labels['comfort_better_label']} "
            f"safety_better={labels['safety_better_label']}"
        )


def expand_input(path):
    if path.is_dir():
        return [item for item in [path / "train.jsonl", path / "val.jsonl"] if item.exists()]
    return [path]


def main():
    args = parse_args()
    target = Path(args.data)
    files = expand_input(target)
    if not files:
        raise FileNotFoundError(f"No data files found under {target}")

    total_records = []
    all_errors = []
    for file_path in files:
        records = load_records(file_path)
        print(f"Loaded {len(records)} context records from {file_path}")
        for sample in records:
            validate_sample(sample, all_errors)
        print_stats(f"Summary for {file_path.name}", summarize_records(records))
        print_preview(records, args.preview_count)
        total_records.extend(records)

    if all_errors:
        print("\nValidation errors:")
        for error in all_errors:
            print(f"  - {error}")
        raise SystemExit(1)

    total_pairs = sum(len(record["training_pairs"]) for record in total_records)
    print(f"\nValidation passed for {len(total_records)} contexts and {total_pairs} training pairs.")


if __name__ == "__main__":
    main()
