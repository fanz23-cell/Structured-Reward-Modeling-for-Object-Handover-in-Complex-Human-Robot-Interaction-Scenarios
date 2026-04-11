import argparse
import json
import math
import random
from pathlib import Path

from generate_structured_pref_synth_v2_cs_rethinking import build_candidate


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "synthetic_v2_cs_rethinking_plus"

SCENE_LIBRARY = {
    "open_space": {
        "scene_type": "open_space",
        "human_position": [0.0, 0.0, 0.0],
        "human_orientation": -0.08,
        "human_hand_position": [0.26, 0.01, 1.01],
        "human_posture_features": {"arm_extension_fraction": 0.53, "torso_lean": 0.04, "shoulder_height": 1.35, "stance_width": 0.33},
        "robot_initial_pose": [-1.28, -0.18, 0.08],
        "robot_initial_velocity": [0.0, 0.0, 0.0],
        "environment_map_id": "plus_open_space",
        "local_costmap_embedding": [0.08, 0.15, 0.10, 0.09],
        "obstacle_proximity_features": {"min_distance_to_obstacle": 0.72, "left_clearance": 0.80, "right_clearance": 0.77},
        "reachable_region_features": {"forward_reach_radius": 0.76, "vertical_band_low": 0.90, "vertical_band_high": 1.14},
        "task_goal": {"goal_type": "approach_and_handover", "handover_location_target": [0.67, -0.03, 1.02], "handover_location_relative_to_human": "front_mid", "handover_timing_constraint": "nominal"},
        "corridor_width": 1.8,
        "doorway_width": 1.8,
    },
    "doorway": {
        "scene_type": "doorway",
        "human_position": [0.0, 0.0, 0.0],
        "human_orientation": 0.0,
        "human_hand_position": [0.24, 0.03, 1.03],
        "human_posture_features": {"arm_extension_fraction": 0.56, "torso_lean": 0.08, "shoulder_height": 1.34, "stance_width": 0.29},
        "robot_initial_pose": [-1.10, -0.10, 0.05],
        "robot_initial_velocity": [0.0, 0.0, 0.0],
        "environment_map_id": "plus_doorway",
        "local_costmap_embedding": [0.15, 0.33, 0.24, 0.18],
        "obstacle_proximity_features": {"min_distance_to_obstacle": 0.42, "left_clearance": 0.36, "right_clearance": 0.28},
        "reachable_region_features": {"forward_reach_radius": 0.72, "vertical_band_low": 0.90, "vertical_band_high": 1.12},
        "task_goal": {"goal_type": "approach_and_handover", "handover_location_target": [0.66, 0.12, 1.02], "handover_location_relative_to_human": "front_mid", "handover_timing_constraint": "nominal"},
        "corridor_width": 1.0,
        "doorway_width": 0.88,
    },
    "narrow_corridor": {
        "scene_type": "narrow_corridor",
        "human_position": [0.0, 0.0, 0.0],
        "human_orientation": 0.12,
        "human_hand_position": [0.23, -0.01, 1.00],
        "human_posture_features": {"arm_extension_fraction": 0.60, "torso_lean": 0.06, "shoulder_height": 1.31, "stance_width": 0.25},
        "robot_initial_pose": [-1.18, 0.18, -0.04],
        "robot_initial_velocity": [0.0, 0.0, 0.0],
        "environment_map_id": "plus_narrow_corridor",
        "local_costmap_embedding": [0.26, 0.38, 0.20, 0.32],
        "obstacle_proximity_features": {"min_distance_to_obstacle": 0.24, "left_clearance": 0.18, "right_clearance": 0.22},
        "reachable_region_features": {"forward_reach_radius": 0.67, "vertical_band_low": 0.88, "vertical_band_high": 1.08},
        "task_goal": {"goal_type": "approach_and_handover", "handover_location_target": [0.60, 0.02, 0.99], "handover_location_relative_to_human": "front_mid", "handover_timing_constraint": "narrow_space"},
        "corridor_width": 0.76,
        "doorway_width": 0.76,
    },
    "table_separated": {
        "scene_type": "table_separated",
        "human_position": [0.0, 0.0, 0.0],
        "human_orientation": -0.04,
        "human_hand_position": [0.28, 0.05, 1.02],
        "human_posture_features": {"arm_extension_fraction": 0.50, "torso_lean": 0.05, "shoulder_height": 1.33, "stance_width": 0.31},
        "robot_initial_pose": [-1.25, -0.30, 0.08],
        "robot_initial_velocity": [0.0, 0.0, 0.0],
        "environment_map_id": "plus_table_separated",
        "local_costmap_embedding": [0.31, 0.27, 0.18, 0.23],
        "obstacle_proximity_features": {"min_distance_to_obstacle": 0.34, "left_clearance": 0.40, "right_clearance": 0.30},
        "reachable_region_features": {"forward_reach_radius": 0.70, "vertical_band_low": 0.89, "vertical_band_high": 1.13},
        "task_goal": {"goal_type": "approach_and_handover", "handover_location_target": [0.73, 0.02, 1.01], "handover_location_relative_to_human": "front_mid", "handover_timing_constraint": "table_barrier"},
        "corridor_width": 1.3,
        "doorway_width": 1.3,
    },
    "corner": {
        "scene_type": "corner",
        "human_position": [0.0, 0.0, 0.0],
        "human_orientation": 0.22,
        "human_hand_position": [0.24, 0.02, 1.00],
        "human_posture_features": {"arm_extension_fraction": 0.58, "torso_lean": 0.07, "shoulder_height": 1.32, "stance_width": 0.28},
        "robot_initial_pose": [-1.08, 0.28, -0.02],
        "robot_initial_velocity": [0.0, 0.0, 0.0],
        "environment_map_id": "plus_corner",
        "local_costmap_embedding": [0.19, 0.36, 0.26, 0.21],
        "obstacle_proximity_features": {"min_distance_to_obstacle": 0.30, "left_clearance": 0.25, "right_clearance": 0.43},
        "reachable_region_features": {"forward_reach_radius": 0.69, "vertical_band_low": 0.88, "vertical_band_high": 1.09},
        "task_goal": {"goal_type": "approach_and_handover", "handover_location_target": [0.59, 0.11, 1.00], "handover_location_relative_to_human": "front_mid", "handover_timing_constraint": "corner_turn"},
        "corridor_width": 0.92,
        "doorway_width": 0.92,
    },
}

COMFORT_FAILURES = ["reachability", "height_alignment", "front_sector_alignment", "posture_cost"]
SAFETY_FAILURES = ["distance_intrusion", "speed_risk", "timing_risk"]
PAIR_BLUEPRINTS = [
    ("classical_vs_learned", "cand_classical", "cand_learned"),
    ("classical_vs_hybrid", "cand_classical", "cand_hybrid"),
    ("learned_vs_hybrid", "cand_learned", "cand_hybrid"),
    ("hybrid_vs_safe_fallback", "cand_hybrid", "cand_safe_fallback"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate higher-quality candidate-first V2 synthetic data.")
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_variants_per_scene", type=int, default=3)
    parser.add_argument("--val_variants_per_scene", type=int, default=1)
    parser.add_argument("--scenes", nargs="+", default=list(SCENE_LIBRARY.keys()))
    return parser.parse_args()


def clamp01(x):
    return max(0.0, min(1.0, x))


def structured_score(comfort_score, safety_score, tau_safe=0.72, lambda_veto=4.0):
    penalty = max(tau_safe - safety_score, 0.0)
    return comfort_score - lambda_veto * penalty * penalty


def dominant_issue(controls, allowed_keys):
    ranked = sorted(
        ((key, float(value)) for key, value in controls.items() if key in allowed_keys),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[0][0] if ranked else "unclear"


def better_label(a, b, tol=0.03):
    if a > b + tol:
        return "A"
    if b > a + tol:
        return "B"
    return "tie"


def reaction_from_reason(reason_label, safety_subreason, comfort_subreason):
    if reason_label == "safety":
        return "interruption" if safety_subreason == "timing_risk" else "avoidance"
    if reason_label == "comfort":
        return "unnatural_posture" if comfort_subreason == "height_alignment" else "overreach_or_reposition"
    if reason_label == "mixed":
        return "hesitation"
    return "unclear"


def comfort_offset_from_failure(mode, sign):
    if mode == "reachability":
        return [0.16, 0.02 * sign, 0.03]
    if mode == "height_alignment":
        return [0.02, 0.01 * sign, 0.13 * sign]
    if mode == "front_sector_alignment":
        return [0.02, 0.17 * sign, 0.02]
    return [-0.08, 0.04 * sign, -0.08]


def build_controls(scene_spec, source, comfort_mode, safety_mode, variant_idx):
    obstacle = scene_spec["obstacle_proximity_features"]
    reachable = scene_spec["reachable_region_features"]
    obstacle_asym = abs(obstacle["left_clearance"] - obstacle["right_clearance"])
    narrowness = clamp01((1.0 - min(scene_spec["corridor_width"], scene_spec["doorway_width"])) * 0.8)

    comfort_controls = {
        "reachability": 0.12,
        "height_alignment": 0.12,
        "front_sector_alignment": 0.12,
        "posture_cost": 0.10,
        "forward_bias": 0.15 + 0.15 * narrowness,
        "reachability_proxy": reachable["forward_reach_radius"],
    }
    safety_controls = {
        "distance_intrusion": 0.08 + 0.10 * narrowness,
        "speed_risk": 0.10,
        "timing_risk": 0.10,
        "directional_approach_risk": 0.05 + 0.08 * obstacle_asym,
        "base_ee_intrusion": 0.10 + 0.10 * narrowness,
    }

    source_bias = {
        "classical": (-0.02, 0.08),
        "learned": (-0.05, -0.10),
        "hybrid": (0.10, 0.04),
        "safe_fallback": (-0.08, 0.16),
    }[source]
    comfort_controls[comfort_mode] += 0.30 + (0.08 if source == "learned" else 0.0) + 0.03 * variant_idx
    safety_controls[safety_mode] += 0.28 + (0.10 if source == "learned" else 0.0) + 0.04 * variant_idx

    comfort_penalty = (
        0.36 * comfort_controls["reachability"]
        + 0.28 * comfort_controls["height_alignment"]
        + 0.24 * comfort_controls["front_sector_alignment"]
        + 0.22 * comfort_controls["posture_cost"]
    )
    safety_penalty = (
        0.38 * safety_controls["distance_intrusion"]
        + 0.26 * safety_controls["speed_risk"]
        + 0.22 * safety_controls["timing_risk"]
        + 0.14 * safety_controls["directional_approach_risk"]
    )
    comfort_score = clamp01(0.92 - comfort_penalty + source_bias[0])
    safety_score = clamp01(0.92 - safety_penalty + source_bias[1])
    return comfort_controls, safety_controls, comfort_score, safety_score


def choose_reason_and_subreasons(candidate_a, candidate_b, overall_preference):
    if overall_preference == "A":
        loser, winner = candidate_b, candidate_a
    elif overall_preference == "B":
        loser, winner = candidate_a, candidate_b
    else:
        loser, winner = candidate_a, candidate_b

    loser_safety = loser["synthetic_scores"]["safety_score_target"]
    loser_comfort = loser["synthetic_scores"]["comfort_score_target"]
    winner_safety = winner["synthetic_scores"]["safety_score_target"]
    winner_comfort = winner["synthetic_scores"]["comfort_score_target"]

    safety_gap = winner_safety - loser_safety
    comfort_gap = winner_comfort - loser_comfort
    if abs(safety_gap) < 0.04 and abs(comfort_gap) < 0.04:
        reason = "unclear"
    elif loser_safety < 0.62 and safety_gap >= comfort_gap - 0.02:
        reason = "safety"
    elif loser_comfort < 0.62 and comfort_gap > safety_gap + 0.02:
        reason = "comfort"
    else:
        reason = "mixed"
    return reason, loser, winner


def build_pair(context_id, pair_index, pair_name, candidate_a, candidate_b):
    comfort_a = candidate_a["synthetic_scores"]["comfort_score_target"]
    comfort_b = candidate_b["synthetic_scores"]["comfort_score_target"]
    safety_a = candidate_a["synthetic_scores"]["safety_score_target"]
    safety_b = candidate_b["synthetic_scores"]["safety_score_target"]
    score_a = structured_score(comfort_a, safety_a)
    score_b = structured_score(comfort_b, safety_b)
    overall_preference = better_label(score_a, score_b, tol=0.025)
    comfort_better = better_label(comfort_a, comfort_b)
    safety_better = better_label(safety_a, safety_b)
    reason_label, loser, winner = choose_reason_and_subreasons(candidate_a, candidate_b, overall_preference)
    reaction_label = reaction_from_reason(
        reason_label,
        loser["synthetic_scores"]["safety_subreason_label"],
        loser["synthetic_scores"]["comfort_subreason_label"],
    )
    return {
        "pair_id": f"{context_id}_pair_{pair_index:02d}",
        "candidate_a_id": candidate_a["candidate_id"],
        "candidate_b_id": candidate_b["candidate_id"],
        "pair_metadata": {
            "pair_generation_protocol": "same_context_candidate_reranking_plus",
            "comparison_goal": pair_name,
            "pair_scene_control": "same_context_same_goal",
        },
        "labels": {
            "overall_preference": overall_preference,
            "reason_label": reason_label,
            "reaction_label": reaction_label,
            "comfort_better_label": comfort_better,
            "safety_better_label": safety_better,
            "comfort_score_target": {"A": comfort_a, "B": comfort_b},
            "safety_score_target": {"A": safety_a, "B": safety_b},
            "safety_subreason_label": loser["synthetic_scores"]["safety_subreason_label"],
            "comfort_subreason_label": loser["synthetic_scores"]["comfort_subreason_label"],
        },
        "quality_debug": {
            "structured_score_a": round(score_a, 4),
            "structured_score_b": round(score_b, 4),
            "winner_candidate_id": winner["candidate_id"] if overall_preference in {"A", "B"} else "tie",
            "loser_candidate_id": loser["candidate_id"],
        },
    }


def make_scene_spec(scene_name, context_index, split):
    base = json.loads(json.dumps(SCENE_LIBRARY[scene_name]))
    asym_sign = -1 if context_index % 2 == 0 else 1
    base["context_id"] = f"ctx_v2_plus_{scene_name}_{split}_{context_index:03d}"
    base["split"] = split
    base["human_orientation"] = round(base["human_orientation"] + 0.04 * (context_index % 3 - 1), 3)
    base["obstacle_proximity_features"]["left_clearance"] = round(base["obstacle_proximity_features"]["left_clearance"] + 0.03 * asym_sign, 3)
    base["obstacle_proximity_features"]["right_clearance"] = round(base["obstacle_proximity_features"]["right_clearance"] - 0.03 * asym_sign, 3)
    base["notes"] = f"Plus synthetic {scene_name} {split} context {context_index}."
    return base


def make_candidate_profiles(scene_spec, context_index):
    sign = -1 if context_index % 2 == 0 else 1
    scene_offset = list(SCENE_LIBRARY.keys()).index(scene_spec["scene_type"])
    pattern = ["mixed", "comfort", "safety", "unclear"][(context_index + scene_offset) % 4]
    comfort_modes = {
        "cand_classical": COMFORT_FAILURES[(context_index + 0) % len(COMFORT_FAILURES)],
        "cand_learned": COMFORT_FAILURES[(context_index + 1) % len(COMFORT_FAILURES)],
        "cand_hybrid": COMFORT_FAILURES[(context_index + 2) % len(COMFORT_FAILURES)],
        "cand_safe_fallback": COMFORT_FAILURES[(context_index + 3) % len(COMFORT_FAILURES)],
    }
    safety_modes = {
        "cand_classical": SAFETY_FAILURES[(context_index + 0) % len(SAFETY_FAILURES)],
        "cand_learned": SAFETY_FAILURES[(context_index + 1) % len(SAFETY_FAILURES)],
        "cand_hybrid": SAFETY_FAILURES[(context_index + 2) % len(SAFETY_FAILURES)],
        "cand_safe_fallback": SAFETY_FAILURES[(context_index + 0) % len(SAFETY_FAILURES)],
    }
    if pattern == "safety":
        safety_modes["cand_learned"] = "timing_risk"
    source_map = {
        "cand_classical": "classical",
        "cand_learned": "learned",
        "cand_hybrid": "hybrid",
        "cand_safe_fallback": "safe_fallback",
    }

    profiles = {}
    for candidate_id, source in source_map.items():
        comfort_mode = comfort_modes[candidate_id]
        safety_mode = safety_modes[candidate_id]
        comfort_controls, safety_controls, comfort_score, safety_score = build_controls(
            scene_spec, source, comfort_mode, safety_mode, context_index % 3
        )
        handover_offset = comfort_offset_from_failure(comfort_mode, sign)
        if safety_mode == "distance_intrusion":
            handover_offset[0] -= 0.05
        if source == "safe_fallback":
            handover_offset[0] -= 0.04
            handover_offset[2] -= 0.03
        speed_scale = 0.72 if source == "safe_fallback" else (1.12 if safety_mode == "speed_risk" else 0.86)
        duration = 2.25 if source == "safe_fallback" else (1.55 if safety_mode == "timing_risk" else 1.95)
        profiles[candidate_id] = {
            "candidate_source": source,
            "comfort_score": round(comfort_score, 4),
            "safety_score": round(safety_score, 4),
            "comfort_issue": comfort_mode,
            "safety_issue": safety_mode,
            "handover_offset": [round(v, 4) for v in handover_offset],
            "speed_scale": round(speed_scale, 4),
            "duration": round(duration, 3),
            "comfort_controls": comfort_controls,
            "safety_controls": safety_controls,
        }

    if pattern == "comfort":
        profiles["cand_learned"]["comfort_score"] = round(clamp01(profiles["cand_learned"]["comfort_score"] - 0.16), 4)
        profiles["cand_learned"]["safety_score"] = round(clamp01(profiles["cand_learned"]["safety_score"] + 0.10), 4)
        profiles["cand_learned"]["safety_controls"]["speed_risk"] = 0.08
        profiles["cand_learned"]["safety_controls"]["timing_risk"] = 0.08
    elif pattern == "safety":
        profiles["cand_learned"]["comfort_score"] = round(clamp01(profiles["cand_learned"]["comfort_score"] + 0.08), 4)
        profiles["cand_learned"]["safety_score"] = round(clamp01(profiles["cand_learned"]["safety_score"] - 0.18), 4)
        profiles["cand_learned"]["safety_issue"] = "timing_risk"
        profiles["cand_learned"]["safety_controls"]["timing_risk"] = max(
            profiles["cand_learned"]["safety_controls"]["timing_risk"],
            0.52,
        )
    elif pattern == "unclear":
        profiles["cand_hybrid"]["comfort_score"] = 0.71
        profiles["cand_hybrid"]["safety_score"] = 0.78
        profiles["cand_safe_fallback"]["comfort_score"] = 0.70
        profiles["cand_safe_fallback"]["safety_score"] = 0.79
        profiles["cand_hybrid"]["comfort_controls"]["front_sector_alignment"] = 0.18
        profiles["cand_safe_fallback"]["comfort_controls"]["front_sector_alignment"] = 0.19
        profiles["cand_hybrid"]["safety_controls"]["distance_intrusion"] = 0.16
        profiles["cand_safe_fallback"]["safety_controls"]["distance_intrusion"] = 0.15

    for profile in profiles.values():
        profile["comfort_issue"] = dominant_issue(
            profile["comfort_controls"],
            {"reachability", "height_alignment", "front_sector_alignment", "posture_cost"},
        )
        profile["safety_issue"] = dominant_issue(
            profile["safety_controls"],
            {"distance_intrusion", "speed_risk", "timing_risk"},
        )
    return profiles


def build_context_record(scene_name, context_index, split):
    spec = make_scene_spec(scene_name, context_index, split)
    spec["candidate_profiles"] = make_candidate_profiles(spec, context_index)
    context = {
        "scene_type": spec["scene_type"],
        "human_position": spec["human_position"],
        "human_orientation": spec["human_orientation"],
        "human_hand_position": spec["human_hand_position"],
        "human_posture_features": spec["human_posture_features"],
        "robot_initial_pose": spec["robot_initial_pose"],
        "robot_initial_velocity": spec["robot_initial_velocity"],
        "environment_map_id": spec["environment_map_id"],
        "local_costmap_embedding": spec["local_costmap_embedding"],
        "obstacle_proximity_features": spec["obstacle_proximity_features"],
        "reachable_region_features": spec["reachable_region_features"],
        "task_goal": spec["task_goal"],
    }

    candidate_lookup = {}
    candidate_set = []
    for candidate_id, profile in spec["candidate_profiles"].items():
        candidate = build_candidate(spec, candidate_id, profile)
        candidate["synthetic_controls"] = {
            "comfort_controls": profile["comfort_controls"],
            "safety_controls": profile["safety_controls"],
            "environment_controls": {
                "scene_type": spec["scene_type"],
                "corridor_width": spec["corridor_width"],
                "doorway_width": spec["doorway_width"],
                "obstacle_asymmetry": round(abs(spec["obstacle_proximity_features"]["left_clearance"] - spec["obstacle_proximity_features"]["right_clearance"]), 4),
                "reachable_region_span": round(spec["reachable_region_features"]["vertical_band_high"] - spec["reachable_region_features"]["vertical_band_low"], 4),
            },
        }
        candidate_set.append(candidate)
        candidate_lookup[candidate_id] = candidate

    training_pairs = []
    for pair_index, (pair_name, a_id, b_id) in enumerate(PAIR_BLUEPRINTS, start=1):
        training_pairs.append(build_pair(spec["context_id"], pair_index, pair_name, candidate_lookup[a_id], candidate_lookup[b_id]))

    return {
        "context_id": spec["context_id"],
        "context": context,
        "candidate_set": candidate_set,
        "training_pairs": training_pairs,
        "annotator_id": "synthetic_generator_v2_cs_rethinking_plus",
        "notes": spec["notes"],
    }


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    train_records = []
    val_records = []

    for scene_name in args.scenes:
        for idx in range(args.train_variants_per_scene):
            train_records.append(build_context_record(scene_name, idx, "train"))
        for idx in range(args.val_variants_per_scene):
            val_records.append(build_context_record(scene_name, 100 + idx, "val"))

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)

    print(f"Wrote {len(train_records)} train contexts to {output_dir / 'train.jsonl'}")
    print(f"Wrote {len(val_records)} val contexts to {output_dir / 'val.jsonl'}")
    print("Coverage:")
    print(f"  scenes: {args.scenes}")
    print(f"  comfort_failure_modes: {COMFORT_FAILURES}")
    print(f"  safety_failure_modes: {SAFETY_FAILURES}")


if __name__ == "__main__":
    main()
