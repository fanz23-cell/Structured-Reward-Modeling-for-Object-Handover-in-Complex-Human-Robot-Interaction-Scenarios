import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "synthetic_v2_cs_rethinking"

CANDIDATE_SOURCE_TO_DETAIL = {
    "classical": {
        "candidate_generator_detail": "rule_based_safe_nominal",
        "planner_family": "geometric",
        "hybrid_role": "none",
        "source_description": "Rule-based geometric baseline with explicit handover margins.",
    },
    "learned": {
        "candidate_generator_detail": "learned_style_proposal",
        "planner_family": "learning_based",
        "hybrid_role": "none",
        "source_description": "Learned proposal candidate with style and timing variation.",
    },
    "hybrid": {
        "candidate_generator_detail": "hybrid_backbone_refinement",
        "planner_family": "hybrid",
        "hybrid_role": "geometric_backbone_learning_refinement",
        "source_description": "Geometric backbone with learned local refinement near handover.",
    },
    "safe_fallback": {
        "candidate_generator_detail": "conservative_safe_fallback",
        "planner_family": "geometric",
        "hybrid_role": "none",
        "source_description": "Always-available conservative fallback with extra clearance and slower timing.",
    },
}

PAIR_BLUEPRINTS = [
    ("classical_vs_learned", "cand_classical", "cand_learned"),
    ("classical_vs_hybrid", "cand_classical", "cand_hybrid"),
    ("learned_vs_hybrid", "cand_learned", "cand_hybrid"),
    ("hybrid_vs_safe_fallback", "cand_hybrid", "cand_safe_fallback"),
]

REACTION_LABELS = {
    "safety": "avoidance",
    "comfort": "overreach_or_reposition",
    "mixed": "hesitation",
    "unclear": "unclear",
}

CONTEXT_SPECS = [
    {
        "context_id": "ctx_v2_doorway_001",
        "split": "train",
        "scene_type": "doorway",
        "human_position": [0.0, 0.0, 0.0],
        "human_orientation": 0.0,
        "human_hand_position": [0.24, 0.03, 1.03],
        "human_posture_features": {
            "arm_extension_fraction": 0.56,
            "torso_lean": 0.08,
            "shoulder_height": 1.34,
            "stance_width": 0.29,
        },
        "robot_initial_pose": [-1.10, -0.10, 0.05],
        "robot_initial_velocity": [0.0, 0.0, 0.0],
        "environment_map_id": "toy_map_doorway_01",
        "local_costmap_embedding": [0.15, 0.33, 0.24, 0.18],
        "obstacle_proximity_features": {
            "min_distance_to_obstacle": 0.42,
            "left_clearance": 0.36,
            "right_clearance": 0.28,
        },
        "reachable_region_features": {
            "forward_reach_radius": 0.72,
            "vertical_band_low": 0.90,
            "vertical_band_high": 1.12,
        },
        "task_goal": {
            "goal_type": "approach_and_handover",
            "handover_location_target": [0.66, 0.12, 1.02],
            "handover_location_relative_to_human": "front_mid",
            "handover_timing_constraint": "nominal",
        },
        "candidate_profiles": {
            "cand_classical": {
                "candidate_source": "classical",
                "safety_score": 0.88,
                "comfort_score": 0.73,
                "safety_issue": "distance_intrusion",
                "comfort_issue": "front_sector_alignment",
                "handover_offset": [0.00, 0.03, 0.01],
                "speed_scale": 0.75,
                "duration": 2.0,
            },
            "cand_learned": {
                "candidate_source": "learned",
                "safety_score": 0.57,
                "comfort_score": 0.45,
                "safety_issue": "speed_risk",
                "comfort_issue": "reachability",
                "handover_offset": [0.18, -0.08, 0.12],
                "speed_scale": 1.18,
                "duration": 1.6,
            },
            "cand_hybrid": {
                "candidate_source": "hybrid",
                "safety_score": 0.82,
                "comfort_score": 0.84,
                "safety_issue": "timing_risk",
                "comfort_issue": "height_alignment",
                "handover_offset": [0.02, 0.00, 0.00],
                "speed_scale": 0.88,
                "duration": 1.9,
            },
            "cand_safe_fallback": {
                "candidate_source": "safe_fallback",
                "safety_score": 0.95,
                "comfort_score": 0.60,
                "safety_issue": "unclear",
                "comfort_issue": "posture_cost",
                "handover_offset": [-0.06, 0.04, -0.05],
                "speed_scale": 0.62,
                "duration": 2.3,
            },
        },
        "notes": "Doorway case with safe fallback staying conservative while hybrid improves comfort.",
    },
    {
        "context_id": "ctx_v2_corridor_001",
        "split": "train",
        "scene_type": "narrow_corridor",
        "human_position": [0.0, 0.0, 0.0],
        "human_orientation": 0.15,
        "human_hand_position": [0.22, -0.02, 1.00],
        "human_posture_features": {
            "arm_extension_fraction": 0.60,
            "torso_lean": 0.06,
            "shoulder_height": 1.31,
            "stance_width": 0.25,
        },
        "robot_initial_pose": [-1.20, 0.20, -0.05],
        "robot_initial_velocity": [0.0, 0.0, 0.0],
        "environment_map_id": "toy_map_corridor_01",
        "local_costmap_embedding": [0.26, 0.38, 0.19, 0.31],
        "obstacle_proximity_features": {
            "min_distance_to_obstacle": 0.24,
            "left_clearance": 0.18,
            "right_clearance": 0.22,
        },
        "reachable_region_features": {
            "forward_reach_radius": 0.67,
            "vertical_band_low": 0.88,
            "vertical_band_high": 1.08,
        },
        "task_goal": {
            "goal_type": "approach_and_handover",
            "handover_location_target": [0.60, 0.02, 0.99],
            "handover_location_relative_to_human": "front_mid",
            "handover_timing_constraint": "narrow_space",
        },
        "candidate_profiles": {
            "cand_classical": {
                "candidate_source": "classical",
                "safety_score": 0.76,
                "comfort_score": 0.65,
                "safety_issue": "distance_intrusion",
                "comfort_issue": "front_sector_alignment",
                "handover_offset": [-0.02, 0.02, 0.00],
                "speed_scale": 0.78,
                "duration": 2.1,
            },
            "cand_learned": {
                "candidate_source": "learned",
                "safety_score": 0.49,
                "comfort_score": 0.54,
                "safety_issue": "distance_intrusion",
                "comfort_issue": "front_sector_alignment",
                "handover_offset": [-0.10, 0.18, 0.04],
                "speed_scale": 1.10,
                "duration": 1.7,
            },
            "cand_hybrid": {
                "candidate_source": "hybrid",
                "safety_score": 0.86,
                "comfort_score": 0.78,
                "safety_issue": "timing_risk",
                "comfort_issue": "height_alignment",
                "handover_offset": [0.01, 0.01, 0.01],
                "speed_scale": 0.84,
                "duration": 1.95,
            },
            "cand_safe_fallback": {
                "candidate_source": "safe_fallback",
                "safety_score": 0.92,
                "comfort_score": 0.52,
                "safety_issue": "unclear",
                "comfort_issue": "posture_cost",
                "handover_offset": [-0.07, -0.03, -0.06],
                "speed_scale": 0.58,
                "duration": 2.45,
            },
        },
        "notes": "Corridor case where safety margins dominate even when direct candidates are somewhat convenient.",
    },
    {
        "context_id": "ctx_v2_open_001",
        "split": "train",
        "scene_type": "open_space",
        "human_position": [0.0, 0.0, 0.0],
        "human_orientation": -0.10,
        "human_hand_position": [0.27, 0.01, 1.01],
        "human_posture_features": {
            "arm_extension_fraction": 0.52,
            "torso_lean": 0.04,
            "shoulder_height": 1.35,
            "stance_width": 0.33,
        },
        "robot_initial_pose": [-1.35, -0.22, 0.08],
        "robot_initial_velocity": [0.0, 0.0, 0.0],
        "environment_map_id": "toy_map_open_01",
        "local_costmap_embedding": [0.08, 0.16, 0.12, 0.10],
        "obstacle_proximity_features": {
            "min_distance_to_obstacle": 0.71,
            "left_clearance": 0.78,
            "right_clearance": 0.81,
        },
        "reachable_region_features": {
            "forward_reach_radius": 0.76,
            "vertical_band_low": 0.90,
            "vertical_band_high": 1.14,
        },
        "task_goal": {
            "goal_type": "approach_and_handover",
            "handover_location_target": [0.68, -0.04, 1.03],
            "handover_location_relative_to_human": "front_mid",
            "handover_timing_constraint": "nominal",
        },
        "candidate_profiles": {
            "cand_classical": {
                "candidate_source": "classical",
                "safety_score": 0.84,
                "comfort_score": 0.74,
                "safety_issue": "timing_risk",
                "comfort_issue": "height_alignment",
                "handover_offset": [0.00, -0.02, 0.02],
                "speed_scale": 0.80,
                "duration": 2.0,
            },
            "cand_learned": {
                "candidate_source": "learned",
                "safety_score": 0.73,
                "comfort_score": 0.68,
                "safety_issue": "speed_risk",
                "comfort_issue": "front_sector_alignment",
                "handover_offset": [0.10, -0.12, 0.05],
                "speed_scale": 1.05,
                "duration": 1.75,
            },
            "cand_hybrid": {
                "candidate_source": "hybrid",
                "safety_score": 0.81,
                "comfort_score": 0.88,
                "safety_issue": "timing_risk",
                "comfort_issue": "unclear",
                "handover_offset": [0.02, -0.01, 0.00],
                "speed_scale": 0.87,
                "duration": 1.9,
            },
            "cand_safe_fallback": {
                "candidate_source": "safe_fallback",
                "safety_score": 0.94,
                "comfort_score": 0.63,
                "safety_issue": "unclear",
                "comfort_issue": "posture_cost",
                "handover_offset": [-0.05, -0.03, -0.04],
                "speed_scale": 0.60,
                "duration": 2.35,
            },
        },
        "notes": "Open-space case where hybrid improves comfort while remaining within an acceptable safety band.",
    },
    {
        "context_id": "ctx_v2_table_001",
        "split": "train",
        "scene_type": "table_separated",
        "human_position": [0.0, 0.0, 0.0],
        "human_orientation": 0.02,
        "human_hand_position": [0.20, 0.00, 1.02],
        "human_posture_features": {
            "arm_extension_fraction": 0.68,
            "torso_lean": 0.12,
            "shoulder_height": 1.30,
            "stance_width": 0.30,
        },
        "robot_initial_pose": [-1.00, 0.00, 0.00],
        "robot_initial_velocity": [0.0, 0.0, 0.0],
        "environment_map_id": "toy_map_table_01",
        "local_costmap_embedding": [0.35, 0.27, 0.21, 0.40],
        "obstacle_proximity_features": {
            "min_distance_to_obstacle": 0.34,
            "left_clearance": 0.48,
            "right_clearance": 0.51,
        },
        "reachable_region_features": {
            "forward_reach_radius": 0.61,
            "vertical_band_low": 0.92,
            "vertical_band_high": 1.09,
        },
        "task_goal": {
            "goal_type": "approach_and_handover",
            "handover_location_target": [0.72, 0.00, 1.00],
            "handover_location_relative_to_human": "front_mid",
            "handover_timing_constraint": "table_edge",
        },
        "candidate_profiles": {
            "cand_classical": {
                "candidate_source": "classical",
                "safety_score": 0.79,
                "comfort_score": 0.70,
                "safety_issue": "timing_risk",
                "comfort_issue": "reachability",
                "handover_offset": [0.01, 0.02, 0.01],
                "speed_scale": 0.76,
                "duration": 2.05,
            },
            "cand_learned": {
                "candidate_source": "learned",
                "safety_score": 0.66,
                "comfort_score": 0.42,
                "safety_issue": "speed_risk",
                "comfort_issue": "reachability",
                "handover_offset": [0.21, -0.04, 0.12],
                "speed_scale": 1.08,
                "duration": 1.65,
            },
            "cand_hybrid": {
                "candidate_source": "hybrid",
                "safety_score": 0.83,
                "comfort_score": 0.80,
                "safety_issue": "distance_intrusion",
                "comfort_issue": "height_alignment",
                "handover_offset": [0.00, 0.01, -0.01],
                "speed_scale": 0.82,
                "duration": 1.98,
            },
            "cand_safe_fallback": {
                "candidate_source": "safe_fallback",
                "safety_score": 0.93,
                "comfort_score": 0.49,
                "safety_issue": "unclear",
                "comfort_issue": "posture_cost",
                "handover_offset": [-0.08, 0.03, -0.08],
                "speed_scale": 0.56,
                "duration": 2.55,
            },
        },
        "notes": "Across-table case where comfort is tightly tied to reachability and height alignment.",
    },
    {
        "context_id": "ctx_v2_corner_001",
        "split": "val",
        "scene_type": "corner",
        "human_position": [0.0, 0.0, 0.0],
        "human_orientation": 0.35,
        "human_hand_position": [0.19, 0.06, 1.01],
        "human_posture_features": {
            "arm_extension_fraction": 0.58,
            "torso_lean": 0.07,
            "shoulder_height": 1.29,
            "stance_width": 0.27,
        },
        "robot_initial_pose": [-1.18, -0.25, 0.10],
        "robot_initial_velocity": [0.0, 0.0, 0.0],
        "environment_map_id": "toy_map_corner_01",
        "local_costmap_embedding": [0.29, 0.31, 0.22, 0.27],
        "obstacle_proximity_features": {
            "min_distance_to_obstacle": 0.30,
            "left_clearance": 0.26,
            "right_clearance": 0.42,
        },
        "reachable_region_features": {
            "forward_reach_radius": 0.66,
            "vertical_band_low": 0.90,
            "vertical_band_high": 1.10,
        },
        "task_goal": {
            "goal_type": "approach_and_handover",
            "handover_location_target": [0.61, 0.11, 1.00],
            "handover_location_relative_to_human": "front_right",
            "handover_timing_constraint": "corner_turn",
        },
        "candidate_profiles": {
            "cand_classical": {
                "candidate_source": "classical",
                "safety_score": 0.81,
                "comfort_score": 0.69,
                "safety_issue": "distance_intrusion",
                "comfort_issue": "front_sector_alignment",
                "handover_offset": [-0.01, 0.03, 0.00],
                "speed_scale": 0.79,
                "duration": 2.0,
            },
            "cand_learned": {
                "candidate_source": "learned",
                "safety_score": 0.60,
                "comfort_score": 0.58,
                "safety_issue": "speed_risk",
                "comfort_issue": "height_alignment",
                "handover_offset": [0.12, -0.06, 0.07],
                "speed_scale": 1.04,
                "duration": 1.7,
            },
            "cand_hybrid": {
                "candidate_source": "hybrid",
                "safety_score": 0.85,
                "comfort_score": 0.76,
                "safety_issue": "timing_risk",
                "comfort_issue": "front_sector_alignment",
                "handover_offset": [0.02, 0.01, 0.01],
                "speed_scale": 0.86,
                "duration": 1.9,
            },
            "cand_safe_fallback": {
                "candidate_source": "safe_fallback",
                "safety_score": 0.94,
                "comfort_score": 0.55,
                "safety_issue": "unclear",
                "comfort_issue": "posture_cost",
                "handover_offset": [-0.07, 0.05, -0.05],
                "speed_scale": 0.57,
                "duration": 2.4,
            },
        },
        "notes": "Corner validation case stressing directional approach comfort and timing risk.",
    },
    {
        "context_id": "ctx_v2_intersection_001",
        "split": "val",
        "scene_type": "intersection",
        "human_position": [0.0, 0.0, 0.0],
        "human_orientation": -0.28,
        "human_hand_position": [0.23, -0.05, 1.04],
        "human_posture_features": {
            "arm_extension_fraction": 0.55,
            "torso_lean": 0.05,
            "shoulder_height": 1.36,
            "stance_width": 0.31,
        },
        "robot_initial_pose": [-1.30, 0.12, -0.07],
        "robot_initial_velocity": [0.0, 0.0, 0.0],
        "environment_map_id": "toy_map_intersection_01",
        "local_costmap_embedding": [0.20, 0.24, 0.26, 0.23],
        "obstacle_proximity_features": {
            "min_distance_to_obstacle": 0.40,
            "left_clearance": 0.46,
            "right_clearance": 0.44,
        },
        "reachable_region_features": {
            "forward_reach_radius": 0.74,
            "vertical_band_low": 0.92,
            "vertical_band_high": 1.13,
        },
        "task_goal": {
            "goal_type": "approach_and_handover",
            "handover_location_target": [0.67, -0.08, 1.05],
            "handover_location_relative_to_human": "front_mid",
            "handover_timing_constraint": "crossing_flow",
        },
        "candidate_profiles": {
            "cand_classical": {
                "candidate_source": "classical",
                "safety_score": 0.83,
                "comfort_score": 0.72,
                "safety_issue": "timing_risk",
                "comfort_issue": "height_alignment",
                "handover_offset": [0.01, -0.01, 0.01],
                "speed_scale": 0.81,
                "duration": 2.0,
            },
            "cand_learned": {
                "candidate_source": "learned",
                "safety_score": 0.63,
                "comfort_score": 0.47,
                "safety_issue": "speed_risk",
                "comfort_issue": "front_sector_alignment",
                "handover_offset": [0.14, -0.12, 0.06],
                "speed_scale": 1.09,
                "duration": 1.68,
            },
            "cand_hybrid": {
                "candidate_source": "hybrid",
                "safety_score": 0.88,
                "comfort_score": 0.79,
                "safety_issue": "distance_intrusion",
                "comfort_issue": "unclear",
                "handover_offset": [0.01, 0.00, 0.00],
                "speed_scale": 0.85,
                "duration": 1.92,
            },
            "cand_safe_fallback": {
                "candidate_source": "safe_fallback",
                "safety_score": 0.96,
                "comfort_score": 0.58,
                "safety_issue": "unclear",
                "comfort_issue": "posture_cost",
                "handover_offset": [-0.06, 0.03, -0.05],
                "speed_scale": 0.55,
                "duration": 2.42,
            },
        },
        "notes": "Intersection validation case with stronger timing pressure near the crossing point.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate V2 synthetic structured preference data with context + candidate_set schema."
    )
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def rotation_2d(yaw):
    return [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]


def rotate_xy(xy, yaw):
    rot = rotation_2d(yaw)
    return [
        rot[0][0] * xy[0] + rot[0][1] * xy[1],
        rot[1][0] * xy[0] + rot[1][1] * xy[1],
    ]


def linspace_vec(start, end, count):
    values = []
    for idx in range(count):
        alpha = idx / (count - 1) if count > 1 else 1.0
        values.append([round(start[i] + alpha * (end[i] - start[i]), 4) for i in range(len(start))])
    return values


def quat_from_yaw(yaw):
    return [0.0, 0.0, round(math.sin(yaw / 2.0), 4), round(math.cos(yaw / 2.0), 4)]


def make_human_pose_seq(spec, steps):
    base = spec["human_position"]
    shoulder_height = spec["human_posture_features"]["shoulder_height"]
    return [
        [
            round(base[0], 4),
            round(base[1], 4),
            round(shoulder_height, 4),
            round(spec["human_orientation"], 4),
            round(spec["human_posture_features"]["torso_lean"], 4),
            round(spec["human_posture_features"]["arm_extension_fraction"], 4),
        ]
        for _ in range(steps)
    ]


def make_human_hand_pose_seq(spec, handover_point, steps):
    hand = spec["human_hand_position"]
    return linspace_vec(hand, [handover_point[0] - 0.03, handover_point[1], handover_point[2]], steps)


def build_candidate(spec, candidate_id, profile):
    cfg = CANDIDATE_SOURCE_TO_DETAIL[profile["candidate_source"]]
    goal = spec["task_goal"]["handover_location_target"]
    handover = [round(goal[i] + profile["handover_offset"][i], 4) for i in range(3)]
    steps = 8
    duration = profile["duration"]
    time_seq = [round(duration * idx / (steps - 1), 3) for idx in range(steps)]
    start_base = spec["robot_initial_pose"]
    yaw_start = start_base[2]
    yaw_end = spec["human_orientation"] * 0.6
    base_end = [handover[0] - 0.32, handover[1] - 0.02, yaw_end]
    base_seq_xyz = linspace_vec(start_base[:2] + [0.0], [base_end[0], base_end[1], 0.0], steps)
    base_yaw_seq = [round(yaw_start + idx * (yaw_end - yaw_start) / (steps - 1), 4) for idx in range(steps)]
    robot_base_pose_seq = [
        [base_seq_xyz[idx][0], base_seq_xyz[idx][1], 0.0] + quat_from_yaw(base_yaw_seq[idx])
        for idx in range(steps)
    ]

    ee_start = [start_base[0] + 0.28, start_base[1] + 0.02, 0.82]
    ee_seq_xyz = linspace_vec(ee_start, handover, steps)
    robot_ee_pose_seq = [
        [ee_seq_xyz[idx][0], ee_seq_xyz[idx][1], ee_seq_xyz[idx][2]] + quat_from_yaw(base_yaw_seq[idx] * 0.7)
        for idx in range(steps)
    ]

    robot_velocity_seq = []
    for idx in range(steps):
        phase = 1.0 - idx / (steps - 1)
        vx = round(0.22 * profile["speed_scale"] * max(phase, 0.15), 4)
        vy = round(0.08 * profile["speed_scale"] * max(phase, 0.12), 4)
        vz = round(0.04 * profile["speed_scale"] * max(phase, 0.10), 4)
        robot_velocity_seq.append([vx, vy, vz])

    robot_action_seq = []
    for idx in range(steps):
        alpha = idx / (steps - 1)
        robot_action_seq.append(
            [
                round(0.25 + 0.22 * alpha * profile["speed_scale"], 4),
                round(0.15 + 0.10 * alpha, 4),
            ]
        )

    human_pose_seq = make_human_pose_seq(spec, steps)
    human_hand_pose_seq = make_human_hand_pose_seq(spec, handover, steps)
    object_pose_seq = [[pose[0], pose[1], pose[2] - 0.02] for pose in robot_ee_pose_seq]

    relative_position_seq = []
    relative_distance_seq = []
    relative_orientation_seq = []
    for pose in robot_ee_pose_seq:
        rel = [
            round(pose[0] - spec["human_position"][0], 4),
            round(pose[1] - spec["human_position"][1], 4),
            round(pose[2] - spec["human_hand_position"][2], 4),
        ]
        relative_position_seq.append(rel)
        relative_distance_seq.append(round(math.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2), 4))
        relative_orientation_seq.append(round(yaw_end - spec["human_orientation"], 4))

    robot_state_seq = []
    gripper_state_seq = []
    for idx in range(steps):
        base_pose = robot_base_pose_seq[idx]
        ee_pose = robot_ee_pose_seq[idx]
        robot_state_seq.append(
            [
                base_pose[0],
                base_pose[1],
                round(base_yaw_seq[idx], 4),
                ee_pose[0],
                ee_pose[1],
                ee_pose[2],
                profile["safety_score"],
                profile["comfort_score"],
            ]
        )
        gripper_state_seq.append([round(1.0 - 0.08 * idx, 4)])

    feasibility_flags = {
        "collision_free": True,
        "kinematically_feasible": True,
        "handover_reachable": profile["comfort_score"] >= 0.48,
        "distance_margin_ok": profile["safety_issue"] != "distance_intrusion",
        "speed_ok": profile["safety_issue"] != "speed_risk",
        "timing_ok": profile["safety_issue"] != "timing_risk",
        "safe_fallback_ready": profile["candidate_source"] == "safe_fallback",
    }

    return {
        "candidate_id": candidate_id,
        "candidate_source": profile["candidate_source"],
        "candidate_generator_detail": cfg["candidate_generator_detail"],
        "planner_family": cfg["planner_family"],
        "hybrid_role": cfg["hybrid_role"],
        "source_description": cfg["source_description"],
        "trajectory": {
            "time_seq": time_seq,
            "robot_state_seq": robot_state_seq,
            "robot_action_seq": robot_action_seq,
            "robot_base_pose_seq": robot_base_pose_seq,
            "robot_ee_pose_seq": robot_ee_pose_seq,
            "robot_velocity_seq": robot_velocity_seq,
            "gripper_state_seq": gripper_state_seq,
            "object_pose_seq": object_pose_seq,
            "human_pose_seq": human_pose_seq,
            "human_hand_pose_seq": human_hand_pose_seq,
            "relative_position_seq": relative_position_seq,
            "relative_distance_seq": relative_distance_seq,
            "relative_orientation_seq": relative_orientation_seq,
        },
        "handover_pose": robot_ee_pose_seq[-1],
        "handover_timing": time_seq[-1],
        "handover_time_context": {
            "handover_index": steps - 1,
            "pre_handover_indices": [steps - 3, steps - 2],
            "post_handover_indices": [steps - 1],
        },
        "feasibility_flags": feasibility_flags,
        "synthetic_scores": {
            "comfort_score_target": round(profile["comfort_score"], 4),
            "safety_score_target": round(profile["safety_score"], 4),
            "comfort_subreason_label": profile["comfort_issue"],
            "safety_subreason_label": profile["safety_issue"],
        },
    }


def structured_score(score_bundle, tau_safe=0.72, lambda_veto=4.0):
    safety = score_bundle["safety_score_target"]
    comfort = score_bundle["comfort_score_target"]
    penalty = max(tau_safe - safety, 0.0)
    return comfort - lambda_veto * penalty * penalty


def better_label(score_a, score_b, tolerance=0.03):
    if score_a > score_b + tolerance:
        return "A"
    if score_b > score_a + tolerance:
        return "B"
    return "tie"


def choose_reason(candidate_a, candidate_b):
    diff_safety = candidate_a["synthetic_scores"]["safety_score_target"] - candidate_b["synthetic_scores"]["safety_score_target"]
    diff_comfort = candidate_a["synthetic_scores"]["comfort_score_target"] - candidate_b["synthetic_scores"]["comfort_score_target"]
    abs_safety = abs(diff_safety)
    abs_comfort = abs(diff_comfort)
    if abs_safety < 0.04 and abs_comfort < 0.04:
        return "unclear"
    if abs_safety >= 0.08 and abs_comfort >= 0.08:
        return "mixed"
    return "safety" if abs_safety >= abs_comfort else "comfort"


def build_pair(context_id, pair_index, pair_name, candidate_a, candidate_b):
    score_a = structured_score(candidate_a["synthetic_scores"])
    score_b = structured_score(candidate_b["synthetic_scores"])
    overall_preference = better_label(score_a, score_b, tolerance=0.025)
    comfort_better = better_label(
        candidate_a["synthetic_scores"]["comfort_score_target"],
        candidate_b["synthetic_scores"]["comfort_score_target"],
    )
    safety_better = better_label(
        candidate_a["synthetic_scores"]["safety_score_target"],
        candidate_b["synthetic_scores"]["safety_score_target"],
    )

    if overall_preference == "A":
        loser = candidate_b
        winner = candidate_a
    elif overall_preference == "B":
        loser = candidate_a
        winner = candidate_b
    else:
        loser = candidate_a if structured_score(candidate_a["synthetic_scores"]) <= structured_score(candidate_b["synthetic_scores"]) else candidate_b
        winner = candidate_b if loser is candidate_a else candidate_a

    reason_label = choose_reason(winner, loser)
    reaction_label = REACTION_LABELS[reason_label]

    return {
        "pair_id": f"{context_id}_pair_{pair_index:02d}",
        "candidate_a_id": candidate_a["candidate_id"],
        "candidate_b_id": candidate_b["candidate_id"],
        "pair_metadata": {
            "pair_generation_protocol": "same_context_candidate_reranking",
            "comparison_goal": pair_name,
            "pair_scene_control": "same_context_same_goal",
        },
        "labels": {
            "overall_preference": overall_preference,
            "reason_label": reason_label,
            "reaction_label": reaction_label,
            "comfort_better_label": comfort_better,
            "safety_better_label": safety_better,
            "comfort_score_target": {
                "A": candidate_a["synthetic_scores"]["comfort_score_target"],
                "B": candidate_b["synthetic_scores"]["comfort_score_target"],
            },
            "safety_score_target": {
                "A": candidate_a["synthetic_scores"]["safety_score_target"],
                "B": candidate_b["synthetic_scores"]["safety_score_target"],
            },
            "safety_subreason_label": loser["synthetic_scores"]["safety_subreason_label"],
            "comfort_subreason_label": loser["synthetic_scores"]["comfort_subreason_label"],
        },
        "synthetic_debug": {
            "structured_score_a": round(score_a, 4),
            "structured_score_b": round(score_b, 4),
            "winner_candidate_id": winner["candidate_id"] if overall_preference in {"A", "B"} else "tie",
            "loser_candidate_id": loser["candidate_id"],
        },
    }


def build_context_record(spec):
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

    candidates = []
    candidate_lookup = {}
    for candidate_id, profile in spec["candidate_profiles"].items():
        candidate = build_candidate(spec, candidate_id, profile)
        candidates.append(candidate)
        candidate_lookup[candidate_id] = candidate

    training_pairs = []
    for pair_index, (pair_name, a_id, b_id) in enumerate(PAIR_BLUEPRINTS, start=1):
        training_pairs.append(build_pair(spec["context_id"], pair_index, pair_name, candidate_lookup[a_id], candidate_lookup[b_id]))

    return {
        "context_id": spec["context_id"],
        "context": context,
        "candidate_set": candidates,
        "training_pairs": training_pairs,
        "annotator_id": "synthetic_generator_v2_cs_rethinking",
        "notes": spec["notes"],
    }, spec["split"]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    splits = {"train": [], "val": []}
    for spec in CONTEXT_SPECS:
        record, split = build_context_record(spec)
        splits[split].append(record)

    write_jsonl(output_dir / "train.jsonl", splits["train"])
    write_jsonl(output_dir / "val.jsonl", splits["val"])

    print(f"Wrote {len(splits['train'])} train contexts to {output_dir / 'train.jsonl'}")
    print(f"Wrote {len(splits['val'])} val contexts to {output_dir / 'val.jsonl'}")
    print(f"Total contexts: {sum(len(records) for records in splits.values())}")


if __name__ == "__main__":
    main()
