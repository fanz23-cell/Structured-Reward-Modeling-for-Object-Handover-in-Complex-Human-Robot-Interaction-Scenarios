# Structured Preference V2 Schema Reference (JSON Examples)

This file provides concrete JSON examples for the v2.0 schema.

## Minimal Example: Single Context with 4 Candidates and 2 Training Pairs

```json
{
  "context": {
    "context_id": "ctx_v2_doorway_001",
    "scene_type": "doorway",
    "human_position": [0.0, 0.0, 0.0],
    "human_orientation": [0.0, 0.0, 0.707, 0.707],
    "human_hand_position": [0.2, 0.0, 1.0],
    "human_posture_features": {
      "reaching_height": 1.0,
      "body_lean_forward": 0.1,
      "arm_extension_fraction": 0.6
    },
    "robot_initial_pose": [
      -1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    ],
    "robot_initial_velocity": [0.0, 0.0, 0.0],
    "environment_map_id": "toy_map_doorway_01",
    "local_costmap_embedding": [0.1, 0.2, 0.15],
    "obstacle_proximity_features": {
      "min_distance_to_obstacle": 0.5,
      "obstacle_count_nearby": 2
    },
    "task_goal": {
      "handover_location_target": [0.6, 0.15, 1.05],
      "handover_location_relative_to_human": "front_mid",
      "handover_timing_constraint": "nominal",
      "additional_notes": "Doorway handover in familiar environment"
    }
  },
  "candidate_set": [
    {
      "candidate_id": "cand_001",
      "candidate_source": "classical",
      "candidate_generator_detail": "rule_based_safe",
      "planner_family": "geometric",
      "hybrid_role": "none",
      "trajectory": {
        "time_seq": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
        "robot_base_pose_seq": [
          [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
          [-0.8, 0.1, 0.0, 0.0, 0.0, 0.1, 0.995],
          [-0.6, 0.15, 0.0, 0.0, 0.0, 0.2, 0.98],
          [-0.4, 0.18, 0.0, 0.0, 0.0, 0.25, 0.968],
          [-0.2, 0.19, 0.0, 0.0, 0.0, 0.25, 0.968],
          [0.0, 0.18, 0.0, 0.0, 0.0, 0.2, 0.98],
          [0.2, 0.15, 0.0, 0.0, 0.0, 0.1, 0.995],
          [0.4, 0.12, 0.0, 0.0, 0.0, 0.0, 1.0],
          [0.5, 0.12, 0.0, 0.0, 0.0, 0.0, 1.0],
          [0.6, 0.15, 0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        "robot_ee_pose_seq": [
          [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
          [0.2, 0.1, 0.85, 0.0, 0.0, 0.05, 0.999],
          [0.4, 0.15, 0.9, 0.0, 0.0, 0.1, 0.995],
          [0.55, 0.18, 0.98, 0.0, 0.0, 0.15, 0.989],
          [0.58, 0.19, 1.02, 0.0, 0.0, 0.15, 0.989],
          [0.6, 0.18, 1.04, 0.0, 0.0, 0.1, 0.995],
          [0.6, 0.15, 1.05, 0.0, 0.0, 0.05, 0.999],
          [0.6, 0.15, 1.05, 0.0, 0.0, 0.0, 1.0],
          [0.6, 0.15, 1.05, 0.0, 0.0, 0.0, 1.0],
          [0.6, 0.15, 1.05, 0.0, 0.0, 0.0, 1.0]
        ],
        "robot_velocity_seq": [
          [0.1, 0.05, 0.0],
          [0.15, 0.06, 0.0],
          [0.18, 0.07, 0.0],
          [0.15, 0.08, 0.0],
          [0.08, 0.05, 0.0],
          [0.04, 0.03, 0.0],
          [0.01, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]
        ],
        "robot_action_seq": [
          [0.3, 0.15],
          [0.35, 0.2],
          [0.4, 0.25],
          [0.45, 0.3],
          [0.48, 0.32],
          [0.5, 0.33],
          [0.5, 0.33],
          [0.5, 0.33],
          [0.5, 0.33],
          [0.5, 0.33]
        ]
      },
      "handover_pose": [0.6, 0.15, 1.05, 0.0, 0.0, 0.0, 1.0],
      "handover_timing": 1.8,
      "feasibility_flags": {
        "no_collision": true,
        "distance_margin_ok": true,
        "speed_ok": true,
        "within_workspace": true,
        "human_reachable": true
      }
    },
    {
      "candidate_id": "cand_002",
      "candidate_source": "learned",
      "candidate_generator_detail": "learned_direct_style",
      "planner_family": "learning_based",
      "hybrid_role": "none",
      "trajectory": {
        "time_seq": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5],
        "robot_base_pose_seq": [
          [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
          [-0.75, 0.05, 0.0, 0.0, 0.0, 0.15, 0.989],
          [-0.5, 0.12, 0.0, 0.0, 0.0, 0.25, 0.968],
          [-0.25, 0.16, 0.0, 0.0, 0.0, 0.3, 0.955],
          [0.0, 0.18, 0.0, 0.0, 0.0, 0.28, 0.961],
          [0.25, 0.17, 0.0, 0.0, 0.0, 0.2, 0.98],
          [0.5, 0.15, 0.0, 0.0, 0.0, 0.08, 0.997],
          [0.7, 0.14, 0.0, 0.0, 0.0, -0.05, 0.999],
          [0.8, 0.14, 0.0, 0.0, 0.0, -0.1, 0.995]
        ],
        "robot_ee_pose_seq": [
          [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
          [0.25, 0.08, 0.88, 0.0, 0.0, 0.1, 0.995],
          [0.45, 0.14, 0.95, 0.0, 0.0, 0.18, 0.984],
          [0.62, 0.18, 1.04, 0.0, 0.0, 0.22, 0.976],
          [0.75, 0.19, 1.1, 0.0, 0.0, 0.2, 0.98],
          [0.82, 0.18, 1.12, 0.0, 0.0, 0.15, 0.989],
          [0.86, 0.16, 1.15, 0.0, 0.0, 0.08, 0.997],
          [0.88, 0.15, 1.16, 0.0, 0.0, 0.02, 0.9998],
          [0.88, 0.15, 1.16, 0.0, 0.0, 0.02, 0.9998]
        ],
        "robot_velocity_seq": [
          [0.25, 0.08, 0.0],
          [0.3, 0.1, 0.0],
          [0.35, 0.12, 0.0],
          [0.32, 0.11, 0.0],
          [0.25, 0.09, 0.0],
          [0.15, 0.05, 0.0],
          [0.08, 0.02, 0.0],
          [0.02, 0.0, 0.0],
          [0.0, 0.0, 0.0]
        ],
        "robot_action_seq": [
          [0.35, 0.2],
          [0.4, 0.25],
          [0.45, 0.3],
          [0.5, 0.35],
          [0.52, 0.36],
          [0.52, 0.36],
          [0.52, 0.36],
          [0.52, 0.36],
          [0.52, 0.36]
        ]
      },
      "handover_pose": [0.88, 0.15, 1.16, 0.0, 0.0, 0.02, 0.9998],
      "handover_timing": 1.5,
      "feasibility_flags": {
        "no_collision": true,
        "distance_margin_ok": false,
        "speed_ok": false,
        "within_workspace": true,
        "human_reachable": false
      }
    },
    {
      "candidate_id": "cand_003",
      "candidate_source": "hybrid",
      "candidate_generator_detail": "hybrid_blend",
      "planner_family": "hybrid",
      "hybrid_role": "geometric_backbone_learning_refinement",
      "trajectory": {
        "time_seq": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.7],
        "robot_base_pose_seq": [
          [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
          [-0.8, 0.08, 0.0, 0.0, 0.0, 0.08, 0.997],
          [-0.6, 0.14, 0.0, 0.0, 0.0, 0.15, 0.989],
          [-0.4, 0.17, 0.0, 0.0, 0.0, 0.22, 0.976],
          [-0.15, 0.18, 0.0, 0.0, 0.0, 0.22, 0.976],
          [0.1, 0.17, 0.0, 0.0, 0.0, 0.18, 0.984],
          [0.35, 0.14, 0.0, 0.0, 0.0, 0.1, 0.995],
          [0.55, 0.13, 0.0, 0.0, 0.0, 0.03, 0.9995],
          [0.6, 0.14, 0.0, 0.0, 0.0, 0.0, 1.0],
          [0.62, 0.15, 0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        "robot_ee_pose_seq": [
          [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
          [0.2, 0.08, 0.86, 0.0, 0.0, 0.08, 0.997],
          [0.38, 0.14, 0.92, 0.0, 0.0, 0.14, 0.99],
          [0.54, 0.17, 1.0, 0.0, 0.0, 0.2, 0.98],
          [0.65, 0.18, 1.05, 0.0, 0.0, 0.2, 0.98],
          [0.69, 0.17, 1.07, 0.0, 0.0, 0.16, 0.987],
          [0.67, 0.15, 1.06, 0.0, 0.0, 0.08, 0.997],
          [0.63, 0.14, 1.04, 0.0, 0.0, 0.02, 0.9998],
          [0.62, 0.145, 1.05, 0.0, 0.0, 0.01, 1.0],
          [0.62, 0.15, 1.05, 0.0, 0.0, 0.0, 1.0]
        ],
        "robot_velocity_seq": [
          [0.18, 0.07, 0.0],
          [0.2, 0.08, 0.0],
          [0.22, 0.09, 0.0],
          [0.2, 0.08, 0.0],
          [0.15, 0.06, 0.0],
          [0.12, 0.04, 0.0],
          [0.08, 0.02, 0.0],
          [0.03, 0.01, 0.0],
          [0.01, 0.0, 0.0],
          [0.0, 0.0, 0.0]
        ],
        "robot_action_seq": [
          [0.32, 0.18],
          [0.35, 0.22],
          [0.4, 0.27],
          [0.43, 0.31],
          [0.46, 0.32],
          [0.48, 0.33],
          [0.49, 0.33],
          [0.5, 0.33],
          [0.5, 0.33],
          [0.5, 0.33]
        ]
      },
      "handover_pose": [0.62, 0.15, 1.05, 0.0, 0.0, 0.0, 1.0],
      "handover_timing": 1.7,
      "feasibility_flags": {
        "no_collision": true,
        "distance_margin_ok": true,
        "speed_ok": true,
        "within_workspace": true,
        "human_reachable": true
      }
    },
    {
      "candidate_id": "cand_004",
      "candidate_source": "safe_fallback",
      "candidate_generator_detail": "manual_conservative",
      "planner_family": "geometric",
      "hybrid_role": "none",
      "trajectory": {
        "time_seq": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
        "robot_base_pose_seq": [
          [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
          [-0.85, 0.05, 0.0, 0.0, 0.0, 0.05, 0.999],
          [-0.7, 0.1, 0.0, 0.0, 0.0, 0.1, 0.995],
          [-0.55, 0.13, 0.0, 0.0, 0.0, 0.15, 0.989],
          [-0.4, 0.15, 0.0, 0.0, 0.0, 0.15, 0.989],
          [-0.25, 0.15, 0.0, 0.0, 0.0, 0.1, 0.995],
          [-0.1, 0.14, 0.0, 0.0, 0.0, 0.05, 0.999],
          [0.05, 0.13, 0.0, 0.0, 0.0, 0.0, 1.0],
          [0.2, 0.13, 0.0, 0.0, 0.0, -0.05, 0.999],
          [0.35, 0.14, 0.0, 0.0, 0.0, -0.08, 0.997],
          [0.5, 0.15, 0.0, 0.0, 0.0, -0.05, 0.999],
          [0.62, 0.15, 0.0, 0.0, 0.0, 0.0, 1.0]
        ],
        "robot_ee_pose_seq": [
          [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
          [0.15, 0.05, 0.83, 0.0, 0.0, 0.05, 0.999],
          [0.3, 0.1, 0.86, 0.0, 0.0, 0.1, 0.995],
          [0.45, 0.13, 0.91, 0.0, 0.0, 0.12, 0.993],
          [0.55, 0.15, 0.96, 0.0, 0.0, 0.12, 0.993],
          [0.58, 0.15, 1.0, 0.0, 0.0, 0.08, 0.997],
          [0.6, 0.14, 1.02, 0.0, 0.0, 0.05, 0.999],
          [0.6, 0.13, 1.04, 0.0, 0.0, 0.02, 0.9998],
          [0.6, 0.13, 1.05, 0.0, 0.0, 0.0, 1.0],
          [0.6, 0.145, 1.05, 0.0, 0.0, 0.0, 1.0],
          [0.62, 0.15, 1.05, 0.0, 0.0, 0.0, 1.0],
          [0.62, 0.15, 1.05, 0.0, 0.0, 0.0, 1.0]
        ],
        "robot_velocity_seq": [
          [0.1, 0.03, 0.0],
          [0.12, 0.04, 0.0],
          [0.14, 0.045, 0.0],
          [0.14, 0.045, 0.0],
          [0.12, 0.04, 0.0],
          [0.1, 0.03, 0.0],
          [0.08, 0.02, 0.0],
          [0.05, 0.01, 0.0],
          [0.04, 0.005, 0.0],
          [0.02, 0.0, 0.0],
          [0.01, 0.0, 0.0],
          [0.0, 0.0, 0.0]
        ],
        "robot_action_seq": [
          [0.28, 0.15],
          [0.3, 0.18],
          [0.32, 0.2],
          [0.33, 0.22],
          [0.34, 0.24],
          [0.35, 0.26],
          [0.36, 0.28],
          [0.38, 0.3],
          [0.42, 0.31],
          [0.46, 0.32],
          [0.49, 0.33],
          [0.5, 0.33]
        ]
      },
      "handover_pose": [0.62, 0.15, 1.05, 0.0, 0.0, 0.0, 1.0],
      "handover_timing": 2.2,
      "feasibility_flags": {
        "no_collision": true,
        "distance_margin_ok": true,
        "speed_ok": true,
        "within_workspace": true,
        "human_reachable": true
      }
    }
  ],
  "training_pairs": [
    {
      "pair_id": "pair_001_vs_002",
      "candidate_a_id": "cand_001",
      "candidate_b_id": "cand_002",
      "labels": {
        "overall_preference": "A",
        "reason_label": "comfort",
        "reaction_label": "overreach_or_reposition",
        "comfort_better_label": "A",
        "safety_better_label": "A",
        "comfort_score_target": null,
        "safety_score_target": null,
        "safety_subreason_label": null,
        "comfort_subreason_label": "reachability"
      }
    },
    {
      "pair_id": "pair_001_vs_003",
      "candidate_a_id": "cand_001",
      "candidate_b_id": "cand_003",
      "labels": {
        "overall_preference": "tie",
        "reason_label": "unclear",
        "reaction_label": "none",
        "comfort_better_label": "A",
        "safety_better_label": "B",
        "comfort_score_target": null,
        "safety_score_target": null,
        "safety_subreason_label": null,
        "comfort_subreason_label": null
      }
    }
  ],
  "annotator_id": "synthetic_v2_0_generator",
  "notes": "V2.0 doorway handover scenario with 4 candidates and 2 training pairs"
}
```

## Key Observations

1. **Context** is the world state that remains constant across all candidates.
2. **Candidate set** includes multiple solutions (classical, learned, hybrid, safe_fallback).
3. **Training pairs** are extracted from the candidate set (not pre-generated).
4. **Labels** are defined per pair, with optional `score_target` fields (null in this example).
5. **Decomposition labels** (`safety_subreason_label`, `comfort_subreason_label`) are optional and often null for real data.

