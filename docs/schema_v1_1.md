# Structured Preference Schema v1.1

This document defines the minimal structured preference format for the first `approach + handover` reward-modeling prototype on top of the existing PrefMMT / APReL baseline.

## Goals

- Keep the current segment-level pairwise preference setup.
- Add explicit candidate-generation metadata inspired by "Rethinking Social Robot Navigation".
- Add reward decomposition metadata inspired by ToolRLA.
- Keep the format simple enough for synthetic data, loader development, and CPU smoke tests.

## Top-Level Sample Format

Each sample is one pair of trajectory segments:

```json
{
  "sample_id": "pair_0001",
  "pair_metadata": {},
  "segment_a": {},
  "segment_b": {},
  "labels": {},
  "reward_decomposition": {},
  "annotator_id": "self_v1",
  "notes": "Optional free-form note"
}
```

## `pair_metadata`

Rethinking-style pair construction metadata:

- `pair_generation_protocol`
  - `same_source_variation`
  - `cross_source_comparison`
  - `hybrid_ablation`
  - `manual_baseline_pair`
- `comparison_goal`
  - `classical_vs_learned`
  - `classical_vs_hybrid`
  - `learned_vs_hybrid`
  - `safe_vs_efficient`
  - `comfortable_vs_efficient`
  - `social_vs_direct`
  - `other`
- `pair_scene_control`
  - `same_scene_same_handover_goal`
  - `same_scene_different_handover_style`
  - `different_scene_exploratory`
- `pair_notes`

## Segment Structure

Each segment must contain:

```json
{
  "segment_id": "seg_0001_a",
  "scene_type": "doorway",
  "segment_start_type": "human_notice",
  "segment_end_type": "handover_success",
  "handover_outcome": "success",
  "candidate_source": "classical",
  "candidate_generator_detail": "rule_based_safe",
  "planner_family": "geometric",
  "hybrid_role": "none",
  "source_description": "Conservative rule-based approach.",
  "context": {},
  "sequences": {}
}
```

### Segment enums

- `scene_type`
  - `open_space`
  - `narrow_corridor`
  - `doorway`
  - `corner`
  - `intersection`
  - `enclosed_space`
  - `table_separated`
- `segment_start_type`
  - `human_notice`
  - `human_first_reaction`
  - `robot_enter_interaction_zone`
- `segment_end_type`
  - `handover_success`
  - `handover_failure`
  - `handover_abort`
- `handover_outcome`
  - `success`
  - `failure`
  - `partial`
  - `unclear`

### Candidate-generation fields

- `candidate_source`
  - `classical`
  - `learned`
  - `hybrid`
  - `manual_baseline`
- `candidate_generator_detail`
  - `rule_based_safe`
  - `rule_based_efficient`
  - `rule_based_social_nominal`
  - `learned_social_style`
  - `learned_direct_style`
  - `hybrid_switch`
  - `hybrid_blend`
  - `manual_good_case`
  - `manual_bad_case`
  - `manual_variant`
- `planner_family`
  - `geometric`
  - `learning_based`
  - `hybrid`
  - `manual`
- `hybrid_role`
  - `none`
  - `geometric_backbone_learning_refinement`
  - `learning_backbone_geometric_guard`
  - `switching_controller`

## `context`

Required fields:

```json
{
  "map_id": "toy_map_01",
  "handover_point_3d": [0.62, 0.15, 1.05],
  "handover_point_relative_to_human": "front_mid",
  "handover_point_relative_to_robot": "front",
  "handover_point_map_zone": "doorway_center",
  "human_initial_relative_position": "front_right",
  "robot_initial_relative_position": "front_left",
  "scene_notes": "Robot approaches in a doorway and hands over in front of the user."
}
```

Suggested categorical values:

- `handover_point_relative_to_human`
  - `front_near`
  - `front_mid`
  - `front_far`
  - `left_near`
  - `right_near`
  - `high_front`
  - `low_front`
  - `far_front_right`
  - `far_front_left`
  - `unclear`
- `handover_point_relative_to_robot`
  - `front`
  - `front_left`
  - `front_right`
  - `left`
  - `right`
  - `high_front`
  - `low_front`
  - `unclear`

## `sequences`

All sequences must use the same segment length:

```json
{
  "time_seq": [0.0, 0.2, 0.4, 0.6],
  "robot_state_seq": [],
  "robot_action_seq": [],
  "base_pose_seq": [],
  "ee_pose_seq": [],
  "velocity_seq": [],
  "gripper_state_seq": [],
  "object_pose_seq": [],
  "human_pose_seq": [],
  "human_hand_pose_seq": [],
  "relative_position_seq": [],
  "relative_distance_seq": [],
  "relative_orientation_seq": []
}
```

The first synthetic release uses simplified low-dimensional numeric sequences but preserves the segment-level sequence layout expected by downstream loaders.

## `labels`

```json
{
  "overall_preference": "A",
  "reason_label": "comfort",
  "reaction_label": "overreach_or_reposition",
  "winner_outcome_alignment": "aligned",
  "dominant_dimension": "comfort",
  "secondary_dimension": "social"
}
```

### Label enums

- `overall_preference`
  - `A`
  - `B`
  - `tie`
- `reason_label`
  - `safety`
  - `comfort`
  - `social`
  - `efficiency`
  - `unclear`
- `reaction_label`
  - `none`
  - `hesitation`
  - `avoidance`
  - `interruption`
  - `overreach_or_reposition`
  - `unnatural_posture`
  - `unclear`
- `winner_outcome_alignment`
  - `aligned`
  - `conflicted`
  - `unclear`
- `dominant_dimension`
  - `safety`
  - `comfort`
  - `social`
  - `efficiency`
  - `unclear`
- `secondary_dimension`
  - `safety`
  - `comfort`
  - `social`
  - `efficiency`
  - `unclear`
  - `none`

## `reward_decomposition`

```json
{
  "dimensions": {
    "safety": {
      "score_a": 5,
      "score_b": 3,
      "better_segment": "A",
      "severity_if_bad": 1,
      "confidence": 0.85,
      "notes": "B is acceptable but slightly closer."
    }
  },
  "priority_order": ["safety", "comfort", "social", "efficiency"],
  "aggregation_rule": "priority_weighted_with_safety_veto",
  "preference_explanation_basis": "dimension_scores_plus_reason_label",
  "decomposition_notes": "Efficiency gain does not outweigh comfort deficit."
}
```

### Dimension requirements

All four dimensions are required:

- `safety`
- `comfort`
- `social`
- `efficiency`

Each dimension must contain:

- `score_a`
- `score_b`
- `better_segment`
- `severity_if_bad`
- `confidence`
- `notes`

### Default aggregation semantics

- `priority_order`: `["safety", "comfort", "social", "efficiency"]`
- `aggregation_rule`: `priority_weighted_with_safety_veto`

The first training version only needs to preserve these fields and load them correctly. Full aggregation logic is intentionally deferred.

## First Synthetic Release

The first synthetic release lives in [`data/synthetic_v1_1`](/home/abc/workspaces/reward_model_projects/data/synthetic_v1_1) and includes:

- `example_pair_0001.json`
- `train.jsonl`
- `val.jsonl`

Current intended use:

- schema validation
- loader/parser development
- label-distribution inspection
- minimal reward-model smoke tests

Deferred to later phases:

- real human annotations
- automatic reaction detection
- policy-learning integration
- decomposition-aware training loss
