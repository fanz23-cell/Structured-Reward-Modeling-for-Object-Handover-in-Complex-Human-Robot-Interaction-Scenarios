import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "synthetic_v1_1"
DIMENSIONS = ["safety", "comfort", "social", "efficiency"]


SOURCE_CONFIGS = {
    "classical_safe": {
        "candidate_source": "classical",
        "candidate_generator_detail": "rule_based_safe",
        "planner_family": "geometric",
        "hybrid_role": "none",
        "source_description": "Rule-based conservative approach with explicit stopping-distance margins.",
    },
    "classical_efficient": {
        "candidate_source": "classical",
        "candidate_generator_detail": "rule_based_efficient",
        "planner_family": "geometric",
        "hybrid_role": "none",
        "source_description": "Rule-based direct approach with short path and minimal waiting.",
    },
    "classical_social": {
        "candidate_source": "classical",
        "candidate_generator_detail": "rule_based_social_nominal",
        "planner_family": "geometric",
        "hybrid_role": "none",
        "source_description": "Rule-based social-style approach with nominal front-facing presentation.",
    },
    "learned_social": {
        "candidate_source": "learned",
        "candidate_generator_detail": "learned_social_style",
        "planner_family": "learning_based",
        "hybrid_role": "none",
        "source_description": "Learned policy that favors readable front-facing presentation and smooth timing.",
    },
    "learned_direct": {
        "candidate_source": "learned",
        "candidate_generator_detail": "learned_direct_style",
        "planner_family": "learning_based",
        "hybrid_role": "none",
        "source_description": "Learned direct policy that prioritizes speed and short trajectories.",
    },
    "hybrid_switch": {
        "candidate_source": "hybrid",
        "candidate_generator_detail": "hybrid_switch",
        "planner_family": "hybrid",
        "hybrid_role": "switching_controller",
        "source_description": "Hybrid controller that switches between geometric guards and learned refinement.",
    },
    "hybrid_blend": {
        "candidate_source": "hybrid",
        "candidate_generator_detail": "hybrid_blend",
        "planner_family": "hybrid",
        "hybrid_role": "geometric_backbone_learning_refinement",
        "source_description": "Hybrid controller with geometric backbone and learned social refinement.",
    },
    "manual_good": {
        "candidate_source": "manual_baseline",
        "candidate_generator_detail": "manual_good_case",
        "planner_family": "manual",
        "hybrid_role": "none",
        "source_description": "Hand-authored nominal baseline with comfortable presentation pose.",
    },
    "manual_bad": {
        "candidate_source": "manual_baseline",
        "candidate_generator_detail": "manual_bad_case",
        "planner_family": "manual",
        "hybrid_role": "none",
        "source_description": "Hand-authored stress case with awkward distance or timing.",
    },
}


SCENE_CONFIGS = {
    "doorway": {
        "map_id": "toy_map_doorway_01",
        "handover_point_map_zone": "doorway_center",
        "human_initial_relative_position": "front_right",
        "robot_initial_relative_position": "front_left",
    },
    "open_space": {
        "map_id": "toy_map_open_01",
        "handover_point_map_zone": "open_space_center",
        "human_initial_relative_position": "front",
        "robot_initial_relative_position": "front_left",
    },
    "narrow_corridor": {
        "map_id": "toy_map_corridor_01",
        "handover_point_map_zone": "corridor_left_side",
        "human_initial_relative_position": "front",
        "robot_initial_relative_position": "front_right",
    },
    "table_separated": {
        "map_id": "toy_map_table_01",
        "handover_point_map_zone": "table_edge",
        "human_initial_relative_position": "across_table",
        "robot_initial_relative_position": "front",
    },
}


SPECS = [
    {
        "id": 1,
        "split": "train",
        "scene_type": "doorway",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "classical_vs_learned",
            "pair_scene_control": "same_scene_same_handover_goal",
            "pair_notes": "Doorway handover with same target point but different candidate styles.",
        },
        "segment_a": {
            "source_key": "classical_safe",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.62, 0.15, 1.05],
            "scores": {"safety": 5, "comfort": 5, "social": 4, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Comfortable doorway stop with readable front-mid presentation.",
        },
        "segment_b": {
            "source_key": "learned_direct",
            "handover_point_relative_to_human": "front_far",
            "handover_point_relative_to_robot": "front_left",
            "handover_point_3d": [0.86, -0.05, 1.18],
            "scores": {"safety": 3, "comfort": 2, "social": 3, "efficiency": 5},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Faster learned path but handover point is farther and slightly high.",
        },
        "labels": {
            "overall_preference": "A",
            "reason_label": "comfort",
            "reaction_label": "overreach_or_reposition",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "comfort",
            "secondary_dimension": "social",
        },
        "decomposition_notes": "Efficiency advantage of B does not outweigh the comfort deficit.",
    },
    {
        "id": 2,
        "split": "train",
        "scene_type": "narrow_corridor",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "classical_vs_hybrid",
            "pair_scene_control": "same_scene_same_handover_goal",
            "pair_notes": "Corridor case comparing conservative rule-based motion against hybrid control.",
        },
        "segment_a": {
            "source_key": "classical_efficient",
            "handover_point_relative_to_human": "front_near",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.48, 0.02, 0.99],
            "scores": {"safety": 2, "comfort": 3, "social": 2, "efficiency": 5},
            "handover_outcome": "partial",
            "end_type": "handover_failure",
            "scene_notes": "Direct corridor pass with little buffer and abrupt final stop.",
        },
        "segment_b": {
            "source_key": "hybrid_switch",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.64, 0.04, 1.01],
            "scores": {"safety": 5, "comfort": 4, "social": 3, "efficiency": 4},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Hybrid guard keeps a safer corridor buffer before presenting the object.",
        },
        "labels": {
            "overall_preference": "B",
            "reason_label": "safety",
            "reaction_label": "avoidance",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "safety",
            "secondary_dimension": "comfort",
        },
        "decomposition_notes": "Hybrid safety gain dominates the classical efficiency gain in the corridor.",
    },
    {
        "id": 3,
        "split": "train",
        "scene_type": "open_space",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "learned_vs_hybrid",
            "pair_scene_control": "same_scene_different_handover_style",
            "pair_notes": "Open space comparison where learned and hybrid both succeed but differ in social quality.",
        },
        "segment_a": {
            "source_key": "learned_social",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front_right",
            "handover_point_3d": [0.66, -0.08, 1.02],
            "scores": {"safety": 4, "comfort": 4, "social": 5, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Smooth open-space approach with readable frontal cue before release.",
        },
        "segment_b": {
            "source_key": "hybrid_blend",
            "handover_point_relative_to_human": "right_near",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.57, -0.26, 1.04],
            "scores": {"safety": 4, "comfort": 3, "social": 2, "efficiency": 4},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Functional hybrid path but final presentation is less socially readable.",
        },
        "labels": {
            "overall_preference": "A",
            "reason_label": "social",
            "reaction_label": "hesitation",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "social",
            "secondary_dimension": "comfort",
        },
        "decomposition_notes": "Both succeed, but A is easier to interpret socially and reduces hesitation.",
    },
    {
        "id": 4,
        "split": "train",
        "scene_type": "table_separated",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "learned_vs_hybrid",
            "pair_scene_control": "same_scene_same_handover_goal",
            "pair_notes": "Across-table handover where directness competes with comfortable reachability.",
        },
        "segment_a": {
            "source_key": "learned_direct",
            "handover_point_relative_to_human": "front_far",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.95, 0.00, 1.12],
            "scores": {"safety": 3, "comfort": 2, "social": 3, "efficiency": 5},
            "handover_outcome": "partial",
            "end_type": "handover_failure",
            "scene_notes": "Direct reach over the table but the object stays slightly too far.",
        },
        "segment_b": {
            "source_key": "hybrid_blend",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front_left",
            "handover_point_3d": [0.71, 0.06, 1.00],
            "scores": {"safety": 4, "comfort": 5, "social": 4, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Hybrid controller slows near the table and presents at a reachable height.",
        },
        "labels": {
            "overall_preference": "B",
            "reason_label": "comfort",
            "reaction_label": "overreach_or_reposition",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "comfort",
            "secondary_dimension": "safety",
        },
        "decomposition_notes": "B is slower but avoids awkward overreach across the table.",
    },
    {
        "id": 5,
        "split": "train",
        "scene_type": "open_space",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "safe_vs_efficient",
            "pair_scene_control": "same_scene_different_handover_style",
            "pair_notes": "Open-space ablation between safety margin and directness.",
        },
        "segment_a": {
            "source_key": "classical_safe",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.67, 0.10, 1.03],
            "scores": {"safety": 5, "comfort": 4, "social": 4, "efficiency": 2},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Deliberate stop before release with extra clearance in open space.",
        },
        "segment_b": {
            "source_key": "learned_direct",
            "handover_point_relative_to_human": "front_near",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.45, 0.05, 1.07],
            "scores": {"safety": 2, "comfort": 3, "social": 2, "efficiency": 5},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Very direct path with fast approach and smaller personal-space margin.",
        },
        "labels": {
            "overall_preference": "A",
            "reason_label": "safety",
            "reaction_label": "avoidance",
            "winner_outcome_alignment": "conflicted",
            "dominant_dimension": "safety",
            "secondary_dimension": "social",
        },
        "decomposition_notes": "Both succeed, but B feels intrusive enough that the safety margin wins.",
    },
    {
        "id": 6,
        "split": "train",
        "scene_type": "doorway",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "comfortable_vs_efficient",
            "pair_scene_control": "same_scene_same_handover_goal",
            "pair_notes": "Doorway comparison between comfortable staging and efficient direct entry.",
        },
        "segment_a": {
            "source_key": "classical_social",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.63, 0.12, 1.04],
            "scores": {"safety": 4, "comfort": 5, "social": 4, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Balanced doorway approach with visible present-and-wait behavior.",
        },
        "segment_b": {
            "source_key": "hybrid_switch",
            "handover_point_relative_to_human": "high_front",
            "handover_point_relative_to_robot": "front_right",
            "handover_point_3d": [0.70, -0.04, 1.26],
            "scores": {"safety": 4, "comfort": 2, "social": 3, "efficiency": 5},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Fast hybrid handoff but presentation is slightly too high for natural reach.",
        },
        "labels": {
            "overall_preference": "A",
            "reason_label": "comfort",
            "reaction_label": "unnatural_posture",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "comfort",
            "secondary_dimension": "social",
        },
        "decomposition_notes": "The faster handoff is not worth the awkward high posture in the doorway.",
    },
    {
        "id": 7,
        "split": "train",
        "scene_type": "narrow_corridor",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "classical_vs_learned",
            "pair_scene_control": "same_scene_different_handover_style",
            "pair_notes": "Corridor case where slower motion is compared to direct learned motion.",
        },
        "segment_a": {
            "source_key": "classical_safe",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front_left",
            "handover_point_3d": [0.61, 0.11, 1.00],
            "scores": {"safety": 4, "comfort": 4, "social": 3, "efficiency": 2},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Slow corridor pacing with clear final pause before object release.",
        },
        "segment_b": {
            "source_key": "learned_direct",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.58, 0.03, 1.01],
            "scores": {"safety": 3, "comfort": 3, "social": 2, "efficiency": 5},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Short corridor handoff with little explicit confirmation pause.",
        },
        "labels": {
            "overall_preference": "B",
            "reason_label": "efficiency",
            "reaction_label": "none",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "efficiency",
            "secondary_dimension": "unclear",
        },
        "decomposition_notes": "Safety remains acceptable, so the faster corridor execution is preferred.",
    },
    {
        "id": 8,
        "split": "train",
        "scene_type": "table_separated",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "classical_vs_hybrid",
            "pair_scene_control": "same_scene_same_handover_goal",
            "pair_notes": "Across-table case where classical motion hesitates and hybrid timing is smoother.",
        },
        "segment_a": {
            "source_key": "classical_social",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.73, 0.04, 1.01],
            "scores": {"safety": 4, "comfort": 4, "social": 3, "efficiency": 2},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Safe presentation but waits too long before release over the table.",
        },
        "segment_b": {
            "source_key": "hybrid_blend",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front_left",
            "handover_point_3d": [0.69, 0.07, 1.00],
            "scores": {"safety": 4, "comfort": 4, "social": 4, "efficiency": 4},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Hybrid timing keeps the pose comfortable while reducing idle wait.",
        },
        "labels": {
            "overall_preference": "B",
            "reason_label": "efficiency",
            "reaction_label": "hesitation",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "efficiency",
            "secondary_dimension": "social",
        },
        "decomposition_notes": "B preserves comfort while reducing extra delay that causes hesitation.",
    },
    {
        "id": 9,
        "split": "train",
        "scene_type": "doorway",
        "pair_metadata": {
            "pair_generation_protocol": "hybrid_ablation",
            "comparison_goal": "classical_vs_hybrid",
            "pair_scene_control": "same_scene_same_handover_goal",
            "pair_notes": "Hybrid ablation around a doorway handover with different final presentation heights.",
        },
        "segment_a": {
            "source_key": "hybrid_switch",
            "handover_point_relative_to_human": "low_front",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.58, 0.09, 0.82],
            "scores": {"safety": 4, "comfort": 2, "social": 3, "efficiency": 4},
            "handover_outcome": "partial",
            "end_type": "handover_abort",
            "scene_notes": "Hybrid controller arrives quickly but presents too low near the doorway.",
        },
        "segment_b": {
            "source_key": "classical_social",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.63, 0.10, 1.02],
            "scores": {"safety": 4, "comfort": 5, "social": 4, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Rule-based social pose keeps the object at a comfortable front-mid height.",
        },
        "labels": {
            "overall_preference": "B",
            "reason_label": "comfort",
            "reaction_label": "unnatural_posture",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "comfort",
            "secondary_dimension": "safety",
        },
        "decomposition_notes": "B avoids the low handoff posture that would force bending or crouching.",
    },
    {
        "id": 10,
        "split": "train",
        "scene_type": "open_space",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "social_vs_direct",
            "pair_scene_control": "same_scene_different_handover_style",
            "pair_notes": "Open-space social style compared with direct style.",
        },
        "segment_a": {
            "source_key": "learned_social",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.68, 0.02, 1.05],
            "scores": {"safety": 4, "comfort": 4, "social": 5, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Readable frontal cue and slight pause before release.",
        },
        "segment_b": {
            "source_key": "classical_efficient",
            "handover_point_relative_to_human": "front_near",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.50, -0.02, 1.04],
            "scores": {"safety": 3, "comfort": 3, "social": 2, "efficiency": 5},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Efficient handoff with minimal cueing and more abrupt arrival.",
        },
        "labels": {
            "overall_preference": "A",
            "reason_label": "social",
            "reaction_label": "interruption",
            "winner_outcome_alignment": "conflicted",
            "dominant_dimension": "social",
            "secondary_dimension": "comfort",
        },
        "decomposition_notes": "B is fast, but A better preserves interaction flow and social readability.",
    },
    {
        "id": 11,
        "split": "train",
        "scene_type": "table_separated",
        "pair_metadata": {
            "pair_generation_protocol": "same_source_variation",
            "comparison_goal": "comfortable_vs_efficient",
            "pair_scene_control": "same_scene_different_handover_style",
            "pair_notes": "Same learned source with comfort-focused and direct variants across a table.",
        },
        "segment_a": {
            "source_key": "learned_social",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front_left",
            "handover_point_3d": [0.72, 0.03, 1.00],
            "scores": {"safety": 4, "comfort": 4, "social": 4, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Learned social variant waits for reachability across the table.",
        },
        "segment_b": {
            "source_key": "learned_direct",
            "handover_point_relative_to_human": "far_front_right",
            "handover_point_relative_to_robot": "front_right",
            "handover_point_3d": [0.92, -0.10, 1.11],
            "scores": {"safety": 3, "comfort": 2, "social": 2, "efficiency": 5},
            "handover_outcome": "partial",
            "end_type": "handover_failure",
            "scene_notes": "Direct learned variant leaves the object farther to the user's right.",
        },
        "labels": {
            "overall_preference": "A",
            "reason_label": "comfort",
            "reaction_label": "overreach_or_reposition",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "comfort",
            "secondary_dimension": "social",
        },
        "decomposition_notes": "A is slower but much easier to receive across the table.",
    },
    {
        "id": 12,
        "split": "train",
        "scene_type": "narrow_corridor",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "classical_vs_hybrid",
            "pair_scene_control": "same_scene_different_handover_style",
            "pair_notes": "Corridor preference where social cueing competes with efficient hybrid motion.",
        },
        "segment_a": {
            "source_key": "classical_social",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.62, 0.05, 1.01],
            "scores": {"safety": 4, "comfort": 4, "social": 4, "efficiency": 2},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Conservative corridor handoff with explicit present-and-pause cue.",
        },
        "segment_b": {
            "source_key": "hybrid_switch",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front_right",
            "handover_point_3d": [0.60, -0.02, 1.00],
            "scores": {"safety": 4, "comfort": 4, "social": 3, "efficiency": 4},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Hybrid timing is cleaner and shorter, with slightly less expressive cueing.",
        },
        "labels": {
            "overall_preference": "B",
            "reason_label": "efficiency",
            "reaction_label": "none",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "efficiency",
            "secondary_dimension": "social",
        },
        "decomposition_notes": "Hybrid efficiency wins because both segments remain safe and comfortable.",
    },
    {
        "id": 13,
        "split": "train",
        "scene_type": "open_space",
        "pair_metadata": {
            "pair_generation_protocol": "manual_baseline_pair",
            "comparison_goal": "other",
            "pair_scene_control": "different_scene_exploratory",
            "pair_notes": "Manual baseline stress test retained for loader/debug coverage.",
        },
        "segment_a": {
            "source_key": "manual_good",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.65, 0.00, 1.03],
            "scores": {"safety": 4, "comfort": 4, "social": 4, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Hand-authored comfortable nominal open-space handoff.",
        },
        "segment_b": {
            "source_key": "manual_bad",
            "handover_point_relative_to_human": "high_front",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.67, 0.01, 1.28],
            "scores": {"safety": 3, "comfort": 2, "social": 2, "efficiency": 3},
            "handover_outcome": "partial",
            "end_type": "handover_abort",
            "scene_notes": "Manual stress case with high presentation and unclear release timing.",
        },
        "labels": {
            "overall_preference": "A",
            "reason_label": "comfort",
            "reaction_label": "unnatural_posture",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "comfort",
            "secondary_dimension": "social",
        },
        "decomposition_notes": "Manual bad case is useful as a clear low-quality synthetic anchor.",
    },
    {
        "id": 14,
        "split": "train",
        "scene_type": "doorway",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "learned_vs_hybrid",
            "pair_scene_control": "same_scene_same_handover_goal",
            "pair_notes": "Doorway case where safety and social quality both favor hybrid motion.",
        },
        "segment_a": {
            "source_key": "learned_direct",
            "handover_point_relative_to_human": "front_near",
            "handover_point_relative_to_robot": "front_left",
            "handover_point_3d": [0.47, -0.03, 1.03],
            "scores": {"safety": 2, "comfort": 3, "social": 2, "efficiency": 5},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Quick doorway entry with short personal-space margin.",
        },
        "segment_b": {
            "source_key": "hybrid_blend",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.64, 0.06, 1.04],
            "scores": {"safety": 5, "comfort": 4, "social": 4, "efficiency": 4},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Hybrid controller stabilizes before the doorway threshold and presents cleanly.",
        },
        "labels": {
            "overall_preference": "B",
            "reason_label": "safety",
            "reaction_label": "avoidance",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "safety",
            "secondary_dimension": "social",
        },
        "decomposition_notes": "Doorway intrusion risk is high enough that hybrid control clearly wins.",
    },
    {
        "id": 15,
        "split": "train",
        "scene_type": "open_space",
        "pair_metadata": {
            "pair_generation_protocol": "same_source_variation",
            "comparison_goal": "social_vs_direct",
            "pair_scene_control": "same_scene_different_handover_style",
            "pair_notes": "Within-source learned variation for social vs direct behavior.",
        },
        "segment_a": {
            "source_key": "learned_social",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front_left",
            "handover_point_3d": [0.67, 0.04, 1.03],
            "scores": {"safety": 4, "comfort": 4, "social": 5, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Learned social policy uses a readable arc and release pause.",
        },
        "segment_b": {
            "source_key": "learned_direct",
            "handover_point_relative_to_human": "front_near",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.51, 0.00, 1.03],
            "scores": {"safety": 3, "comfort": 3, "social": 2, "efficiency": 5},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Direct learned policy is faster but less communicative before release.",
        },
        "labels": {
            "overall_preference": "A",
            "reason_label": "social",
            "reaction_label": "hesitation",
            "winner_outcome_alignment": "conflicted",
            "dominant_dimension": "social",
            "secondary_dimension": "efficiency",
        },
        "decomposition_notes": "A better supports cooperative timing even though B is faster.",
    },
    {
        "id": 16,
        "split": "train",
        "scene_type": "table_separated",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "classical_vs_learned",
            "pair_scene_control": "same_scene_same_handover_goal",
            "pair_notes": "Across-table pair where directness clearly improves the outcome without major downsides.",
        },
        "segment_a": {
            "source_key": "classical_safe",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.76, 0.02, 1.01],
            "scores": {"safety": 4, "comfort": 4, "social": 3, "efficiency": 2},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Comfortable but quite slow across-table handoff with extra waiting.",
        },
        "segment_b": {
            "source_key": "learned_social",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front_left",
            "handover_point_3d": [0.71, 0.04, 1.00],
            "scores": {"safety": 4, "comfort": 4, "social": 4, "efficiency": 4},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Learned social variant keeps comfort while reducing idle wait.",
        },
        "labels": {
            "overall_preference": "B",
            "reason_label": "efficiency",
            "reaction_label": "none",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "efficiency",
            "secondary_dimension": "social",
        },
        "decomposition_notes": "B is simply cleaner and faster without sacrificing comfort.",
    },
    {
        "id": 17,
        "split": "val",
        "scene_type": "doorway",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "classical_vs_learned",
            "pair_scene_control": "same_scene_same_handover_goal",
            "pair_notes": "Validation doorway case with comfort-led preference.",
        },
        "segment_a": {
            "source_key": "classical_social",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.64, 0.10, 1.03],
            "scores": {"safety": 4, "comfort": 5, "social": 4, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Comfortable doorway handoff with clear present pose.",
        },
        "segment_b": {
            "source_key": "learned_direct",
            "handover_point_relative_to_human": "far_front_left",
            "handover_point_relative_to_robot": "front_left",
            "handover_point_3d": [0.89, 0.12, 1.16],
            "scores": {"safety": 3, "comfort": 2, "social": 2, "efficiency": 5},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Fast doorway approach with farther left-side presentation.",
        },
        "labels": {
            "overall_preference": "A",
            "reason_label": "comfort",
            "reaction_label": "overreach_or_reposition",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "comfort",
            "secondary_dimension": "social",
        },
        "decomposition_notes": "A is easier to receive and better aligned with a doorway handoff.",
    },
    {
        "id": 18,
        "split": "val",
        "scene_type": "narrow_corridor",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "classical_vs_hybrid",
            "pair_scene_control": "same_scene_same_handover_goal",
            "pair_notes": "Validation corridor case with a clear safety-led preference.",
        },
        "segment_a": {
            "source_key": "classical_efficient",
            "handover_point_relative_to_human": "front_near",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.46, 0.00, 1.00],
            "scores": {"safety": 2, "comfort": 3, "social": 2, "efficiency": 5},
            "handover_outcome": "partial",
            "end_type": "handover_failure",
            "scene_notes": "Fast corridor approach with minimal buffer and abrupt stop.",
        },
        "segment_b": {
            "source_key": "hybrid_switch",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.63, 0.03, 1.01],
            "scores": {"safety": 5, "comfort": 4, "social": 3, "efficiency": 4},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Hybrid guard moderates the corridor approach and reduces personal-space pressure.",
        },
        "labels": {
            "overall_preference": "B",
            "reason_label": "safety",
            "reaction_label": "avoidance",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "safety",
            "secondary_dimension": "comfort",
        },
        "decomposition_notes": "Safety remains the dominant criterion in narrow-corridor exchanges.",
    },
    {
        "id": 19,
        "split": "val",
        "scene_type": "open_space",
        "pair_metadata": {
            "pair_generation_protocol": "same_source_variation",
            "comparison_goal": "social_vs_direct",
            "pair_scene_control": "same_scene_different_handover_style",
            "pair_notes": "Validation learned-style ablation focused on social readability.",
        },
        "segment_a": {
            "source_key": "learned_social",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front_right",
            "handover_point_3d": [0.69, -0.05, 1.04],
            "scores": {"safety": 4, "comfort": 4, "social": 5, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Learned social variant gives a clearer cue before release.",
        },
        "segment_b": {
            "source_key": "learned_direct",
            "handover_point_relative_to_human": "front_near",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.50, 0.00, 1.03],
            "scores": {"safety": 3, "comfort": 3, "social": 2, "efficiency": 5},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Direct learned variant is faster but less communicative.",
        },
        "labels": {
            "overall_preference": "A",
            "reason_label": "social",
            "reaction_label": "hesitation",
            "winner_outcome_alignment": "conflicted",
            "dominant_dimension": "social",
            "secondary_dimension": "efficiency",
        },
        "decomposition_notes": "Validation keeps a social-led case where the faster option is not preferred.",
    },
    {
        "id": 20,
        "split": "val",
        "scene_type": "table_separated",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "learned_vs_hybrid",
            "pair_scene_control": "same_scene_same_handover_goal",
            "pair_notes": "Validation table-separated case emphasizing comfort and posture.",
        },
        "segment_a": {
            "source_key": "learned_direct",
            "handover_point_relative_to_human": "high_front",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.83, 0.01, 1.24],
            "scores": {"safety": 3, "comfort": 2, "social": 2, "efficiency": 5},
            "handover_outcome": "partial",
            "end_type": "handover_abort",
            "scene_notes": "Direct over-table handoff is fast but slightly too high and far.",
        },
        "segment_b": {
            "source_key": "hybrid_blend",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front_left",
            "handover_point_3d": [0.71, 0.04, 1.00],
            "scores": {"safety": 4, "comfort": 5, "social": 4, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Hybrid blend keeps a reachable height and stable timing across the table.",
        },
        "labels": {
            "overall_preference": "B",
            "reason_label": "comfort",
            "reaction_label": "unnatural_posture",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "comfort",
            "secondary_dimension": "safety",
        },
        "decomposition_notes": "Comfort and posture dominate in the across-table setting.",
    },
    {
        "id": 21,
        "split": "val",
        "scene_type": "open_space",
        "pair_metadata": {
            "pair_generation_protocol": "manual_baseline_pair",
            "comparison_goal": "safe_vs_efficient",
            "pair_scene_control": "different_scene_exploratory",
            "pair_notes": "Validation manual pair kept for edge-case coverage.",
        },
        "segment_a": {
            "source_key": "manual_good",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.65, 0.00, 1.02],
            "scores": {"safety": 4, "comfort": 4, "social": 4, "efficiency": 3},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Manual nominal case with balanced presentation quality.",
        },
        "segment_b": {
            "source_key": "manual_bad",
            "handover_point_relative_to_human": "front_near",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.44, 0.00, 1.00],
            "scores": {"safety": 2, "comfort": 3, "social": 2, "efficiency": 4},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Manual stress case is quicker but too intrusive near the person.",
        },
        "labels": {
            "overall_preference": "A",
            "reason_label": "safety",
            "reaction_label": "avoidance",
            "winner_outcome_alignment": "conflicted",
            "dominant_dimension": "safety",
            "secondary_dimension": "social",
        },
        "decomposition_notes": "Manual bad case remains useful as a safety-negative comparison.",
    },
    {
        "id": 22,
        "split": "val",
        "scene_type": "table_separated",
        "pair_metadata": {
            "pair_generation_protocol": "cross_source_comparison",
            "comparison_goal": "classical_vs_learned",
            "pair_scene_control": "same_scene_same_handover_goal",
            "pair_notes": "Validation pair where efficiency wins without harming interaction quality.",
        },
        "segment_a": {
            "source_key": "classical_safe",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front",
            "handover_point_3d": [0.77, 0.03, 1.02],
            "scores": {"safety": 4, "comfort": 4, "social": 3, "efficiency": 2},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Slow but acceptable across-table handoff.",
        },
        "segment_b": {
            "source_key": "learned_social",
            "handover_point_relative_to_human": "front_mid",
            "handover_point_relative_to_robot": "front_left",
            "handover_point_3d": [0.71, 0.05, 1.01],
            "scores": {"safety": 4, "comfort": 4, "social": 4, "efficiency": 4},
            "handover_outcome": "success",
            "end_type": "handover_success",
            "scene_notes": "Quicker across-table handoff while preserving the reachable pose.",
        },
        "labels": {
            "overall_preference": "B",
            "reason_label": "efficiency",
            "reaction_label": "none",
            "winner_outcome_alignment": "aligned",
            "dominant_dimension": "efficiency",
            "secondary_dimension": "social",
        },
        "decomposition_notes": "Validation includes one clean efficiency-led preference case.",
    },
]


def compute_better_segment(score_a, score_b):
    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "B"
    return "tie"


def build_sequences(scores, handover_point, winner_side, segment_side):
    is_winner = winner_side == segment_side
    comfort = scores["comfort"]
    safety = scores["safety"]
    social = scores["social"]
    efficiency = scores["efficiency"]
    length = 6
    time_seq = [round(i * 0.2, 1) for i in range(length)]
    final_x, final_y, final_z = handover_point
    base_speed = 0.10 + 0.04 * efficiency
    ee_speed = 0.08 + 0.03 * efficiency
    yaw_bias = 0.02 * (social - 3)
    clearance = 1.10 - 0.12 * comfort + 0.05 * max(0, 4 - safety)
    if not is_winner:
        clearance += 0.06

    base_pose_seq = []
    ee_pose_seq = []
    velocity_seq = []
    robot_state_seq = []
    robot_action_seq = []
    gripper_state_seq = []
    object_pose_seq = []
    human_pose_seq = []
    human_hand_pose_seq = []
    relative_position_seq = []
    relative_distance_seq = []
    relative_orientation_seq = []

    start_x = -0.55 - 0.03 * (5 - efficiency)
    start_y = 0.18 if final_y >= 0 else -0.18
    start_z = max(0.82, final_z - 0.05)

    for idx, t in enumerate(time_seq):
        progress = idx / (length - 1)
        base_x = round(start_x + (final_x - start_x - 0.12) * progress, 3)
        base_y = round(start_y + (final_y * 0.5 - start_y) * progress, 3)
        yaw = round(yaw_bias * progress, 3)
        ee_x = round(start_x + (final_x - start_x) * progress, 3)
        ee_y = round(start_y * 0.6 + (final_y - start_y * 0.6) * progress, 3)
        ee_z = round(start_z + (final_z - start_z) * progress, 3)
        lin_speed = round(base_speed * (1.0 - 0.12 * idx), 3)
        ang_speed = round(abs(yaw_bias) * 0.8, 3)
        eff_speed = round(ee_speed * (0.9 + 0.05 * idx), 3)
        grip = 1.0 if idx < length - 1 else 0.0

        human_head_x = 1.02
        human_head_y = 0.0
        torso_x = 0.92
        torso_y = 0.0
        hand_x = round(final_x + (0.04 if is_winner else 0.09), 3)
        hand_y = round(final_y * 0.8, 3)
        hand_z = round(final_z + (-0.01 if comfort >= 4 else 0.05), 3)

        dx = round(hand_x - ee_x, 3)
        dy = round(hand_y - ee_y, 3)
        dz = round(hand_z - ee_z, 3)
        dist = round((dx ** 2 + dy ** 2 + dz ** 2) ** 0.5 + (0.02 * (3 - min(safety, 3))), 3)
        orientation = round((0.08 - 0.015 * social) + (0.02 if not is_winner else 0.0), 3)

        base_pose_seq.append([base_x, base_y, yaw])
        ee_pose_seq.append([ee_x, ee_y, ee_z])
        velocity_seq.append([lin_speed, ang_speed, eff_speed])
        robot_state_seq.append([base_x, base_y, yaw, ee_x, ee_y, ee_z, grip])
        robot_action_seq.append([lin_speed, yaw_bias, final_x - ee_x, grip - 0.5])
        gripper_state_seq.append([grip])
        object_pose_seq.append([ee_x, ee_y, ee_z])
        human_pose_seq.append([human_head_x, human_head_y, torso_x, torso_y, hand_x, hand_y])
        human_hand_pose_seq.append([hand_x, hand_y, hand_z])
        relative_position_seq.append([dx, dy, dz])
        relative_distance_seq.append(dist)
        relative_orientation_seq.append(orientation)

    return {
        "time_seq": time_seq,
        "robot_state_seq": robot_state_seq,
        "robot_action_seq": robot_action_seq,
        "base_pose_seq": base_pose_seq,
        "ee_pose_seq": ee_pose_seq,
        "velocity_seq": velocity_seq,
        "gripper_state_seq": gripper_state_seq,
        "object_pose_seq": object_pose_seq,
        "human_pose_seq": human_pose_seq,
        "human_hand_pose_seq": human_hand_pose_seq,
        "relative_position_seq": relative_position_seq,
        "relative_distance_seq": relative_distance_seq,
        "relative_orientation_seq": relative_orientation_seq,
    }


def build_segment(sample_id, scene_type, side_key, segment_spec, labels):
    source = SOURCE_CONFIGS[segment_spec["source_key"]]
    scene = SCENE_CONFIGS[scene_type]
    winner_side = labels["overall_preference"]
    side = side_key[-1].upper()
    return {
        "segment_id": f"seg_{sample_id}_{side.lower()}",
        "scene_type": scene_type,
        "segment_start_type": "human_notice",
        "segment_end_type": segment_spec["end_type"],
        "handover_outcome": segment_spec["handover_outcome"],
        **deepcopy(source),
        "context": {
            "map_id": scene["map_id"],
            "handover_point_3d": segment_spec["handover_point_3d"],
            "handover_point_relative_to_human": segment_spec["handover_point_relative_to_human"],
            "handover_point_relative_to_robot": segment_spec["handover_point_relative_to_robot"],
            "handover_point_map_zone": scene["handover_point_map_zone"],
            "human_initial_relative_position": scene["human_initial_relative_position"],
            "robot_initial_relative_position": scene["robot_initial_relative_position"],
            "scene_notes": segment_spec["scene_notes"],
        },
        "sequences": build_sequences(
            segment_spec["scores"],
            segment_spec["handover_point_3d"],
            winner_side,
            side,
        ),
    }


def build_reward_decomposition(spec):
    dims = {}
    for name in DIMENSIONS:
        score_a = spec["segment_a"]["scores"][name]
        score_b = spec["segment_b"]["scores"][name]
        dims[name] = {
            "score_a": score_a,
            "score_b": score_b,
            "better_segment": compute_better_segment(score_a, score_b),
            "severity_if_bad": max(0, 5 - min(score_a, score_b) - 1),
            "confidence": round(0.7 + 0.05 * abs(score_a - score_b), 2),
            "notes": f"Synthetic {name} comparison: A={score_a}, B={score_b}.",
        }
    return {
        "dimensions": dims,
        "priority_order": ["safety", "comfort", "social", "efficiency"],
        "aggregation_rule": "priority_weighted_with_safety_veto",
        "preference_explanation_basis": "dimension_scores_plus_reason_label",
        "decomposition_notes": spec["decomposition_notes"],
    }


def build_sample(spec):
    sample_id = f"pair_{spec['id']:04d}"
    labels = deepcopy(spec["labels"])
    sample = {
        "sample_id": sample_id,
        "pair_metadata": deepcopy(spec["pair_metadata"]),
        "segment_a": build_segment(sample_id, spec["scene_type"], "segment_a", spec["segment_a"], labels),
        "segment_b": build_segment(sample_id, spec["scene_type"], "segment_b", spec["segment_b"], labels),
        "labels": labels,
        "reward_decomposition": build_reward_decomposition(spec),
        "annotator_id": "self_v1",
        "notes": spec["decomposition_notes"],
    }
    return sample


def write_jsonl(path, samples):
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=True) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = [build_sample(spec) for spec in SPECS]
    train_samples = [sample for spec, sample in zip(SPECS, samples) if spec["split"] == "train"]
    val_samples = [sample for spec, sample in zip(SPECS, samples) if spec["split"] == "val"]

    with (OUT_DIR / "example_pair_0001.json").open("w", encoding="utf-8") as fh:
        json.dump(train_samples[0], fh, indent=2, ensure_ascii=True)
        fh.write("\n")

    write_jsonl(OUT_DIR / "train.jsonl", train_samples)
    write_jsonl(OUT_DIR / "val.jsonl", val_samples)

    print(f"Wrote example pair to {OUT_DIR / 'example_pair_0001.json'}")
    print(f"Wrote {len(train_samples)} train samples to {OUT_DIR / 'train.jsonl'}")
    print(f"Wrote {len(val_samples)} val samples to {OUT_DIR / 'val.jsonl'}")


if __name__ == "__main__":
    main()
