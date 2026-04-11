# Structured Reward V2: Comfort/Safety + Rethinking Candidate Reranker

## Scope

This document defines the new **V2 main path** for the handover structured reward model.

V2 replaces the old "flat four-dimension structured preference" interpretation with a new system:

1. **Layer 0: Candidate Generation Layer**
2. **Layer 1: Context + Candidate Input**
3. **Layer 2: Geometry Layer**
4. **Layer 3: Backbone Fusion**
5. **Layer 4: External Supervision Heads**
6. **Layer 5: Structured Reward / Candidate Reranking**

The model is no longer conceptualized as "input one A/B pair and directly learn everything." The correct system view is:

`context + candidate_set -> compare candidates -> rerank/select`

The pairwise A/B view remains only as a **training export format**.

## V2 Principles

### Candidate-First

The Rethinking-inspired part is not just metadata. V2 adds a real **Candidate Generation Layer**. A safe backbone proposes multiple feasible candidates, and the structured reward model reranks them.

Required `candidate_source` values:

- `classical`
- `learned`
- `hybrid`
- `safe_fallback`

### Two-Dimension Main Reward

V2 only keeps two main reward dimensions:

- `safety`
- `comfort`

The old flat four-dimension path is not the V2 main path.

### Hierarchical Reward

The main reward is hierarchical rather than a flat weighted sum:

```text
segment_score_structured = comfort_score - lambda_veto * ReLU(tau_safe - safety_score)^2
```

Interpretation:

- `safety` is the gate / prerequisite
- `comfort` is the main quality objective after safety is acceptable
- severe safety problems cannot be compensated by high comfort

In the current implementation, V2 keeps three structured branches:

- `geometry_structured_branch`: computed directly from geometry comfort/safety scores
- `learned_structured_branch`: computed from bounded learned comfort/safety score heads
- `final_structured_branch`: geometry-dominant fusion, with `alpha_geom > alpha_learned`

The default prototype uses:

```text
segment_score_structured_final =
    alpha_geom * segment_score_structured_geom
  + alpha_learned * segment_score_structured_learned
```

with `alpha_geom = 0.8` and `alpha_learned = 0.2`.

### Reason Label Definition

`reason_label` is defined from the **loser perspective**:

> In the current A/B pair, what is the worse candidate mainly worse at?

Allowed values:

- `safety`
- `comfort`
- `mixed`
- `unclear`

### Optional Targets

Required in Phase 1:

- `overall_preference`
- `reason_label`
- `reaction_label`
- `comfort_better_label`
- `safety_better_label`

Optional in Phase 1:

- `comfort_score_target`
- `safety_score_target`
- `safety_subreason_label`
- `comfort_subreason_label`

For score targets, the schema stores pairwise candidate scores as:

```json
"comfort_score_target": { "A": 0.82, "B": 0.46 }
```

and similarly for `safety_score_target`.

Real data may leave these fields `null`. Synthetic data may fill them.

## Layer 0: Candidate Generation Layer

### Inputs

- current human state
- current robot state
- local environment map / scene context
- task goal for approach + handover

### Outputs

One `candidate_set` per context. Each candidate must include:

- `candidate_id`
- `candidate_source`
- `candidate_generator_detail`
- `planner_family`
- `hybrid_role`
- `trajectory`
- `handover_pose`
- `handover_timing`
- `handover_time_context`
- `feasibility_flags`

### Phase 1 Synthetic Requirement

Synthetic V2 data must simulate Layer 0 explicitly. Each context includes at least:

- 1 `classical` candidate
- 1 `learned` candidate
- 1 `hybrid` candidate
- 1 `safe_fallback` candidate

Then `training_pairs` are sampled from that same candidate set, for example:

- `classical` vs `learned`
- `classical` vs `hybrid`
- `learned` vs `hybrid`
- `hybrid` vs `safe_fallback`

## Phase 1 Schema

Top-level record:

```json
{
  "context_id": "ctx_v2_open_001",
  "context": { "...": "..." },
  "candidate_set": [{ "...": "..." }],
  "training_pairs": [{ "...": "..." }],
  "annotator_id": "synthetic_generator_v2",
  "notes": "Synthetic context-level candidate set"
}
```

### `context`

Required fields:

- `scene_type`
- `human_position`
- `human_orientation`
- `human_hand_position`
- `human_posture_features`
- `robot_initial_pose`
- `robot_initial_velocity`
- `environment_map_id`
- `local_costmap_embedding`
- `obstacle_proximity_features`
- `reachable_region_features`
- `task_goal`

### `candidate_set[*]`

Required fields:

- `candidate_id`
- `candidate_source`
- `candidate_generator_detail`
- `planner_family`
- `hybrid_role`
- `source_description`
- `trajectory`
- `handover_pose`
- `handover_timing`
- `handover_time_context`
- `feasibility_flags`

Required `trajectory` fields:

- `time_seq`
- `robot_state_seq`
- `robot_action_seq`
- `robot_base_pose_seq`
- `robot_ee_pose_seq`
- `robot_velocity_seq`
- `gripper_state_seq`
- `object_pose_seq`
- `human_pose_seq`
- `human_hand_pose_seq`
- `relative_position_seq`
- `relative_distance_seq`
- `relative_orientation_seq`

### `training_pairs[*]`

Required fields:

- `pair_id`
- `candidate_a_id`
- `candidate_b_id`
- `pair_metadata`
- `labels`

Required `labels` fields:

- `overall_preference`
- `reason_label`
- `reaction_label`
- `comfort_better_label`
- `safety_better_label`

Optional `labels` fields:

- `comfort_score_target`
- `safety_score_target`
- `safety_subreason_label`
- `comfort_subreason_label`

Recommended subreason enums:

- `safety_subreason_label`: `distance_intrusion | speed_risk | timing_risk | mixed | unclear`
- `comfort_subreason_label`: `reachability | height_alignment | front_sector_alignment | posture_cost | mixed | unclear`

## Current Prototype Notes

### Bounded Learned Score Heads

The learned `comfort_score_head` and `safety_score_head` are bounded to `[0, 1]` via sigmoid outputs. This keeps the learned branch calibrated to the same scale as the geometry branch and synthetic score targets.

### Geometry-As-Anchor Structured Preference

The prototype does not treat the geometry layer as decoration. It keeps three explicit structured branches:

- `structured_preference_logits_geom`
- `structured_preference_logits_learned`
- `structured_preference_logits_final`

`structured_preference_logits_final` is the main reranking branch used for structured comparison and selection.

The intended role split is:

- geometry branch = primary comfort/safety ruler
- learned branch = calibration / residual correction
- final branch = geometry-dominant structured preference

### Fixed vs Trainable Geometry Modes

The current prototype supports two geometry modes:

- `geometry_mode = fixed`
- `geometry_mode = trainable`

`fixed` keeps the original fixed geometry preprocessing path and is mainly used as a baseline / ablation path.

`trainable` moves geometry into the model forward graph:

- loader provides raw geometry inputs
- the model computes geometry scores internally
- geometry parameters are trainable
- geometry remains the dominant structured reward anchor
- `geometry_raw_feature_dim` is currently `33`

Inside `trainable`, the current prototype now supports two parameterization styles:

- `geometry_parameterization_mode = global`
- `geometry_parameterization_mode = contextual`

`global` means the model learns one shared comfort fan and one shared safety ellipsoid for the whole dataset.

`contextual` means the model learns a global base ruler and then applies a structured context-conditioned adapter so geometry parameters can shift per sample.

The current contextual prototype conditions not only on local handover geometry but also on a compact environment summary, including:

- local costmap embedding
- obstacle proximity features
- scene type id

### Consistency And Alignment

The current prototype keeps two minimal consistency terms:

- `overall_preference_logits` aligned to `structured_preference_logits_final`
- `structured_preference_logits_learned` aligned to `structured_preference_logits_geom`

The second term is implemented in a teacher-student style:

- geometry acts as the anchor
- learned structured preference aligns to geometry
- gradients are stopped on the geometry anchor side

In addition, the learned comfort/safety score heads are explicitly aligned to geometry comfort/safety scores through a score-level alignment loss.

### Pair-Conditioned Subreason Heads

`safety_subreason` and `comfort_subreason` are predicted from a pair-conditioned loser/winner representation rather than a single-candidate representation:

- loser embedding
- winner embedding
- loser-winner diff
- absolute diff

During training, loser selection uses the ground-truth pair preference.

During prediction, loser selection follows the `final_structured_branch`.

### Context-Conditioned Geometry Parameters

The trainable geometry path is no longer limited to one global fixed ruler.

The current trainable prototype can use either:

- a global learnable base geometry parameter set only (`global`)
- or a global base geometry parameter set plus a structured context-conditioned adapter (`contextual`)

The current contextual adapter is no longer a single flat delta MLP. It now uses:

- a shared context encoder
- separate comfort / safety adapters
- explicit split between:
  - local handover geometry context
  - environment summary context

The current `geometry_raw_feature_dim = 33` is organized as:

- `23` local geometry / timing fields
- `10` environment summary fields

In practice this means the learned comfort fan and safety ellipsoid can change with:

- human state
- handover pose
- approach geometry
- speed / timing context
- obstacle clearance and local free-space summary
- scene category

The current debug output also exposes adapter summaries such as:

- shared context norm
- local context norm
- environment context norm
- adapter delta norm
- adapter gate mean

This is the current prototype approximation of "learning the ruler" rather than only learning around a manually fixed ruler.

### Tie / Unclear Subreason Behavior

When the final structured preference prediction is `tie` or `unclear`, the prototype does not emit a concrete loser-conditioned subreason explanation. In debug output this appears as:

- `subreason_active_pred = False`
- `loser_id_pred = None`
- `safety_subreason_pred = None`
- `comfort_subreason_pred = None`

## Phase 1 Loader Output

The V2 loader expands `context + candidate_set + training_pairs` into pairwise items while preserving a candidate-first schema upstream.

Main batch fields:

- `observations`, `actions`, `observations_2`, `actions_2`
- `timestep_1`, `timestep_2`
- `attn_mask`, `attn_mask_2`
- `preference_label`
- `preference_distribution`
- `reason_label`
- `reaction_label`
- `comfort_better_label`
- `safety_better_label`
- `comfort_score_targets`, `comfort_score_masks`
- `safety_score_targets`, `safety_score_masks`
- `safety_subreason_label`, `safety_subreason_mask`
- `comfort_subreason_label`, `comfort_subreason_mask`
- `context_features`
- `candidate_a_geometry_raw_features`, `candidate_b_geometry_raw_features`
- `candidate_a_aux_features`, `candidate_b_aux_features`
- `sample_id`
- `metadata`

Phase 1 does **not** change the model trunk yet.

## Current External Heads

The current V2 prototype keeps the following external heads:

- `overall_preference_head`
- `reason_head`
- `reaction_head`
- `comfort_better_head`
- `safety_better_head`
- `comfort_score_head`
- `safety_score_head`
- `safety_subreason_head`
- `comfort_subreason_head`

The first five are the main external supervision heads.

The score and subreason heads may train with missing targets masked out.

## Current Training Losses

The current V2 prototype keeps the following loss terms:

- `L_preference`
- `L_reason`
- `L_reaction`
- `L_comfort_better`
- `L_safety_better`
- `L_comfort_score` when score targets exist
- `L_safety_score` when score targets exist
- `L_safety_subreason` when subreason targets exist and the reason hierarchy activates it
- `L_comfort_subreason` when subreason targets exist and the reason hierarchy activates it
- `L_structured_preference_consistency_overall`
- `L_structured_preference_consistency_learned_to_geom`
- `L_score_alignment`
- `L_geometry_prior` in `trainable` geometry mode

The important design choice is that geometry remains the anchor and learned heads are auxiliary.

### Config Boundary

`context_feature_dim` and `geometry_raw_feature_dim` are different concepts:

- `context_feature_dim`
  full context embedding consumed by the V2 fusion path
- `geometry_raw_feature_dim`
  compact geometry-oriented raw bundle consumed by the trainable geometry layer

The current values in the synthetic V2 setup are:

- `context_feature_dim = 31`
- `geometry_raw_feature_dim = 33`

Legacy `dominant / secondary / decomp` flat-path config fields remain in the codebase only for backward compatibility and are not part of the V2 main path.

## Debug And Reproduction

### Shortest Acceptance Path

If you want the shortest end-to-end acceptance flow for the current prototype, run:

```bash
python scripts/check_structured_pref_data_v2_cs_rethinking.py \
  --data data/synthetic_v2_cs_rethinking \
  --preview-count 1

python scripts/debug_structured_pref_pipeline_v2_cs_rethinking.py \
  --data_dir data/synthetic_v2_cs_rethinking \
  --preview_count 1

python scripts/train_structured_pref_v2_cs_rethinking.py \
  --data_dir data/synthetic_v2_cs_rethinking \
  --epochs 1 \
  --batch_size 4 \
  --max_seq_len 8 \
  --device cpu
```

This verifies:

- schema validity
- candidate-set to pairwise loader behavior
- geometry / learned / final structured debug outputs
- 1 epoch CPU training smoke test

### Data Schema Check

```bash
python scripts/check_structured_pref_data_v2_cs_rethinking.py \
  --data data/synthetic_v2_cs_rethinking \
  --preview-count 1
```

Data / pipeline debug:

```bash
python scripts/debug_structured_pref_pipeline_v2_cs_rethinking.py \
  --data_dir data/synthetic_v2_cs_rethinking \
  --preview_count 1
```

1 epoch CPU smoke test:

```bash
python scripts/train_structured_pref_v2_cs_rethinking.py \
  --data_dir data/synthetic_v2_cs_rethinking \
  --epochs 1 \
  --batch_size 4 \
  --max_seq_len 8 \
  --device cpu
```

If the default Python environment does not include the training stack, use the local project environment:

```bash
./.conda-handover-render/bin/python scripts/debug_structured_pref_pipeline_v2_cs_rethinking.py \
  --data_dir data/synthetic_v2_cs_rethinking \
  --preview_count 1

./.conda-handover-render/bin/python scripts/train_structured_pref_v2_cs_rethinking.py \
  --data_dir data/synthetic_v2_cs_rethinking \
  --epochs 1 \
  --batch_size 4 \
  --max_seq_len 8 \
  --device cpu
```

Current debug output is expected to expose:

- geometry comfort/safety scores for candidate A/B
- learned comfort/safety scores for candidate A/B
- geometry / learned / final structured segment scores
- overall preference prediction
- geometry / learned / final structured preference predictions
- loser-conditioned subreason predictions
- `subreason_active_pred` for `tie/unclear` cases
- comfort geometry parameters
- safety geometry parameters

### Trainable Geometry Reproduction

Trainable geometry mode, global parameters:

```bash
python scripts/train_structured_pref_v2_cs_rethinking.py \
  --data_dir data/synthetic_v2_cs_rethinking \
  --epochs 1 \
  --batch_size 4 \
  --max_seq_len 8 \
  --device cpu \
  --disable_jit \
  --geometry_mode trainable \
  --geometry_parameterization_mode global
```

Trainable geometry mode, contextual parameters:

```bash
python scripts/train_structured_pref_v2_cs_rethinking.py \
  --data_dir data/synthetic_v2_cs_rethinking \
  --epochs 1 \
  --batch_size 4 \
  --max_seq_len 8 \
  --device cpu \
  --disable_jit \
  --geometry_mode trainable \
  --geometry_parameterization_mode contextual
```

Fixed geometry baseline:

```bash
python scripts/train_structured_pref_v2_cs_rethinking.py \
  --data_dir data/synthetic_v2_cs_rethinking \
  --epochs 1 \
  --batch_size 4 \
  --max_seq_len 8 \
  --device cpu \
  --disable_jit \
  --geometry_mode fixed
```

Trainable geometry debug from checkpoint:

```bash
python scripts/debug_structured_pref_pipeline_v2_cs_rethinking.py \
  --data_dir data/synthetic_v2_cs_rethinking \
  --max_seq_len 8 \
  --preview_count 1 \
  --checkpoint outputs/structured_pref_v2_cs_rethinking_trainable_ctx2/latest.pkl
```

### Geometry Ablation Summary

To compare:

- `fixed`
- `trainable-global`
- `trainable-contextual`

run:

```bash
python scripts/run_geometry_mode_ablation_v2_cs_rethinking.py \
  --data_dir data/synthetic_v2_cs_rethinking \
  --epochs 1 \
  --batch_size 4 \
  --max_seq_len 8 \
  --device cpu \
  --disable_jit \
  --seeds 0 1 2
```

This writes:

- `outputs/structured_pref_v2_cs_rethinking_ablation/summary.json`
- `outputs/structured_pref_v2_cs_rethinking_ablation/summary.md`

### Missing-Target Readiness Check

The current V2 prototype includes a dedicated readiness check for real-data style missing optional targets:

```bash
python scripts/check_structured_pref_missing_target_readiness_v2_cs_rethinking.py \
  --source_dir data/synthetic_v2_cs_rethinking \
  --output_root data/synthetic_v2_cs_rethinking_missing_optional_checks \
  --run_train_smoke
```

This script rewrites synthetic data into three scenarios and verifies that schema checks, loader behavior, and training still work:

- `scores_missing`
- `subreasons_missing`
- `all_optional_missing`

This is the main readiness check for the "real data phase 1 may omit precise scores and subreason labels" requirement.

## Current Prototype Boundary

This V2 path is now a working reranker / selector prototype, not a policy-learning stack.

What it already is:

- candidate-first
- geometry-guided
- geometry-trainable
- context-conditioned geometry prototype
- safety-to-comfort hierarchical
- pairwise-trainable
- debuggable and smoke-testable

What it is not yet:

- a real planner-integrated production stack
- a full real-data training pipeline
- a final ablation-tuned benchmarked release

## Current Status Summary

At the current stage, the prototype already satisfies the main V2 restructuring goals:

- candidate generation layer is explicit in the data model
- geometry modules are explicit and connected
- structured reward is safety-to-comfort hierarchical
- geometry is the dominant structured reward anchor
- learned scores are bounded and treated as calibration / residual signals
- subreason prediction is pair-conditioned and loser-conditioned
- `tie/unclear` cases do not force a fake loser-conditioned explanation
- optional score/subreason targets may be missing without breaking training

The main remaining work is no longer architectural refactoring. It is downstream hardening:

- real annotated data integration
- hyperparameter tuning and ablations
- stronger planner/runtime integration

The most important experimental caveat is that the current small synthetic ablation still shows `fixed` as the strongest baseline. `trainable-contextual` is more aligned with the intended research direction than `trainable-global`, but it has not yet stably surpassed `fixed`. So the current status should be described as a high-completion research prototype rather than a fully closed experimental conclusion.

## Phase Plan

### Phase 1

- new V2 document
- new V2 schema
- synthetic candidate-set generator
- generated `train/val` files
- V2 loader
- V2 data check + debug scripts

### Phase 2

- `ComfortFanModule`
- `SafetyEllipsoidModule`
- geometry debug outputs

### Phase 3

- backbone fusion
- new external heads

### Phase 4

- structured reward branch
- consistency loss
- V2 train script
- 1 epoch CPU smoke test

## Not In The V2 Main Path

The following are not part of the V2 main training path:

- `secondary_dimension`
- old flat four-way `dominant_dimension`
- old flat `decomp_better`
- old flat `decomp_score`
- old four-dimension consistency stack
- old `priority_weighted_with_safety_veto` main-path aggregation

Old files may remain in the repo, but V2 should not depend on them as the main route.
