# GPT Handoff: Structured Reward V2 Current Status

This note is a direct handoff summary for another GPT/Codex conversation.

## What The Project Is Now

The project is no longer a flat pairwise structured preference baseline.

It is now a V2 candidate-first structured reward / reranker prototype with:

- `context + candidate_set + training_pairs` schema
- candidate-first synthetic data generation
- explicit comfort/safety geometry layer
- safety-to-comfort hierarchical structured reward
- geometry / learned / final structured branches
- pair-conditioned, loser-conditioned subreason prediction
- optional score / subreason targets allowed to be missing

The model is positioned as a:

- candidate reranker
- selector
- local refinement guide

not as an end-to-end policy learner.

## Core Architecture Status

### 1. Candidate-First Data Path

Implemented.

Top-level V2 sample structure:

- `context_id`
- `context`
- `candidate_set`
- `training_pairs`
- `annotator_id`
- `notes`

Synthetic generator now creates one context with multiple candidate sources:

- `classical`
- `learned`
- `hybrid`
- `safe_fallback`

and then derives pairwise comparisons from the candidate set.

### 2. Main Reward Dimensions

Implemented.

The V2 main path only uses:

- `safety`
- `comfort`

Old flat four-way main-path logic is not the V2 primary route.

### 3. Reason Label Semantics

Implemented.

`reason_label` is defined from the loser perspective:

- `safety`
- `comfort`
- `mixed`
- `unclear`

Meaning:

the worse candidate mainly fails on what dimension.

### 4. Geometry Layer

Implemented and upgraded.

There are now two geometry modules:

- `ComfortFanModule` / trainable comfort fan path
- `SafetyEllipsoidModule` / trainable safety ellipsoid path

The system has gone through three geometry stages:

1. fixed geometry preprocessing
2. trainable global geometry
3. trainable contextual geometry

Current modes available:

- `geometry_mode = fixed`
- `geometry_mode = trainable`

Inside `trainable`, parameterization modes available:

- `geometry_parameterization_mode = global`
- `geometry_parameterization_mode = contextual`

Interpretation:

- `fixed`: hand-crafted ruler baseline
- `global`: one learned shared ruler
- `contextual`: learned base ruler plus per-sample context-conditioned adjustment

### 5. Contextual Geometry Status

Implemented at prototype level.

The contextual geometry path now conditions not only on local handover geometry but also on compact environment summary features.

Current contextual geometry raw bundle includes:

- human position / orientation / hand position
- reachable band summary
- handover pose
- final base pose
- final ee pose
- handover timing
- handover index
- max approach speed
- timing-window speed
- local costmap embedding
- obstacle proximity features
- obstacle clearance asymmetry / width summary
- scene type id

Current `geometry_raw_feature_dim` is `33`:

- `23` local geometry / timing values
- `10` environment summary values

This means the geometry ruler is no longer only a fixed or purely local ruler.

It now has a stronger prototype form of:

- scene-aware
- obstacle-aware
- context-conditioned geometry parameter adaptation

The current contextual adapter is no longer just a light delta MLP.

It now uses:

- a shared context encoder
- explicit separation between local handover geometry context and environment summary context
- separate comfort / safety adapter heads with gated delta modulation

## Structured Reward Status

Implemented.

There are three structured branches:

### geometry branch

Uses geometry comfort/safety scores:

- `comfort_score_geom`
- `safety_score_geom`

and computes:

`segment_score_structured_geom = comfort_score_geom - lambda_veto * ReLU(tau_safe - safety_score_geom)^2`

### learned branch

Uses bounded learned score heads:

- `comfort_score_preds`
- `safety_score_preds`

### final branch

Geometry-dominant combination:

- `alpha_geom = 0.8`
- `alpha_learned = 0.2`

So final structured preference is geometry-led, not learned-score-led.

## Learned Score Status

Implemented and bounded.

`comfort_score_pred` and `safety_score_pred` are sigmoid-bounded to `[0,1]`.

There is also:

- score alignment to geometry
- learned structured branch alignment to geometry branch

Geometry remains the anchor side with stop-gradient behavior in teacher-student alignment.

## Subreason Status

Implemented with the intended semantics.

Subreason is no longer a single-candidate attribute head.

It is pair-conditioned and loser-conditioned.

Current subreason input uses:

- loser embedding
- winner embedding
- loser-winner diff
- absolute diff

There are two internal explanation heads:

- `safety_subreason_head`
- `comfort_subreason_head`

Reason-conditioned masking is also implemented:

- `reason = safety` or `mixed` -> safety subreason active
- `reason = comfort` or `mixed` -> comfort subreason active
- `reason = unclear` -> subreason masked

Tie / unclear prediction behavior is also fixed:

If final structured preference is `tie` or `unclear`, then:

- `subreason_active_pred = False`
- `loser_id_pred = None`
- `safety_subreason_pred = None`
- `comfort_subreason_pred = None`

## Optional Target Handling Status

Implemented and validated.

Required labels remain:

- `overall_preference`
- `reason_label`
- `reaction_label`
- `comfort_better_label`
- `safety_better_label`

Optional labels may be missing:

- `comfort_score_target`
- `safety_score_target`
- `safety_subreason_label`
- `comfort_subreason_label`

Dedicated readiness checks were run for:

- missing score targets
- missing subreason targets
- all optional targets missing

and training still runs.

## Main Scripts And Files

### docs

- `docs/structured_reward_v2_cs_rethinking.md`
- `docs/gpt_handoff_structured_reward_v2_status.md`

### data and loader

- `scripts/generate_structured_pref_synth_v2_cs_rethinking.py`
- `scripts/check_structured_pref_data_v2_cs_rethinking.py`
- `PrefMMT/JaxPref/data/structured_pref_dataset_v2_cs_rethinking.py`

### geometry

- `PrefMMT/JaxPref/geometry_modules.py`
- `PrefMMT/JaxPref/trainable_geometry_modules.py`

### model

- `PrefMMT/JaxPref/PrefMMT.py`
- `PrefMMT/JaxPref/structured_reward_v2_heads.py`

### train / debug / checks

- `scripts/train_structured_pref_v2_cs_rethinking.py`
- `scripts/debug_structured_pref_pipeline_v2_cs_rethinking.py`
- `scripts/check_structured_pref_missing_target_readiness_v2_cs_rethinking.py`
- `scripts/run_geometry_mode_ablation_v2_cs_rethinking.py`

## Current Experimental Status

Multi-seed ablation summary exists at:

- `outputs/structured_pref_v2_cs_rethinking_ablation/summary.json`
- `outputs/structured_pref_v2_cs_rethinking_ablation/summary.md`

Current 3-run summary on the small synthetic setup:

- `fixed` remains the strongest baseline
- `contextual` is better aligned with the intended research direction than `global`
- but `contextual` has not yet empirically surpassed `fixed`

This means:

- architecture goals are mostly met
- research hypothesis is not fully validated yet
- the right summary is "high-completion research prototype", not "experimentally closed final system"

## What Is Fully Achieved

The following are essentially done at prototype level:

- candidate-first schema
- reranker framing
- comfort/safety-only main reward path
- geometry-dominant structured reward
- learned calibration branch
- pair-conditioned loser-conditioned subreason
- optional target masking
- fixed/global/contextual geometry modes
- train/debug/checkpoint pipeline
- missing-target readiness checks
- ablation summary pipeline

## What Is Not Fully Finished Yet

The remaining gap is no longer a major architecture gap.

It is mainly:

1. stronger experimental validation
2. stronger contextual geometry parameter generator
3. real data integration
4. longer training / multi-seed / richer ablations

More specifically:

- `contextual` currently uses a compact context-conditioned adapter
- it is stronger than the earlier version, but still not a fully rich scene-conditioned parameter generator
- no result yet proves contextual beats fixed on the current small synthetic setting
- so any high-level summary should explicitly keep the experimental caveat: implementation is largely in place, but the strongest claim that is currently supported is prototype readiness rather than final empirical superiority

## Best Current Overall Assessment

If the question is:

"Has the intended V2 architecture basically been implemented?"

Answer:

Yes. It is very close to complete.

If the question is:

"Has the full research hypothesis already been validated experimentally?"

Answer:

Not yet.

Best concise status:

- implementation completeness: about `98%`
- architecture completeness: very high
- experimental proof completeness: still incomplete

Current remaining work is primarily research validation, not a large architecture gap.

## Recommended Next Step

If continuing from here, the best next step is not another large refactor.

The best next step is:

- strengthen the contextual geometry parameter generator further
- rerun fixed/global/contextual ablations
- evaluate whether contextual can beat fixed

Secondary next step:

- connect real annotated data and rerun the same pipeline
