# Weakly-Real Annotation Template: V2 Comfort/Safety Candidate-First

Weakly-real records should keep the same V2 top-level schema:

- `context`
- `candidate_set`
- `training_pairs`

Required labels:

- `overall_preference`
- `reason_label`
- `reaction_label`
- `comfort_better_label`
- `safety_better_label`

Optional labels that may be missing:

- `comfort_score_target`
- `safety_score_target`
- `comfort_subreason_label`
- `safety_subreason_label`

Template file:

- `data/weakly_real_v2_template/example_weakly_real_context.json`

Validation:

```bash
./.conda-handover-render/bin/python scripts/check_weakly_real_v2_cs_rethinking.py \
  --data data/weakly_real_v2_template/example_weakly_real_context.json \
  --max_seq_len 8
```
