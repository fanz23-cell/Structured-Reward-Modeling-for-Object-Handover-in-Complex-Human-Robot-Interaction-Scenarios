# Real Data Annotation Template: V2 Comfort/Safety Structured Reward

This note defines the minimum real-data annotation shape for the current V2 path.

## Goal

The real-data phase 1 path should be compatible with the current V2 loader without requiring:

- precise comfort score labels
- precise safety score labels
- safety subreason labels
- comfort subreason labels

Those four fields may be missing.

## Minimum Required Top-Level Structure

Each record should still follow the V2 candidate-first schema:

- `context_id`
- `context`
- `candidate_set`
- `training_pairs`
- `annotator_id`
- `notes`

## Minimum Required Labels

Each `training_pairs[*].labels` entry must include:

- `overall_preference`
- `reason_label`
- `reaction_label`
- `comfort_better_label`
- `safety_better_label`

The following may be `null`:

- `comfort_score_target`
- `safety_score_target`
- `safety_subreason_label`
- `comfort_subreason_label`

## Template File

Use this minimal example as the starting point:

- `data/real_pref_v2_template/example_real_context.json`

It is a single-record JSON template that can be checked directly by the real-data validation script.

## Validation

Run:

```bash
python scripts/check_real_pref_v2_cs_rethinking.py \
  --data data/real_pref_v2_template/example_real_context.json \
  --max_seq_len 8
```

If the default Python environment does not include the V2 stack, use:

```bash
./.conda-handover-render/bin/python scripts/check_real_pref_v2_cs_rethinking.py \
  --data data/real_pref_v2_template/example_real_context.json \
  --max_seq_len 8
```

## Notes

- The example file keeps the same candidate-first structure as synthetic V2.
- It uses the minimum required supervision for real-data phase 1.
- It is fully compatible with the current V2 loader after wrapping as JSONL.
