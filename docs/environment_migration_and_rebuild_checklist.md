# Environment Migration And Rebuild Checklist

This note is the "delete local environments now, rebuild later on the desktop" guide for this repository.

It records:

- what currently occupies disk space in the repo
- which local environments exist right now
- which project components depend on which environment family
- how to rebuild those environments later
- which exact package snapshots were exported before cleanup

## Short Answer

Yes. You can delete the local environments and keep the codebase fully usable as long as you preserve:

- this repository
- the dependency entry points already tracked in git
- the exported package snapshots under `docs/env_snapshots/`

That is the safer path anyway, because the current local environments are not clean minimal environments. They contain extra packages unrelated to this repository, especially ROS-related packages in both conda environments.

## Disk Usage Snapshot

Snapshot taken from the repo root on 2026-04-29:

- repo total: `15G`
- `.conda-handover-render`: `8.5G`
- `.conda-handover-render36`: `2.8G`
- `.venv-handover-render`: `13M`
- `outputs/`: `577M`
- `PrefMMT/logs/`: `521M`
- `iGibson/`: `1.8G`
- `cogail_downloads/`: `145M`
- `bridge/`: `165M`

If you delete only the three local environments, you should recover about `11.3G`.

## Current Local Environments

### 1. Main structured-reward environment

- path: `./.conda-handover-render`
- Python: `3.10.20`
- intended use:
  - structured reward model scripts in `scripts/`
  - docs examples that call `./.conda-handover-render/bin/python`
  - bridge utilities built around the current structured reward work

Observed key packages from the current snapshot:

- `jax==0.6.2`
- `jaxlib==0.6.2`
- `flax==0.10.7`
- `optax==0.2.7`
- `torch==2.10.0+cu128`
- `transformers==5.3.0`
- `gym==0.26.2`
- `pybullet==3.2.7`

Important note:

- this environment is not a clean project-only env
- it includes many ROS 2 and robot-stack packages
- `stable_baselines3` was listed in `PrefMMT/requirements.txt` but was not importable in this snapshot
- `d4rl` was also not importable in this snapshot

So this environment should be treated as a useful snapshot of "what existed locally", not as the best future target state.

### 2. Legacy Co-GAIL / iGibson environment

- path: `./.conda-handover-render36`
- Python: `3.6.15`
- intended use:
  - `cogail/`
  - `iGibson/`
  - older handover simulation and demonstration generation path

Observed key packages from the current snapshot:

- `igibson==1.0.3`
- editable install from `git+https://github.com/j96w/iGibson.git@78895f0d9a255e0df51a70a63e07e6b44bd097c1`
- `torch==1.10.2+cu102`
- `gym==0.21.0`
- `pybullet==3.2.7`

Important note:

- this environment also contains many extra ROS 2 packages
- `pygame` and `h5py` are listed by local project requirements but were not importable in this snapshot
- that suggests the environment drifted from a clean install over time

### 3. Small local venv

- path: `./.venv-handover-render`
- Python: `3.13.11`
- size: `13M`

This looks like a lightweight auxiliary venv rather than the main training environment.

## What To Keep In Git Before Cleanup

The repository already preserves the install entry points:

- `PrefMMT/requirements.txt`
- `PrefMMT/requirements_core.txt`
- `PrefMMT/requirements_nosb3.txt`
- `PrefMMT/d4rl/setup.py`
- `cogail/requirements.txt`
- `cogail/README.md`
- `APReL/requirements.txt`
- `APReL/setup.py`
- `iGibson/setup.py`

This commit also exports exact package snapshots:

- `docs/env_snapshots/conda-handover-render-pip-freeze.txt`
- `docs/env_snapshots/conda-handover-render36-pip-freeze.txt`
- `docs/env_snapshots/venv-handover-render-pip-freeze.txt`

Use the requirements files as the rebuild recipe.
Use the freeze files as the forensic record of what was actually installed on this machine.

## Recommended Rebuild Strategy On The Desktop

Do not try to preserve and copy these entire environments byte-for-byte.
Instead, rebuild cleanly on the desktop from versioned instructions.

### Environment A: structured reward / PrefMMT / bridge

Recommended target:

- Python `3.10`
- one fresh conda env dedicated to:
  - `scripts/`
  - `PrefMMT/`
  - `APReL/`
  - `bridge/`

Suggested rebuild order:

1. Create env:

```bash
conda create -n reward-model-py310 python=3.10 -y
conda activate reward-model-py310
```

2. Upgrade pip tooling:

```bash
python -m pip install --upgrade pip setuptools wheel
```

3. Install the lighter dependency base first:

```bash
python -m pip install -r PrefMMT/requirements_core.txt
```

4. Install project extras usually needed for this repo:

```bash
python -m pip install pandas wandb transformers
python -m pip install -r APReL/requirements.txt
python -m pip install -e APReL
```

5. Install local D4RL copy if you need PrefMMT offline RL data tooling:

```bash
python -m pip install -e PrefMMT/d4rl
```

6. Install the local PrefMMT package if needed by your scripts:

```bash
python -m pip install -e PrefMMT
```

7. Install JAX and jaxlib that match the desktop CUDA stack.

Important:

- do not blindly reuse the current `jax==0.6.2` and `jaxlib==0.6.2` unless they match the desktop GPU/CUDA driver stack
- use the desktop machine's CUDA support as the source of truth

8. Run a smoke check:

```bash
python scripts/check_structured_pref_data_v2_cs_rethinking.py
python scripts/debug_structured_pref_pipeline_v2_cs_rethinking.py --help
```

### Environment B: Co-GAIL / iGibson legacy simulation

Recommended target:

- Python `3.6` only if you truly need the legacy Co-GAIL path
- keep this separate from the Python 3.10 env

Suggested rebuild order:

1. Create env:

```bash
conda create -n cogail-py36 python=3.6 -y
conda activate cogail-py36
```

2. Upgrade pip tooling as far as Python 3.6-compatible wheels allow:

```bash
python -m pip install --upgrade pip setuptools wheel
```

3. Install local iGibson variant:

```bash
python -m pip install -e iGibson
```

4. Install Co-GAIL requirements:

```bash
python -m pip install -r cogail/requirements.txt
```

5. If needed, also install the local package in editable mode:

```bash
cd cogail
python -m pip install -e .
cd ..
```

6. Restore the required iGibson assets and Co-GAIL datasets separately.

Important:

- the repo does not contain the full iGibson external asset download
- the repo does not contain the full Co-GAIL external demo dataset
- those must be reacquired on the desktop

## RL And VLA Migration Advice

For your desktop robot-training setup, the safe split is:

- use the Python 3.10 env for the structured reward model, preference modeling, bridge/export code, and most new experimentation
- keep the Python 3.6 Co-GAIL/iGibson env only if you still need that exact legacy simulator path
- do not merge the two environments into one unless you are intentionally doing a larger modernization pass

Why:

- `cogail/` and the local iGibson fork clearly come from an older dependency era
- `PrefMMT/` and the newer structured reward code are more naturally aligned with a modern Python 3.10 stack
- a single mixed env would be harder to debug and less reproducible

## Delete Commands

### Delete only local environments, keep code and docs

Run from the repo root:

```bash
rm -rf .conda-handover-render .conda-handover-render36 .venv-handover-render
```

### Optional additional cleanup

If you also want to remove generated logs and checkpoints later:

```bash
rm -rf outputs PrefMMT/logs
```

That would free about another `1.1G` based on the snapshot above.

## Exact Snapshot Files

These were exported before cleanup:

- [conda-handover-render-pip-freeze.txt](env_snapshots/conda-handover-render-pip-freeze.txt)
- [conda-handover-render36-pip-freeze.txt](env_snapshots/conda-handover-render36-pip-freeze.txt)
- [venv-handover-render-pip-freeze.txt](env_snapshots/venv-handover-render-pip-freeze.txt)

They are intentionally verbose.
They are the best record of the local machine state at cleanup time.

## Final Recommendation

The best practice for your next machine is:

1. keep this repo and these docs in git
2. delete the local env directories here
3. move or clone the repo onto the desktop
4. rebuild a clean Python 3.10 env for structured reward work
5. rebuild the Python 3.6 Co-GAIL/iGibson env only if you still need that legacy path

That gives you smaller storage usage now and a more reproducible setup later.
