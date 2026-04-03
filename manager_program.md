# Aether Laser-SEP Experiment Manager Program

## Scope

The manager may mutate campaign specifications, search bounds, and run budgets.
It may not edit core physics, surrogate propagation, validation logic, or beam/SEP dynamics source files.

Protected zones:

- `src/aether_traj/dynamics.py`
- `src/aether_traj/gravity.py`
- `src/aether_traj/ephemeris.py`
- `src/aether_traj/sep_jax_workflow.py`

## Research Priorities

1. Prefer truth-validated direct-capture behaviour over surrogate-only objective gains.
2. Reward safe perilune margin, stable capture duration, final-mass margin, and operationally credible beam dwell.
3. Penalize surrogate-to-truth disagreement, thermal overload, missed beam windows, and station-heavy one-off wins.
4. Treat Earth-side and lunar-side beam stations as reusable logistics nodes, not disposable helpers.

## Allowed Mutation Knobs

- Search bounds and seed priors
- Optimizer budgets and hyperparameters
- Profile allocation across campaign generations
- Truth-promotion budget
- Candidate-promotion thresholds

## Iteration Policy

1. Start broad with the SEP-only baseline and single-role beam studies.
2. Promote the best candidates into truth validation using a physics-first ranking.
3. Only after credible fixed dual-window behaviour appears, launch focused follow-on campaigns around the best beam/station families.
4. Keep immutable run directories and preserve provenance through `campaign_id`, `design_space_id`, `physics_model_id`, and `git_sha`.

## Observation Priorities

- Capture enabled by beam support when the SEP-only baseline cannot close
- Marginal value of `perigee_boost` versus `lunar_brake`
- Surrogate-to-truth disagreement
- Station-burden regressions that buy little capture benefit
- Reusable station placements that remain competitive across neighboring mission cases
