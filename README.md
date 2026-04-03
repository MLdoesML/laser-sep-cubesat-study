# Aether Laser-SEP CubeSat Study

This is a standalone study workspace for a CAPSTONE-class, laser-augmented SEP
direct-lunar-capture research program.

Status: standalone study scaffold plus first CLI-only experiment foundation.

The current focus is to co-design:

- the spacecraft transfer orbit
- the timing and role of beam support
- the placement of reusable cislunar laser-power stations

The laser is modeled as an external power source for onboard electric
propulsion. It is not treated as photon-pressure propulsion.

Core review documents:

- `docs/STUDY_PLAN.md`
- `docs/mission_definition.md`
- `docs/publication_positioning.md`
- `docs/experiment_matrix.md`
- `docs/review_questions.md`

Initial configuration files:

- `configs/vehicles/capstone_class_demo_v1.toml`
- `configs/profiles/sep_baseline_direct_capture.toml`
- `configs/profiles/laser_dual_window_fixed.toml`
- `configs/profiles/laser_dual_window_codesign.toml`
- `campaign.toml`
- `manager_program.md`

Initial command surface:

- `aether-study-jax`
- `aether-study-lbfgs`
- `aether-study-de`
- `aether-study-pso`
- `aether-experiment`
- `aether-campaign`
- `aether-worker`

First milestone notes:

- The surrogate layer is now written in JAX and differentiable.
- The initial optimizer breadth covers Adam (`sep_jax`), L-BFGS-B (`sep_lbfgs`), Differential Evolution (`sep_de`), and PSO (`sep_pso`).
- The campaign tooling is designed to support autoresearch-style experiment management and follow-on campaigns.
- The surrogate and truth models are intentionally separated so we can evolve the physics validation stack without rewriting the campaign layer.

This workspace is intentionally isolated from the existing solar-sail repo.
