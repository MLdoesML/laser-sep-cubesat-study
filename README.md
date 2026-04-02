# Aether Laser-SEP CubeSat Study

This is a standalone study workspace for a CAPSTONE-class, laser-augmented SEP
direct-lunar-capture research program.

Status: study definition and scaffold.

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
- `configs/profiles/laser_dual_window_codesign.toml`

This workspace is intentionally isolated from the existing solar-sail repo.
