# CAPSTONE-Class Laser-Augmented SEP Study Plan

## 1. Study Purpose

This study asks whether a CAPSTONE-class CubeSat can achieve direct lunar
capture more credibly when onboard micro-Hall SEP is augmented by reusable
cislunar laser power-beaming infrastructure.

The work is not framed as a one-off best-trajectory search. It is framed as a
mission-and-infrastructure co-design problem:

- spacecraft trajectory design
- beam timing and operations design
- Earth-side and lunar-side station placement
- infrastructure reuse across nearby mission variants

## 2. Primary Mission Thesis

The central hypothesis is:

> A CAPSTONE-scale spacecraft that cannot reliably close direct lunar capture
> with SEP alone may be able to do so when external beam power is applied in two
> operationally targeted phases: repeated Earth perigee boost and lunar approach
> braking.

The paper-worthy contribution is not "laser power beaming exists" or "CubeSat
low-thrust design exists." The contribution is the integrated co-design of:

- CAPSTONE-class direct-capture mission geometry
- beam-augmented onboard SEP
- reusable cislunar station placement
- logistics-aware evaluation of infrastructure reuse and operating burden

## 3. Logistics Framing

The study should use a logistics-centered design lens inspired by the published
space-logistics work of researchers such as Koki Ho:

- infrastructure should be treated as persistent and reusable
- the design should be judged at both mission scale and network scale
- infrastructure burden matters, not just trajectory performance
- service cadence, reuse, and operability matter alongside propellant use

This framing must be written carefully in any manuscript:

- do not say the study reflects what Koki Ho personally endorses
- do say that the framing is consistent with published work on reusable
  infrastructure, campaign-level modeling, and logistics-aware optimization

## 4. Mission Definition

### 4.1 Spacecraft Class

The reference spacecraft is a CAPSTONE-like pathfinder CubeSat, scaled for a
credible flight demonstrator rather than a heavier smallsat.

Reference mass model:

- wet mass: `25 kg`
- dry mass: `18 kg`
- usable propellant: `5 kg`
- protected reserve: `2 kg`

### 4.2 Propulsion and Power

Reference propulsion model:

- micro-Hall SEP
- specific impulse: `1400-1600 s` (nominal `1500 s`)
- thruster efficiency: `0.40-0.50` (nominal `0.45`)
- onboard solar power: `100-150 W` (nominal `120 W`)
- beam-to-electric receiver efficiency: `0.35-0.50` (nominal `0.45`)
- spacecraft propulsion power cap: `500-700 W` (nominal `600 W`)

Laser support is modeled as extra electrical power available to the SEP system.
It does not directly produce spacecraft force.

### 4.3 Launch Lane

Baseline launch lane:

- `GTO secondary`

This is more credible than a generic LEO rideshare for a first closure attempt
at this spacecraft scale.

### 4.4 Direct-Capture Success Gate

A mission counts as successful only if it satisfies all of the following:

- stable bound lunar orbit for at least `3 days`
- perilune altitude between `100 km` and `1000 km`
- capture apoapsis no larger than `30,000 km`
- final mass `≥ 20 kg` (dry mass `18 kg` + reserve `2 kg`)

Secondary mission-quality goals:

- low time of flight
- low mission operations burden
- low station burden
- station placement reuse across nearby flights

## 5. Beam-Support Architecture

The v1 architecture is limited to at most two space-based beam stations.

### 5.1 Outbound Beam Role

Outbound beam support is applied near repeated Earth perigee passes:

- role name: `perigee_boost`
- purpose: increase outbound energy growth efficiently during high-velocity
  passes
- primary question: does Earth-side beam support reduce transfer time and make
  later capture easier enough to justify the station?

### 5.2 Inbound Beam Role

Inbound beam support is applied near the Moon:

- role name: `lunar_brake`
- purpose: increase available SEP braking power during approach and early bound
  orbit shaping
- primary question: does lunar-side beam support convert near-capture flybys
  into direct capture more efficiently than outbound boosting alone?

### 5.3 Architecture Scope Limits

Out of scope for v1:

- ground-based lasers
- photon sails
- direct beam-to-force propulsion
- chemical LOI in the baseline architecture
- more than two stations

## 6. Design Space

### 6.1 Spacecraft Variables

The spacecraft optimization space should include:

- departure apogee
- departure perigee altitude
- departure inclination
- RAAN
- argument of latitude
- epoch shift
- departure wait
- speed scale
- flight-path angle
- number of boosted perigee passes: `1-5`
- SEP start delay
- 4-knot RTN steering:
  - throttle
  - azimuth
  - elevation

### 6.2 Station Families

Candidate `perigee_boost` station families:

- `GEO`
- `high Earth elliptical relay`
- `supersynchronous relay`

Candidate `lunar_brake` station families:

- `EML1 halo`
- `NRHO`
- `DRO`

### 6.3 Continuous Station Variables

Within each selected family, optimize:

- orbit-family amplitude or size
- phase placement
- beam source power
- aperture
- duty cycle
- allowed pointing cone

## 7. Technical Model

### 7.1 Propagation State

The propagated state must be 7D:

`(x, y, z, vx, vy, vz, mass_kg)`

### 7.2 Power and Thrust Coupling

Available propulsion power is:

`available_power = onboard_solar + received_beam_power`

The received beam power is limited by:

- line of sight
- occultation by Earth or Moon
- range loss
- aperture limits
- pointing limits
- receiver efficiency
- thermal cap
- station duty-cycle cap

Thrust is then computed from available electrical power, thruster efficiency,
and exhaust velocity. Mass flow follows from thrust and `Isp`.

### 7.3 Search and Truth Models

Search surrogate:

- fixed-step RK4 or comparable non-symplectic scheme suitable for variable-mass
  low-thrust propagation

Truth validation:

- `DOP853`
- direct ephemerides
- stronger lunar gravity model

### 7.4 Required Operational Realism

The minimum operational realism for publication should include:

- station eclipse or source-power availability
- beam acquisition window
- minimum engagement duration threshold
- station retargeting penalty or slew-rate burden

The model should not become a pure controls study. The point is mission design
with enforceable but compact operational realism.

## 8. Study Ladder

### 8.1 SEP-Only Baseline (Phase A)

Purpose:

- establish the control case
- identify near-capture orbit families without beaming

Key output:

- map of capture gap for CAPSTONE-class SEP-only missions

Decision gate:

- determine whether the system is close enough to capture that beam support
  is a plausible closer rather than a wholesale replacement; if not, revise
  mission scope before proceeding

### 8.2 Perigee-Boost Only (Phase B1)

Purpose:

- isolate the value of repeated Earth-side beam support

Key question:

- does outbound boosting near perigee materially reduce mission burden or open a
  new family of lunar-approach conditions?

### 8.3 Lunar-Brake Only (Phase B2)

Purpose:

- isolate the value of lunar-side beam support

Key question:

- is beam energy more valuable when spent near capture rather than near
  departure?

### 8.4 Dual-Window Fixed Architecture (Phase C1)

Purpose:

- prove the concept with one fixed Earth-side station and one fixed lunar-side
  station before full co-design

Key question:

- does the basic architecture close at all under realistic constraints?

Decision gate:

- confirm that joint co-design of orbit and station placement is justified
  before proceeding to C2; if fixed architecture already fails to close, revise
  station families or beam budget

### 8.5 Dual-Window Co-Design (Phase C2)

Purpose:

- jointly optimize spacecraft trajectory, beam timing, and both station
  placements

This is the main publication result.

### 8.6 Infrastructure Reuse Extension (Phase D1)

Purpose:

- hold station placements fixed and test nearby mission variants

Key question:

- are the chosen stations reusable infrastructure nodes or only one-mission
  helpers?

### 8.7 Operational Robustness (Phase D2)

Purpose:

- test whether the beam-assisted architecture survives realistic operational
  perturbations

Perturbations to apply:

- timing error
- pointing loss
- missed beam engagement
- reduced duty cycle
- station outage

Key metrics:

- sensitivity of capture success to each perturbation
- sensitivity of propellant use and final mass

## 9. Objectives and Constraints

### 9.1 Scalar Objective

The scalar ranking should reward:

- direct-capture success
- low propellant used
- low time of flight
- low station burden
- low operations burden
- strong beam engagement margin

### 9.2 Multi-Objective Formulation

The NSGA-II vector should minimize:

- propellant used
- time of flight
- capture apoapsis error
- infrastructure burden

### 9.3 Hard Constraints

Treat the following as hard constraints:

- no direct capture
- unsafe perilune
- reserve violation
- thermal overload
- insufficient beam dwell
- infeasible station geometry

### 9.4 Required Logistics Metrics

Each shortlisted design must report:

- station utilization fraction
- total beam energy delivered
- average retargeting burden
- worst-case retargeting burden
- sensitivity to a missed engagement window
- reuse score across neighboring mission cases

## 10. Novelty and Publication Positioning

The paper should not claim novelty in:

- laser power beaming itself
- CubeSat low-thrust mission design itself
- cislunar logistics as a field

The novelty should be explicitly stated as the integration of:

- CAPSTONE-class direct-capture mission design
- beam-augmented onboard SEP
- transfer-orbit and station-placement co-design
- reusable infrastructure evaluation
- truth-validated capture under operational constraints

The strongest publication framing is a methods paper with a case study.

Likely target venues:

- `Acta Astronautica`
- `Journal of Spacecraft and Rockets`
- `Journal of Guidance, Control, and Dynamics`

## 11. Acceptance Criteria

The study is publication-ready only if it yields all of the following:

- a validated SEP-only baseline
- a validated dual-window beam-assisted case
- explicit station placements, not vague family labels
- a clear marginal-value story for perigee boost and lunar braking
- at least one reuse analysis across neighboring missions
- an uncertainty analysis for timing, pointing, and station outage

If direct capture still does not close, the fallback paper is still acceptable
if it clearly demonstrates:

- where beam support helps
- where it does not
- which station placements are operationally attractive
- why reusable infrastructure is hard at this spacecraft scale

## 12. Minimum Publishable Package

The smallest acceptable result set for a submission is:

- **A1** SEP-only baseline with capture-gap map
- **B1** perigee-boost only comparison
- **B2** lunar-brake only comparison
- **C2** dual-window co-design with explicit station placements and Pareto set
- **D1** one compact infrastructure reuse analysis
- **D2** one compact operational robustness analysis

Experiments C1 (dual-window fixed) is a stepping stone, not a required
publication result, but must pass its decision gate before proceeding to C2.

## 13. Immediate Implementation Priorities

Before starting implementation, run through `docs/review_questions.md` and
confirm the go/no-go statement is satisfied.

Phase 0 — lock definitions:

- lock mission definition (`docs/mission_definition.md`)
- lock publication framing (`docs/publication_positioning.md`)
- lock reference spacecraft and station families

Phase 1 — infrastructure (pre-Phase A):

- implement config files and data models
- implement 7D propagation scaffold
- implement beam geometry and power-budget logic

Phase 2 — control and single-role studies (Phases A and B):

- implement SEP-only baseline (A1) and collect capture-gap map
- pass Phase A decision gate before continuing
- implement perigee-boost only (B1) and lunar-brake only (B2)

Phase 3 — architecture closure (Phase C):

- implement dual-window fixed architecture (C1) and pass decision gate
- implement dual-window co-design (C2) as primary publication result

Phase 4 — logistics extension (Phase D):

- implement infrastructure reuse analysis (D1)
- implement operational robustness analysis (D2)

## 14. Key References

- Koki Ho profile, Georgia Tech:
  `https://space.gatech.edu/koki-ho`
- Ho, "Modeling and Optimization for Space Logistics Operations: Review of State
  of the Art" (2023 preprint):
  `https://doi.org/10.48550/arXiv.2306.01107`
- Jagannatha and Ho, "Event-driven space logistics network optimization for
  cislunar supply chain design with high-thrust and low-thrust propulsion
  technologies" (2018):
  `https://experts.illinois.edu/en/publications/event-driven-space-logistics-network-optimization-for-cislunar-su/`
- NASA CAPSTONE mission page:
  `https://www.nasa.gov/smallspacecraft/capstone/`
- NASA Glenn, "Power Beaming From Lunar Orbit to Small Lunar Science Assets"
  (2024):
  `https://ntrs.nasa.gov/citations/20230018523`
- Williams et al., "A solar electric propulsion mission for lunar power
  beaming" (2009):
  `https://www.sciencedirect.com/science/article/abs/pii/S0094576509000629`
