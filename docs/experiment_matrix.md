# Experiment Matrix

## Purpose

This matrix is the first-pass campaign structure for the study. It is designed
to answer the publication questions in the smallest credible sequence.

## Phase A: Control

### A1. SEP-only baseline

Goal:

- quantify the direct-capture gap for the CAPSTONE-class spacecraft without beam
  support

Outputs:

- best SEP-only candidate
- top failure modes
- orbit-family map of near-capture cases

Decision gate:

- identify whether the system is close enough to capture that beam support is a
  plausible closer rather than a complete replacement

## Phase B: Single-Role Beam Studies

### B1. Perigee-boost only

Goal:

- quantify the value of repeated Earth-side boosting

Variables:

- number of boosted perigee passes
- Earth-side station family
- beam power
- aperture
- duty cycle

Key metrics:

- time-of-flight reduction
- propellant saved
- arrival-energy reduction
- station utilization

### B2. Lunar-brake only

Goal:

- quantify the value of lunar-side braking support

Variables:

- lunar-side station family
- beam power
- aperture
- phase placement
- duty cycle

Key metrics:

- conversion of flyby cases into direct capture
- perilune safety
- capture duration
- beam dwell near first encounter

## Phase C: Architecture Closure

### C1. Dual-window fixed architecture

Goal:

- prove the concept under fixed reference station placements

Outputs:

- one fixed Earth-side station case
- one fixed lunar-side station case
- comparison against both single-role studies

Decision gate:

- determine whether joint co-design is justified

### C2. Dual-window co-design

Goal:

- jointly optimize orbit and station placement

Outputs:

- primary publication architecture
- explicit station placements
- explicit beam windows
- Pareto trade set

## Phase D: Logistics Extension

### D1. Infrastructure reuse

Goal:

- evaluate whether the chosen station pair is reusable

Method:

- hold station placements fixed
- rerun a small neighborhood of nearby launch and timing cases

Key metrics:

- reuse score
- mission success rate across nearby cases
- utilization under notional campaign demand

### D2. Operational robustness

Goal:

- test whether the beam-assisted architecture is operationally credible

Perturbations:

- timing error
- pointing loss
- missed beam engagement
- reduced duty cycle
- station outage

Key metrics:

- sensitivity of capture success
- sensitivity of propellant use
- sensitivity of final mass

## Minimum Publishable Package

The smallest acceptable package for publication is:

- A1 SEP-only baseline
- B1 perigee-boost only
- B2 lunar-brake only
- C2 dual-window co-design
- D1 one compact reuse analysis
- D2 one compact robustness analysis
