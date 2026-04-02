# Mission Definition

## Purpose

This file locks the v1 mission assumptions for the CAPSTONE-class
laser-augmented SEP study.

## Spacecraft

- mission class: CAPSTONE-like cislunar pathfinder CubeSat
- wet mass: `25 kg`
- dry mass: `18 kg`
- usable propellant: `5 kg`
- protected reserve: `2 kg`

## Propulsion

- propulsion type: micro-Hall SEP
- specific impulse range: `1400-1600 s`
- nominal `Isp`: `1500 s`
- thruster efficiency range: `0.40-0.50`
- nominal efficiency: `0.45`

## Power

- onboard solar power range: `100-150 W`
- nominal onboard solar power: `120 W`
- beam receiver electrical efficiency range: `0.35-0.50`
- nominal beam receiver efficiency: `0.45`
- spacecraft propulsion electrical cap range: `500-700 W`
- nominal propulsion electrical cap: `600 W`

## Launch

- baseline launch lane: `GTO secondary`
- rationale: credible energy state for a small lunar pathfinder and a better
  first closure target than generic LEO rideshare

## Beam Roles

Two beam roles are allowed in v1.

### `perigee_boost`

- applied near repeated Earth perigee passes
- objective: raise outbound energy efficiently
- station families: `GEO`, `high Earth elliptical relay`, `supersynchronous relay`

### `lunar_brake`

- applied near lunar approach and early bound-orbit shaping
- objective: increase braking authority for direct capture
- station families: `EML1 halo`, `NRHO`, `DRO`

## Out of Scope

- photon sails
- direct photon-pressure propulsion
- ground-based laser systems
- chemical LOI in the baseline architecture
- more than two stations

## Success Criteria

The mission is considered successful only if all criteria below are satisfied.

- stable bound lunar orbit for `>= 3 days`
- perilune altitude between `100 km` and `1000 km`
- capture apoapsis `<= 30,000 km`
- final mass greater than or equal to `20 kg`

## Secondary Evaluation Criteria

- low propellant consumption
- low time of flight
- low operations burden
- low infrastructure burden
- station placement reuse across nearby mission cases
