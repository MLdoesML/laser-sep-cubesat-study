from __future__ import annotations

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from aether_traj.jax_surrogate import JaxSurrogateProblem, classification_from_metrics, config_from_unit, design_dict_from_unit, metrics_dict_from_unit


@dataclass(frozen=True)
class OptimizerCandidate:
    candidate_id: str
    source: str
    unit_vector: np.ndarray
    objective: float
    metrics: dict[str, float]
    design: dict[str, float]


@dataclass(frozen=True)
class OptimizerResult:
    workflow_id: str
    backend: str
    best_candidate: OptimizerCandidate
    candidates: list[OptimizerCandidate]
    history: list[dict[str, float]]


def _logit(unit: np.ndarray) -> np.ndarray:
    clipped = np.clip(unit, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(clipped / (1.0 - clipped))


def _initial_units(problem: JaxSurrogateProblem, count: int, random_seed: int, jitter: float = 0.16) -> list[np.ndarray]:
    rng = np.random.default_rng(random_seed)
    default = np.asarray(problem.search_space.default_unit, dtype=np.float64)
    units = [default]
    while len(units) < max(1, count):
        units.append(np.clip(default + rng.normal(scale=jitter, size=problem.search_space.dimension), 0.0, 1.0))
    return units


def _build_candidate(problem: JaxSurrogateProblem, workflow_id: str, source: str, index: int, unit_vector: np.ndarray) -> OptimizerCandidate:
    metrics = metrics_dict_from_unit(problem, unit_vector)
    design = design_dict_from_unit(problem.search_space, unit_vector)
    config = config_from_unit(problem.base_config, problem.search_space, unit_vector)
    metrics["classification"] = classification_from_metrics(metrics, config)
    return OptimizerCandidate(
        candidate_id=f"{workflow_id}-candidate-{index:02d}",
        source=source,
        unit_vector=np.asarray(unit_vector, dtype=np.float64),
        objective=float(metrics["objective"]),
        metrics=metrics,
        design=design,
    )


def optimize_with_adam(problem: JaxSurrogateProblem, workflow_id: str, iterations: int, seed_count: int, random_seed: int, learning_rate: float = 0.08) -> OptimizerResult:
    objective_from_logits = jax.jit(lambda logits: problem.objective_fn(jax.nn.sigmoid(logits)))
    value_and_grad = jax.jit(jax.value_and_grad(objective_from_logits))
    history: list[dict[str, float]] = []
    candidates: list[OptimizerCandidate] = []
    best_candidate: OptimizerCandidate | None = None
    global_best = float("inf")

    for seed_index, seed_unit in enumerate(_initial_units(problem, seed_count, random_seed), start=1):
        logits = jnp.asarray(_logit(seed_unit), dtype=jnp.float64)
        first_moment = jnp.zeros_like(logits)
        second_moment = jnp.zeros_like(logits)
        best_logits = logits
        best_seed_objective = float("inf")
        for iteration in range(1, max(2, iterations) + 1):
            objective_value, gradient = value_and_grad(logits)
            beta1 = 0.9
            beta2 = 0.999
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * jnp.square(gradient)
            first_hat = first_moment / (1.0 - beta1**iteration)
            second_hat = second_moment / (1.0 - beta2**iteration)
            logits = logits - learning_rate * first_hat / (jnp.sqrt(second_hat) + 1.0e-8)
            objective_scalar = float(objective_value)
            if objective_scalar < best_seed_objective:
                best_seed_objective = objective_scalar
                best_logits = logits
            global_best = min(global_best, best_seed_objective)
            history.append(
                {
                    "iteration": float(iteration),
                    "restart": float(seed_index),
                    "objective": objective_scalar,
                    "best_objective": global_best,
                }
            )
        candidate = _build_candidate(problem, workflow_id, f"adam_seed_{seed_index:02d}", seed_index, np.asarray(jax.nn.sigmoid(best_logits), dtype=np.float64))
        candidates.append(candidate)
        if best_candidate is None or candidate.objective < best_candidate.objective:
            best_candidate = candidate

    assert best_candidate is not None
    return OptimizerResult(workflow_id=workflow_id, backend="jax_adam", best_candidate=best_candidate, candidates=sorted(candidates, key=lambda item: item.objective), history=history)


def optimize_with_lbfgs(problem: JaxSurrogateProblem, workflow_id: str, iterations: int, seed_count: int, random_seed: int) -> OptimizerResult:
    history: list[dict[str, float]] = []
    candidates: list[OptimizerCandidate] = []
    best_candidate: OptimizerCandidate | None = None
    global_best = float("inf")

    def fun(unit_vector: np.ndarray) -> float:
        return float(problem.objective_fn(jnp.asarray(unit_vector, dtype=jnp.float64)))

    def jac(unit_vector: np.ndarray) -> np.ndarray:
        _, gradient = problem.value_and_grad_fn(jnp.asarray(unit_vector, dtype=jnp.float64))
        return np.asarray(gradient, dtype=np.float64)

    for seed_index, seed_unit in enumerate(_initial_units(problem, seed_count, random_seed), start=1):
        result = scipy_minimize(
            fun=fun,
            x0=np.asarray(seed_unit, dtype=np.float64),
            jac=jac,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * problem.search_space.dimension,
            options={"maxiter": max(8, iterations)},
        )
        candidate = _build_candidate(problem, workflow_id, f"lbfgs_seed_{seed_index:02d}", seed_index, np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0))
        candidates.append(candidate)
        global_best = min(global_best, candidate.objective)
        history.append(
            {
                "iteration": float(seed_index),
                "restart": float(seed_index),
                "objective": float(result.fun),
                "best_objective": global_best,
            }
        )
        if best_candidate is None or candidate.objective < best_candidate.objective:
            best_candidate = candidate

    assert best_candidate is not None
    return OptimizerResult(workflow_id=workflow_id, backend="jax_lbfgs", best_candidate=best_candidate, candidates=sorted(candidates, key=lambda item: item.objective), history=history)


def optimize_with_de(problem: JaxSurrogateProblem, workflow_id: str, generations: int, population_size: int, random_seed: int, mutation_factor: float = 0.8, crossover_rate: float = 0.9) -> OptimizerResult:
    rng = np.random.default_rng(random_seed)
    population = np.asarray(_initial_units(problem, max(4, population_size), random_seed), dtype=np.float64)
    objective_batch = lambda values: np.asarray(problem.batched_objective_fn(jnp.asarray(values, dtype=jnp.float64)), dtype=np.float64)
    scores = objective_batch(population)
    history: list[dict[str, float]] = []

    for generation in range(1, max(2, generations) + 1):
        trials = population.copy()
        for index in range(population.shape[0]):
            candidates = [item for item in range(population.shape[0]) if item != index]
            a_idx, b_idx, c_idx = rng.choice(candidates, size=3, replace=False)
            mutant = np.clip(population[a_idx] + mutation_factor * (population[b_idx] - population[c_idx]), 0.0, 1.0)
            crossover_mask = rng.random(problem.search_space.dimension) < crossover_rate
            crossover_mask[rng.integers(0, problem.search_space.dimension)] = True
            trials[index] = np.where(crossover_mask, mutant, population[index])
        trial_scores = objective_batch(trials)
        improved = trial_scores < scores
        population = np.where(improved[:, None], trials, population)
        scores = np.where(improved, trial_scores, scores)
        history.append({"iteration": float(generation), "objective": float(scores.min()), "best_objective": float(scores.min())})

    ordering = np.argsort(scores)
    candidates = [_build_candidate(problem, workflow_id, "de_population", rank + 1, population[index]) for rank, index in enumerate(ordering[: min(len(ordering), max(4, problem.search_space.dimension))])]
    best_candidate = min(candidates, key=lambda item: item.objective)
    return OptimizerResult(workflow_id=workflow_id, backend="jax_de", best_candidate=best_candidate, candidates=sorted(candidates, key=lambda item: item.objective), history=history)


def optimize_with_pso(
    problem: JaxSurrogateProblem,
    workflow_id: str,
    iterations: int,
    swarm_size: int,
    random_seed: int,
    inertia_start: float = 0.9,
    inertia_end: float = 0.4,
    cognitive_weight: float = 1.4,
    social_weight: float = 1.6,
    max_velocity: float = 0.18,
) -> OptimizerResult:
    rng = np.random.default_rng(random_seed)
    positions = np.asarray(_initial_units(problem, max(4, swarm_size), random_seed), dtype=np.float64)
    velocities = rng.normal(scale=0.05, size=positions.shape)
    objective_batch = lambda values: np.asarray(problem.batched_objective_fn(jnp.asarray(values, dtype=jnp.float64)), dtype=np.float64)
    scores = objective_batch(positions)
    personal_best_positions = positions.copy()
    personal_best_scores = scores.copy()
    best_index = int(np.argmin(scores))
    global_best_position = positions[best_index].copy()
    global_best_score = float(scores[best_index])
    history: list[dict[str, float]] = []

    for iteration in range(1, max(2, iterations) + 1):
        inertia = inertia_start + (inertia_end - inertia_start) * (iteration - 1) / max(iterations - 1, 1)
        r1 = rng.random(size=positions.shape)
        r2 = rng.random(size=positions.shape)
        velocities = inertia * velocities + cognitive_weight * r1 * (personal_best_positions - positions) + social_weight * r2 * (global_best_position[None, :] - positions)
        velocities = np.clip(velocities, -max_velocity, max_velocity)
        positions = np.clip(positions + velocities, 0.0, 1.0)
        scores = objective_batch(positions)
        improved = scores < personal_best_scores
        personal_best_positions = np.where(improved[:, None], positions, personal_best_positions)
        personal_best_scores = np.where(improved, scores, personal_best_scores)
        best_index = int(np.argmin(personal_best_scores))
        global_best_position = personal_best_positions[best_index].copy()
        global_best_score = float(personal_best_scores[best_index])
        history.append({"iteration": float(iteration), "objective": global_best_score, "best_objective": global_best_score})

    ordering = np.argsort(personal_best_scores)
    candidates = [_build_candidate(problem, workflow_id, "pso_personal_best", rank + 1, personal_best_positions[index]) for rank, index in enumerate(ordering[: min(len(ordering), max(4, problem.search_space.dimension))])]
    best_candidate = min(candidates, key=lambda item: item.objective)
    return OptimizerResult(workflow_id=workflow_id, backend="jax_pso", best_candidate=best_candidate, candidates=sorted(candidates, key=lambda item: item.objective), history=history)
