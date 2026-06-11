"""Genetic algorithm that evolves the car-driving neural networks.

Algorithm per generation:

    1. Rank cars by fitness (descending).
    2. Keep the top ELITISM_COUNT individuals UNCHANGED in the next gen
       (true elitism — guarantees performance cannot regress).
    3. Take the top ELITE_FRACTION as the breeding pool.
    4. Fill the remaining slots by repeating:
         - pick two random parents from the elite pool
         - uniform crossover their flat weight vectors
         - per-gene Gaussian mutation
    5. Build fresh Car objects at the track's start pose, each driven by
       one of the new weight vectors.
"""

import random
import numpy as np

import config as C
from car import Car
from neural_network import NeuralNetwork


class Evolution:
    """Tracks population statistics and produces each new generation."""

    def __init__(self):
        self.generation = 1
        self.best_fitness_ever = 0.0
        self.best_fitness_this_gen = 0.0
        self.avg_fitness = 0.0
        # Rolling list of "best fitness per generation" used by the live graph.
        self.history = []

    # ------------------------------------------------------------------
    # Population creation
    # ------------------------------------------------------------------

    def spawn_initial_population(self, track):
        """Create POPULATION_SIZE cars with fresh random networks at the start pose."""
        sx, sy, sa = track.get_start_pose()
        return [Car(sx, sy, sa, NeuralNetwork()) for _ in range(C.POPULATION_SIZE)]

    def new_generation(self, cars, track):
        """Score the current generation and return a fresh list of evolved cars."""
        # 1. Rank by fitness (highest first).
        ranked = sorted(cars, key=lambda c: c.fitness, reverse=True)

        # 2. Update stats and rolling history.
        fitnesses = [c.fitness for c in ranked]
        self.best_fitness_this_gen = fitnesses[0]
        self.best_fitness_ever = max(self.best_fitness_ever, self.best_fitness_this_gen)
        self.avg_fitness = sum(fitnesses) / len(fitnesses)
        self.history.append(self.best_fitness_this_gen)
        if len(self.history) > C.HISTORY_LENGTH:
            self.history = self.history[-C.HISTORY_LENGTH:]

        # 3. Choose elites: the top ELITE_FRACTION of the population becomes
        #    the breeding pool. We always keep at least 2 so crossover has
        #    two distinct parents to pick from.
        elite_count = max(2, int(C.POPULATION_SIZE * C.ELITE_FRACTION))
        elites = ranked[:elite_count]
        elite_weights = [c.nn.get_weights() for c in elites]

        # 4. Build next generation's weight vectors.
        next_weights = []
        # 4a. Elitism — the top performers survive unchanged.
        for i in range(min(C.ELITISM_COUNT, len(elite_weights))):
            next_weights.append(elite_weights[i].copy())
        # 4b. Fill the remainder via crossover + mutation.
        while len(next_weights) < C.POPULATION_SIZE:
            p1, p2 = random.sample(elite_weights, 2)
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            next_weights.append(child)

        # 5. Instantiate fresh Car objects at the start pose.
        sx, sy, sa = track.get_start_pose()
        next_cars = [Car(sx, sy, sa, NeuralNetwork(weights=w)) for w in next_weights]

        self.generation += 1
        return next_cars

    # ------------------------------------------------------------------
    # Genetic operators (act on flat 1D weight vectors)
    # ------------------------------------------------------------------

    def _crossover(self, w1, w2):
        """Uniform crossover: each gene is independently picked from either parent."""
        # Boolean mask: True = take this gene from parent 1, False = parent 2.
        mask = np.random.rand(w1.size) < 0.5
        return np.where(mask, w1, w2)

    def _mutate(self, w):
        """Per-weight Gaussian mutation. Each gene mutates independently."""
        # Decide per-gene whether to perturb, then add a Gaussian of size
        # MUTATION_SIGMA to those genes. This is the only source of brand-new
        # genetic material once the initial random pool exists.
        mutate_mask = np.random.rand(w.size) < C.MUTATION_RATE
        noise = np.random.randn(w.size) * C.MUTATION_SIGMA
        return w + mutate_mask * noise

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Wipe stats and counter back to generation 1."""
        self.generation = 1
        self.best_fitness_ever = 0.0
        self.best_fitness_this_gen = 0.0
        self.avg_fitness = 0.0
        self.history = []
