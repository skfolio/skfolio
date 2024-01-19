"""Fast non-dominated sorting module"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

import numpy as np

__all__ = ["dominate", "non_denominated_sort"]


def dominate(fitness_1: np.ndarray, fitness_2: np.ndarray) -> bool:
    """Compute the domination of two fitness arrays.

    Domination of `fitness_1` over `fitness_2` means that each objective (value) of
    `fitness_1` is not strictly worse than the corresponding objective of `fitness_2`
    and at least one objective is strictly better.

    Parameters
    ----------
    fitness_1 : ndarray of floats of shape (n_objectives,)
        Fitness array 1.

    fitness_2 : ndarray of floats of shape (n_objectives,)
        Fitness array 2.

    Returns
    -------
    is_dominated : bool
        Ture if `fitness_1` dominates `fitness_2`, False otherwise.
    """
    if fitness_1.ndim != fitness_2.ndim != 1:
        raise ValueError("fitness_1 and fitness_2 must be 1D array")
    not_equal = False
    for self_value, other_value in zip(fitness_1, fitness_2, strict=True):
        if self_value > other_value:
            not_equal = True
        elif self_value < other_value:
            return False
    return not_equal


def non_denominated_sort(
    fitnesses: np.ndarray, first_front_only: bool
) -> list[list[int]]:
    """Fast non-dominated sorting.

    Sort the fitnesses into different non-domination levels.
    Complexity O(MN^2) where M is the number of objectives and N the number of
    portfolios.

    Parameters
    ----------
    fitnesses: ndarray of shape(n, n_fitness)
        Fitnesses array.

    first_front_only : bool
        If this is set to True, only the first front is computed and returned.

    Returns
    -------
    fronts: list[list[int]]
      A list of Pareto fronts (lists), the first list includes non-dominated fitnesses.
    """
    n = len(fitnesses)
    fronts = []
    if n == 0:
        return fronts

    # final rank that will be returned
    n_ranked = 0
    ranked = np.array([0 for _ in range(n)])

    # for each portfolio a list of all portfolios that are dominated by this one
    is_dominating = [[x for x in range(0)] for _ in range(n)]

    # storage for the number of solutions dominated this one
    n_dominated = [0 for _ in range(n)]

    current_front = [x for x in range(0)]

    for i in range(n):
        for j in range(i + 1, n):
            if dominate(fitnesses[i], fitnesses[j]):
                is_dominating[i].append(j)
                n_dominated[j] += 1
            elif dominate(fitnesses[j], fitnesses[i]):
                is_dominating[j].append(i)
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            current_front.append(i)
            ranked[i] = 1.0
            n_ranked += 1

    # append the first front to the current front
    fronts.append(current_front)

    if first_front_only:
        return fronts

    # while not all solutions are assigned to a pareto front
    while n_ranked < n:
        next_front = []
        # for each portfolio in the current front
        for i in current_front:
            # all solutions that are dominated by this portfolio
            for j in is_dominating[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    next_front.append(j)
                    ranked[j] = 1.0
                    n_ranked += 1

        fronts.append(next_front)
        current_front = next_front

    return fronts
