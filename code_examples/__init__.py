"""
Dynamic Programming algorithms for Markov Decision Processes.

This package provides implementations of:
- Value Iteration (standard, Gauss-Seidel, Jacobi, Prioritized Sweeping)
- Policy Iteration (standard, exact, modified)
- Grid World MDP environment

Example:
    >>> from code_examples.mdp import GridWorld
    >>> from code_examples.value_iteration import ValueIteration
    >>> 
    >>> gw = GridWorld(grid_size=4, goal_pos=(3, 3))
    >>> vi = ValueIteration(gw)
    >>> policy, V, _ = vi.run(gamma=0.9)
"""

from .mdp import MDP, GridWorld
from .value_iteration import (
    ValueIteration,
    GaussSeidelValueIteration,
    JacobiValueIteration,
    PrioritizedSweepingVI,
)
from .policy_iteration import (
    policy_iteration,
    policy_iteration_exact,
    modified_policy_iteration,
)

__all__ = [
    "MDP",
    "GridWorld",
    "ValueIteration",
    "GaussSeidelValueIteration",
    "JacobiValueIteration",
    "PrioritizedSweepingVI",
    "policy_iteration",
    "policy_iteration_exact",
    "modified_policy_iteration",
]
