# Dynamic Programming for MDPs

Code examples demonstrating Value Iteration and Policy Iteration algorithms for solving Markov Decision Processes (MDPs).

## Overview

This module provides clean, well-documented implementations of classic dynamic programming algorithms used in reinforcement learning:

- **Value Iteration** - Iteratively apply Bellman optimality operator
- **Policy Iteration** - Alternate between policy evaluation and improvement
- **Variants** - Gauss-Seidel, Jacobi, Prioritized Sweeping, Modified PI

All algorithms are demonstrated on a **Grid World** environment - a standard RL testbed.

## Files

| File | Description |
|------|-------------|
| `mdp.py` | MDP base class and Grid World environment |
| `value_iteration.py` | Value Iteration algorithm variants |
| `policy_iteration.py` | Policy Iteration algorithm variants |
| `demo.py` | Demonstration script with examples |

## Quick Start

```bash
# Run the demonstration
python demo.py
```

## Usage

### Creating a Grid World

```python
from mdp import GridWorld

# Create a 4x4 grid with goal and hazards
gw = GridWorld(
    grid_size=4,
    goal_pos=(3, 3),           # Top-right corner
    goal_reward=10.0,           # Reward for reaching goal
    hazard_positions=[(1, 2)],  # Hazard locations
    hazard_reward=-5.0,         # Penalty for hazards
    step_cost=-0.04,            # Small cost per step
    slip_prob=0.1,              # 10% slip to perpendicular
    goal_terminal=True          # Goal is absorbing state
)
```

### Running Value Iteration

```python
from value_iteration import ValueIteration

vi = ValueIteration(gw)
policy, V, history = vi.run(
    theta=0.001,    # Convergence threshold
    gamma=0.9       # Discount factor
)

# Display results
gw.print_values(V, "Optimal Values")
gw.print_policy(policy, "Optimal Policy")
```

### Running Policy Iteration

```python
from policy_iteration import policy_iteration, policy_iteration_exact

# Standard PI (iterative evaluation)
V, policy, iters, _ = policy_iteration(gw, gamma=0.9)

# PI with exact matrix inversion
V, policy, iters, _ = policy_iteration_exact(gw, gamma=0.9)
```

### Comparing VI Variants

```python
from value_iteration import (
    ValueIteration,
    GaussSeidelValueIteration,
    JacobiValueIteration,
    PrioritizedSweepingVI,
)

# Standard VI
vi = ValueIteration(gw)
policy, V, _ = vi.run(gamma=0.9)

# Gauss-Seidel (uses updates immediately)
vi_gs = GaussSeidelValueIteration(gw)
policy, V, _ = vi_gs.run(gamma=0.9)

# Prioritized Sweeping (updates highest-error states first)
vi_ps = PrioritizedSweepingVI(gw)
policy, V, _ = vi_ps.run(gamma=0.9)
```

## Algorithms

### Value Iteration

Repeatedly applies the Bellman optimality operator until convergence:

$$V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} T(s'|s,a) V_k(s') \right]$$

**Variants:**
- **Standard**: Uses values from previous sweep for all updates
- **Gauss-Seidel**: Uses updated values immediately within sweep
- **Jacobi**: Explicitly handles self-loops for faster convergence
- **Prioritized Sweeping**: Updates states by Bellman error priority

### Policy Iteration

Alternates between evaluation and improvement:

1. **Policy Evaluation**: Compute $V^\pi$ for current policy
2. **Policy Improvement**: Update $\pi$ to be greedy w.r.t. $V^\pi$

**Variants:**
- **Standard**: Iterative evaluation until convergence
- **Exact (Matrix Inversion)**: Solve $V = (I - \gamma T^\pi)^{-1} R^\pi$
- **Modified**: Limited evaluation steps (hybrid VI/PI)

## Grid World Environment

The Grid World is a common RL testbed:

```
+---+---+---+---+
|   | H |   | G |   G = Goal (+10 reward, terminal)
+---+---+---+---+   H = Hazard (-5 penalty)
|   |   |   |   |
+---+---+---+---+   Agent starts anywhere, tries to
|   |   | H |   |   reach G while avoiding H.
+---+---+---+---+
|   |   |   |   |   Stochastic: 80% intended direction,
+---+---+---+---+   10% slip to each perpendicular.
```

**Coordinate System:**
- (x, y) where x=column (0=left), y=row (0=bottom)
- State index: `s = y * grid_size + x`

## Dependencies

- NumPy

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction" (Ch. 4)
- Puterman, "Markov Decision Processes" (Ch. 6)
- Bertsekas, "Dynamic Programming and Optimal Control"

## Authors

Based on original code by Peter Henderson and Wei-Di Chang.
Cleaned up and documented for COS 435 Reinforcement Learning.
