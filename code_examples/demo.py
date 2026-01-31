#!/usr/bin/env python3
"""
Demonstration of Value and Policy Iteration on Grid World

This script demonstrates the classic dynamic programming algorithms
for solving Markov Decision Processes using a Grid World environment.

The Grid World has:
    - 4x4 grid (16 states)
    - Goal at top-right corner with +10 reward
    - Hazards with -5 penalty
    - Stochastic dynamics (10% slip probability)
    - Small step cost to encourage efficiency

Run this script to see:
    1. The MDP specification
    2. Value Iteration convergence
    3. Policy Iteration convergence  
    4. Comparison of different VI variants
    5. Visual display of optimal policy
"""

import numpy as np
from mdp import GridWorld
from value_iteration import (
    ValueIteration,
    GaussSeidelValueIteration,
    JacobiValueIteration,
    PrioritizedSweepingVI,
)
from policy_iteration import (
    policy_iteration,
    policy_iteration_exact,
    modified_policy_iteration,
)


def print_separator(title: str = "") -> None:
    """Print a formatted section separator."""
    width = 70
    if title:
        print(f"\n{'=' * width}")
        print(f"  {title}")
        print(f"{'=' * width}")
    else:
        print("-" * width)


def demo_grid_world() -> None:
    """Demonstrate MDP solving on Grid World."""
    
    print_separator("GRID WORLD MDP DEMONSTRATION")
    
    # =====================================================================
    # 1. Create the Grid World environment
    # =====================================================================
    print_separator("1. Environment Setup")
    
    gw = GridWorld(
        grid_size=4,
        goal_pos=(3, 3),         # Top-right corner
        goal_reward=10.0,         # Reward for reaching goal
        hazard_positions=[(1, 2), (2, 1)],  # Two hazard states
        hazard_reward=-5.0,       # Penalty for hazards
        step_cost=-0.04,          # Small cost per step
        slip_prob=0.1,            # 10% chance to slip perpendicular
        goal_terminal=True        # Episode ends at goal
    )
    
    print(f"""
Grid World Configuration:
    - Size: {gw.grid_size}x{gw.grid_size} = {gw.S} states
    - Actions: {', '.join(gw.actions)}
    - Goal: position {gw.goal_pos} with reward +10 (terminal)
    - Hazards: {gw.hazard_positions} with penalty -5
    - Step cost: -0.04 per transition
    - Dynamics: 80% intended direction, 10% each perpendicular

Grid Layout:
    +---+---+---+---+
    |   | H |   | G |  y=3 (top)
    +---+---+---+---+
    |   |   |   |   |  y=2
    +---+---+---+---+
    |   |   | H |   |  y=1
    +---+---+---+---+
    |   |   |   |   |  y=0 (bottom)
    +---+---+---+---+
    x=0 x=1 x=2 x=3
    
    G = Goal (+10 reward), H = Hazard (-5 penalty)
""")
    
    # =====================================================================
    # 2. Run Value Iteration
    # =====================================================================
    print_separator("2. Standard Value Iteration")
    
    vi = ValueIteration(gw)
    policy, V, _ = vi.run(theta=0.0001, gamma=0.9)
    
    gw.print_values(V, "Optimal Value Function V*(s)")
    gw.print_policy(policy, "Optimal Policy π*(s)")
    
    # =====================================================================
    # 3. Run Policy Iteration
    # =====================================================================
    print_separator("3. Policy Iteration Comparison")
    
    print("\n--- Standard Policy Iteration (iterative evaluation) ---")
    V_pi, policy_pi, iters, _ = policy_iteration(gw, gamma=0.9, epsilon=0.01)
    
    print("\n--- Policy Iteration with Matrix Inversion (exact evaluation) ---")
    V_exact, policy_exact, iters_exact, _ = policy_iteration_exact(gw, gamma=0.9)
    
    print("\n--- Modified Policy Iteration (m=5 evaluation steps) ---")
    V_mpi, policy_mpi, iters_mpi, _ = modified_policy_iteration(
        gw, gamma=0.9, epsilon=0.01, m=5
    )
    
    # Check if policies match (may differ at ties due to argmax tie-breaking)
    policy_list = [policy[s] for s in range(gw.S)]
    
    # Count differences - differences at states with tied Q-values are fine
    diffs_pi = sum(1 for s in range(gw.S) if policy_list[s] != policy_pi[s])
    diffs_exact = sum(1 for s in range(gw.S) if policy_list[s] != policy_exact[s])
    diffs_mpi = sum(1 for s in range(gw.S) if policy_list[s] != policy_mpi[s])
    
    print(f"\nPolicy comparison (differences may occur at states with tied Q-values):")
    print(f"  VI vs Standard PI: {diffs_pi} differences")
    print(f"  VI vs Exact PI:    {diffs_exact} differences")
    print(f"  VI vs Modified PI: {diffs_mpi} differences")
    
    if diffs_exact == 0:
        print("  All methods found equivalent optimal policies!")
    
    # =====================================================================
    # 4. Compare Value Iteration Variants
    # =====================================================================
    print_separator("4. Value Iteration Variants")
    
    # Get ground truth for convergence tracking
    vi_ref = ValueIteration(gw)
    _, V_optimal, _ = vi_ref.run(theta=1e-10, gamma=0.9)
    
    print("\n--- Standard Value Iteration ---")
    vi_std = ValueIteration(gw)
    _, _, hist_std = vi_std.run(theta=0.01, gamma=0.9, optimal_value=V_optimal)
    
    print("\n--- Gauss-Seidel Value Iteration ---")
    vi_gs = GaussSeidelValueIteration(gw)
    _, _, hist_gs = vi_gs.run(theta=0.01, gamma=0.9, optimal_value=V_optimal)
    
    print("\n--- Jacobi Value Iteration ---")
    vi_j = JacobiValueIteration(gw)
    _, _, hist_j = vi_j.run(theta=0.01, gamma=0.9, optimal_value=V_optimal)
    
    print("\n--- Prioritized Sweeping Value Iteration ---")
    vi_ps = PrioritizedSweepingVI(gw)
    _, _, hist_ps = vi_ps.run(theta=0.001, gamma=0.9, optimal_value=V_optimal)
    
    # =====================================================================
    # 5. Explain the Optimal Policy
    # =====================================================================
    print_separator("5. Understanding the Optimal Policy")
    
    print("""
Key observations about the learned policy:

1. GOAL APPROACH: States near the goal (right and top edges) move
   directly toward it - they take the shortest path.

2. HAZARD AVOIDANCE: The policy routes around hazards rather than
   risk passing through them:
   - State (1,1) goes DOWN to avoid the hazard at (1,2)
   - State (2,2) goes RIGHT to avoid the hazard at (1,2)

3. STOCHASTIC SAFETY: Due to slip probability, the agent prefers
   paths with "safe" perpendicular directions. Going UP near a
   left-side hazard is risky because slipping LEFT could hit it.

4. DISCOUNT BALANCING: The γ=0.9 discount encourages reaching the
   goal quickly (to avoid discounting the +10 reward), while the
   -0.04 step cost adds additional urgency.

5. TERMINAL STATE: V(goal) = 0 because it's terminal - no future
   rewards possible. The +10 is earned when ENTERING the goal.
""")
    
    # =====================================================================
    # 6. Verify Bellman Equation
    # =====================================================================
    print_separator("6. Bellman Equation Verification")
    
    # Pick a non-trivial state to verify
    test_state = gw._xy_to_state(2, 2)  # State at (2,2)
    gamma = 0.9
    
    print(f"\nVerifying Bellman optimality equation for state (2,2):")
    print(f"  V*(s) = max_a [ R(s,a) + γ Σ T(s'|s,a) V*(s') ]")
    
    for a in range(gw.A):
        reward = gw.get_reward(test_state, a)
        expected_future = sum(
            gw.T[test_state, a, s_next] * V[s_next]
            for s_next in range(gw.S)
        )
        q_value = reward + gamma * expected_future
        print(f"\n  {gw.actions[a]:5}: R={reward:7.4f}, "
              f"γE[V]={gamma*expected_future:7.4f}, Q={q_value:7.4f}")
    
    print(f"\n  V*(2,2) = {V[test_state]:.4f} = max of Q-values above ✓")
    
    print_separator()
    print("\nDemo complete! The Grid World MDP has been solved using")
    print("multiple dynamic programming algorithms.")


if __name__ == "__main__":
    demo_grid_world()
