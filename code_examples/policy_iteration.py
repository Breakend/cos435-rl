"""
Policy Iteration Algorithms for MDPs

This module implements Policy Iteration variants for solving MDPs:

1. Standard Policy Iteration - Iterative policy evaluation
2. Policy Iteration by Matrix Inversion - Exact evaluation via linear algebra
3. Modified Policy Iteration - Limited evaluation steps (hybrid VI/PI)

Policy Iteration alternates between:
    - Policy Evaluation: Compute V^π for current policy π
    - Policy Improvement: Update π to be greedy w.r.t. V^π

References:
    - Sutton & Barto, "Reinforcement Learning: An Introduction", Ch. 4
    - Puterman, "Markov Decision Processes", Ch. 6
"""

import numpy as np
from typing import List, Tuple, Optional
from mdp import MDP


def policy_iteration(
    mdp: MDP,
    gamma: float = 0.9,
    epsilon: float = 0.01,
    optimal_value: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[int], int, List[float]]:
    """
    Standard Policy Iteration algorithm.
    
    Alternates between:
    1. Policy Evaluation: Iteratively compute V^π until convergence
    2. Policy Improvement: Make policy greedy w.r.t. V^π
    
    Terminates when the policy is stable (no changes during improvement).
    
    Args:
        mdp: MDP object with S, A, T, R attributes
        gamma: Discount factor (0 < gamma < 1)
        epsilon: Convergence threshold for policy evaluation
        optimal_value: Optional ground truth V* for tracking convergence
        
    Returns:
        V: Value function of the optimal policy
        policy: Optimal policy as list of actions
        num_iterations: Number of policy improvement iterations
        convergence_history: List of ||V - V*|| at each evaluation step
    
    Example:
        >>> V, policy, iters, history = policy_iteration(gridworld, gamma=0.9)
        >>> print(f"Converged in {iters} policy improvements")
    """
    # Initialize
    V = np.zeros(mdp.S)
    policy = [0] * mdp.S  # Start with action 0 everywhere
    convergence_history: List[float] = []
    num_iterations = 0
    num_evaluations = 0
    
    while True:
        # === Policy Evaluation ===
        # Compute V^π by iterating: V(s) = R(s,π(s)) + γ Σ T(s'|s,π(s)) V(s')
        while True:
            delta = 0.0
            for s in range(mdp.S):
                v_old = V[s]
                
                # Get reward for current policy action
                reward = mdp.get_reward(s, policy[s])
                
                # Compute expected future value under current policy
                expected_future = sum(
                    mdp.T[s, policy[s], s_next] * V[s_next]
                    for s_next in range(mdp.S)
                )
                
                V[s] = reward + gamma * expected_future
                delta = max(delta, abs(v_old - V[s]))
                
                # Track convergence
                if optimal_value is not None:
                    convergence_history.append(np.linalg.norm(V - optimal_value))
                num_evaluations += 1
            
            if delta < epsilon:
                break
        
        # === Policy Improvement ===
        # Update policy to be greedy: π(s) = argmax_a Q(s,a)
        policy_stable = True
        
        for s in range(mdp.S):
            old_action = policy[s]
            
            # Compute Q(s,a) for all actions
            q_values = []
            for a in range(mdp.A):
                reward = mdp.get_reward(s, a)
                expected_future = sum(
                    mdp.T[s, a, s_next] * V[s_next]
                    for s_next in range(mdp.S)
                )
                q_values.append(reward + gamma * expected_future)
            
            # Select best action
            policy[s] = int(np.argmax(q_values))
            
            if old_action != policy[s]:
                policy_stable = False
        
        num_iterations += 1
        
        # Terminate if policy unchanged
        if policy_stable:
            print(f"Policy Iteration converged in {num_iterations} iterations "
                  f"({num_evaluations} evaluation sweeps)")
            return V, policy, num_iterations, convergence_history


def policy_iteration_exact(
    mdp: MDP,
    gamma: float = 0.9,
    optimal_value: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[int], int, List[float]]:
    """
    Policy Iteration with exact policy evaluation via matrix inversion.
    
    Instead of iterating to evaluate V^π, we solve the linear system:
        V^π = R^π + γ T^π V^π
        V^π = (I - γ T^π)^{-1} R^π
    
    where T^π[s,s'] = T(s'|s,π(s)) and R^π[s] = R(s,π(s)).
    
    This requires O(S³) computation for the matrix inverse but gives
    exact values in one step rather than iterating to convergence.
    
    Args:
        mdp: MDP object
        gamma: Discount factor
        optimal_value: Optional ground truth for convergence tracking
        
    Returns:
        V: Optimal value function
        policy: Optimal policy
        num_iterations: Number of policy improvement iterations
        convergence_history: List of ||V - V*|| after each evaluation
    """
    V = np.zeros(mdp.S)
    policy = [0] * mdp.S
    convergence_history: List[float] = []
    num_iterations = 0
    
    while True:
        # === Exact Policy Evaluation via Matrix Inversion ===
        # Build T^π: transition matrix under current policy
        T_policy = np.zeros((mdp.S, mdp.S))
        R_policy = np.zeros(mdp.S)
        
        for s in range(mdp.S):
            T_policy[s, :] = mdp.T[s, policy[s], :]
            R_policy[s] = mdp.get_reward(s, policy[s])
        
        # Solve: V = (I - γT)^{-1} R
        identity = np.eye(mdp.S)
        V = np.linalg.solve(identity - gamma * T_policy, R_policy)
        
        if optimal_value is not None:
            convergence_history.append(np.linalg.norm(V - optimal_value))
        
        # === Policy Improvement ===
        policy_stable = True
        
        for s in range(mdp.S):
            old_action = policy[s]
            
            q_values = []
            for a in range(mdp.A):
                reward = mdp.get_reward(s, a)
                expected_future = sum(
                    mdp.T[s, a, s_next] * V[s_next]
                    for s_next in range(mdp.S)
                )
                q_values.append(reward + gamma * expected_future)
            
            policy[s] = int(np.argmax(q_values))
            
            if old_action != policy[s]:
                policy_stable = False
        
        num_iterations += 1
        
        if policy_stable:
            print(f"Policy Iteration (exact) converged in {num_iterations} iterations")
            return V, policy, num_iterations, convergence_history


def modified_policy_iteration(
    mdp: MDP,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    m: int = 10,
    optimal_value: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[int], int, List[float]]:
    """
    Modified Policy Iteration with limited evaluation steps.
    
    A hybrid between Value Iteration and Policy Iteration:
    - Like PI: alternates evaluation and improvement
    - Like VI: limits evaluation to m steps (doesn't wait for convergence)
    
    When m=1, this is equivalent to Value Iteration.
    When m=∞, this is equivalent to Policy Iteration.
    
    Args:
        mdp: MDP object
        gamma: Discount factor
        epsilon: Convergence threshold for outer loop
        m: Maximum number of policy evaluation sweeps per iteration
        optimal_value: Optional ground truth for convergence tracking
        
    Returns:
        V: Value function
        policy: Policy
        num_iterations: Number of outer loop iterations
        convergence_history: Convergence tracking data
    """
    V = np.zeros(mdp.S)
    policy = [0] * mdp.S
    convergence_history: List[float] = []
    num_iterations = 0
    num_evaluations = 0
    
    while True:
        # === Policy Improvement (first) ===
        policy_stable = True
        max_change = 0.0
        
        for s in range(mdp.S):
            old_action = policy[s]
            
            q_values = []
            for a in range(mdp.A):
                reward = mdp.get_reward(s, a)
                expected_future = sum(
                    mdp.T[s, a, s_next] * V[s_next]
                    for s_next in range(mdp.S)
                )
                q_values.append(reward + gamma * expected_future)
            
            best_q = max(q_values)
            policy[s] = int(np.argmax(q_values))
            max_change = max(max_change, abs(V[s] - best_q))
            
            if old_action != policy[s]:
                policy_stable = False
        
        # Check for convergence
        if max_change < epsilon:
            print(f"Modified PI converged in {num_iterations} iterations "
                  f"({num_evaluations} evaluation sweeps)")
            return V, policy, num_iterations, convergence_history
        
        # === Limited Policy Evaluation ===
        for _ in range(m):
            delta = 0.0
            for s in range(mdp.S):
                v_old = V[s]
                
                reward = mdp.get_reward(s, policy[s])
                expected_future = sum(
                    mdp.T[s, policy[s], s_next] * V[s_next]
                    for s_next in range(mdp.S)
                )
                
                V[s] = reward + gamma * expected_future
                delta = max(delta, abs(v_old - V[s]))
                
                if optimal_value is not None:
                    convergence_history.append(np.linalg.norm(V - optimal_value))
                num_evaluations += 1
            
            # Early termination if evaluation converged
            if delta < epsilon / (2 * gamma):
                break
        
        num_iterations += 1
