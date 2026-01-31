"""
Value Iteration Algorithms for MDPs

This module implements several variants of the Value Iteration algorithm
for solving Markov Decision Processes:

1. Standard Value Iteration - Classic Bellman backup approach
2. Gauss-Seidel Value Iteration - Uses updated values immediately
3. Jacobi Value Iteration - Handles self-loops explicitly  
4. Prioritized Sweeping - Updates states by priority of change

References:
    - Sutton & Barto, "Reinforcement Learning: An Introduction", Ch. 4
    - Puterman, "Markov Decision Processes", Ch. 6
    - Shlakhter, "Prioritized Value Iteration" (PhD thesis, U. Toronto)
"""

import numpy as np
import heapq
from typing import Dict, List, Tuple, Optional
from mdp import MDP


class ValueIteration:
    """
    Standard Value Iteration algorithm.
    
    Value Iteration finds the optimal value function V* by iteratively
    applying the Bellman optimality operator:
    
        V_{k+1}(s) = max_a [ R(s,a) + γ Σ_{s'} T(s'|s,a) V_k(s') ]
    
    The algorithm converges when ||V_{k+1} - V_k||_∞ < θ.
    
    Attributes:
        mdp: The MDP to solve
        gauss_seidel: If True, use updated values immediately within a sweep
    
    Example:
        >>> vi = ValueIteration(gridworld)
        >>> policy, values, history = vi.run(gamma=0.9, theta=0.001)
    """
    
    def __init__(self, mdp: MDP, gauss_seidel: bool = False):
        """
        Initialize Value Iteration.
        
        Args:
            mdp: MDP object with S, A, T, R attributes
            gauss_seidel: If True, use Gauss-Seidel updates (in-place)
        """
        self.mdp = mdp
        self.gauss_seidel = gauss_seidel
    
    def run(
        self,
        theta: float = 0.001,
        gamma: float = 0.9,
        optimal_value: Optional[np.ndarray] = None
    ) -> Tuple[Dict[int, int], np.ndarray, List[float]]:
        """
        Run Value Iteration until convergence.
        
        Args:
            theta: Convergence threshold - stop when max change < theta
            gamma: Discount factor (0 < gamma <= 1)
            optimal_value: Optional ground truth V* for tracking convergence
            
        Returns:
            policy: Optimal policy as dict {state: action}
            V: Optimal value function as array
            convergence_history: List of ||V - V*|| at each state update
                                (empty if optimal_value not provided)
        """
        V = np.zeros(self.mdp.S)
        convergence_history: List[float] = []
        num_sweeps = 0
        
        while True:
            delta = 0.0
            
            # Choose update strategy
            if self.gauss_seidel:
                # Gauss-Seidel: use updated values immediately
                V_old = V
            else:
                # Standard: use values from previous sweep
                V_old = V.copy()
            
            # Sweep through all states
            for s in range(self.mdp.S):
                # Track convergence if ground truth provided
                if optimal_value is not None:
                    convergence_history.append(np.linalg.norm(V - optimal_value))
                
                v_old = V_old[s]
                
                # Bellman optimality update: V(s) = max_a Q(s,a)
                q_values = []
                for a in range(self.mdp.A):
                    reward = self.mdp.get_reward(s, a)
                    expected_future = sum(
                        self.mdp.T[s, a, s_next] * V_old[s_next]
                        for s_next in range(self.mdp.S)
                    )
                    q_values.append(reward + gamma * expected_future)
                
                V[s] = max(q_values)
                delta = max(delta, abs(v_old - V[s]))
            
            num_sweeps += 1
            
            # Check convergence
            if delta < theta:
                break
        
        print(f"Value Iteration converged in {num_sweeps} sweeps")
        
        # Extract greedy policy from value function
        policy = self._extract_policy(V, gamma)
        return policy, V, convergence_history
    
    def _extract_policy(self, V: np.ndarray, gamma: float) -> Dict[int, int]:
        """
        Extract greedy policy from value function.
        
        π(s) = argmax_a [ R(s,a) + γ Σ_{s'} T(s'|s,a) V(s') ]
        
        Args:
            V: Value function
            gamma: Discount factor
            
        Returns:
            policy: Dictionary mapping state to best action
        """
        policy: Dict[int, int] = {}
        
        for s in range(self.mdp.S):
            q_values = []
            for a in range(self.mdp.A):
                reward = self.mdp.get_reward(s, a)
                expected_future = sum(
                    self.mdp.T[s, a, s_next] * V[s_next]
                    for s_next in range(self.mdp.S)
                )
                q_values.append(reward + gamma * expected_future)
            
            # Select action with highest Q-value
            policy[s] = int(np.argmax(q_values))
        
        return policy


class GaussSeidelValueIteration(ValueIteration):
    """
    Gauss-Seidel Value Iteration.
    
    A variant that uses updated values immediately within a sweep,
    rather than waiting for the sweep to complete. This typically
    converges faster than standard VI in practice.
    
    The update for state s uses the most recent values:
        V(s) = max_a [ R(s,a) + γ Σ_{s'<s} T(s'|s,a) V_{k+1}(s') 
                              + γ Σ_{s'≥s} T(s'|s,a) V_k(s') ]
    """
    
    def __init__(self, mdp: MDP):
        super().__init__(mdp, gauss_seidel=True)


class JacobiValueIteration(ValueIteration):
    """
    Jacobi Value Iteration with self-loop handling.
    
    This variant explicitly handles self-loops in the transition dynamics.
    When a state has positive probability of transitioning to itself
    (T[s,a,s] > 0), the standard update can be rewritten:
    
        V(s) = max_a [ (R(s,a) + γ Σ_{s'≠s} T(s'|s,a) V(s')) / (1 - γ T(s|s,a)) ]
    
    This can improve convergence for MDPs with significant self-loops.
    """
    
    def run(
        self,
        theta: float = 0.01,
        gamma: float = 0.9,
        optimal_value: Optional[np.ndarray] = None
    ) -> Tuple[Dict[int, int], np.ndarray, List[float]]:
        """
        Run Jacobi Value Iteration with self-loop handling.
        """
        V = np.zeros(self.mdp.S)
        convergence_history: List[float] = []
        num_sweeps = 0
        
        while True:
            delta = 0.0
            
            if self.gauss_seidel:
                V_old = V
            else:
                V_old = V.copy()
            
            for s in range(self.mdp.S):
                if optimal_value is not None:
                    convergence_history.append(np.linalg.norm(V - optimal_value))
                
                v_old = V_old[s]
                
                q_values = []
                for a in range(self.mdp.A):
                    reward = self.mdp.get_reward(s, a)
                    
                    # Sum over non-self transitions
                    sum_others = sum(
                        self.mdp.T[s, a, s_next] * V_old[s_next]
                        for s_next in range(self.mdp.S)
                        if s_next != s
                    )
                    
                    # Handle self-loop: solve V(s) = R + γ*T(s|s,a)*V(s) + γ*sum_others
                    # => V(s) * (1 - γ*T(s|s,a)) = R + γ*sum_others
                    self_loop_prob = self.mdp.T[s, a, s]
                    denominator = 1.0 - gamma * self_loop_prob
                    
                    if denominator > 1e-10:
                        q_values.append((reward + gamma * sum_others) / denominator)
                    else:
                        # Fallback if denominator is too small
                        q_values.append(reward + gamma * sum_others)
                
                V[s] = max(q_values)
                delta = max(delta, abs(v_old - V[s]))
            
            num_sweeps += 1
            
            if delta < theta:
                break
        
        print(f"Jacobi VI converged in {num_sweeps} sweeps")
        policy = self._extract_policy(V, gamma)
        return policy, V, convergence_history


class PrioritizedSweepingVI(ValueIteration):
    """
    Value Iteration with Prioritized Sweeping.
    
    Instead of sweeping through states in order, this algorithm maintains
    a priority queue and updates states based on their expected change
    (Bellman error). States whose predecessors have large updates are
    prioritized.
    
    This can be much more efficient for sparse MDPs where only a subset
    of states need frequent updates.
    
    Reference:
        Moore & Atkeson, "Prioritized Sweeping: Reinforcement Learning
        with Less Data and Less Time", Machine Learning, 1993.
    """
    
    def run(
        self,
        theta: float = 0.0001,
        gamma: float = 0.9,
        max_iterations: int = 5000,
        optimal_value: Optional[np.ndarray] = None
    ) -> Tuple[Dict[int, int], np.ndarray, List[float]]:
        """
        Run Prioritized Sweeping Value Iteration.
        
        Args:
            theta: Priority threshold for updates
            gamma: Discount factor
            max_iterations: Maximum number of state updates
            optimal_value: Optional ground truth for convergence tracking
        """
        V = np.zeros(self.mdp.S)
        convergence_history: List[float] = []
        
        # Build predecessor graph: predecessors[s'] = {states that can reach s'}
        predecessors: Dict[int, set] = {s: set() for s in range(self.mdp.S)}
        for s in range(self.mdp.S):
            for a in range(self.mdp.A):
                for s_next in range(self.mdp.S):
                    if self.mdp.T[s, a, s_next] > 0:
                        predecessors[s_next].add(s)
        
        # Initialize priority queue with Bellman errors
        # Priority queue entries: (-priority, state) [negated for min-heap]
        pq: List[Tuple[float, int]] = []
        in_queue: set = set()
        
        for s in range(self.mdp.S):
            bellman_error = self._compute_bellman_error(s, V, gamma)
            if bellman_error > theta:
                heapq.heappush(pq, (-bellman_error, s))
                in_queue.add(s)
        
        # Main loop
        for iteration in range(max_iterations):
            if not pq:
                break
            
            if optimal_value is not None:
                convergence_history.append(np.linalg.norm(V - optimal_value))
            
            # Pop highest priority state
            _, s = heapq.heappop(pq)
            in_queue.discard(s)
            
            # Update V[s]
            V[s] = self._compute_max_q(s, V, gamma)
            
            # Update priorities of predecessors
            for pred in predecessors[s]:
                bellman_error = self._compute_bellman_error(pred, V, gamma)
                if bellman_error > theta and pred not in in_queue:
                    heapq.heappush(pq, (-bellman_error, pred))
                    in_queue.add(pred)
        
        print(f"Prioritized Sweeping converged in {iteration + 1} updates")
        policy = self._extract_policy(V, gamma)
        return policy, V, convergence_history
    
    def _compute_bellman_error(self, s: int, V: np.ndarray, gamma: float) -> float:
        """Compute |V(s) - max_a Q(s,a)|."""
        max_q = self._compute_max_q(s, V, gamma)
        return abs(V[s] - max_q)
    
    def _compute_max_q(self, s: int, V: np.ndarray, gamma: float) -> float:
        """Compute max_a Q(s,a) for state s."""
        q_values = []
        for a in range(self.mdp.A):
            reward = self.mdp.get_reward(s, a)
            expected_future = sum(
                self.mdp.T[s, a, s_next] * V[s_next]
                for s_next in range(self.mdp.S)
            )
            q_values.append(reward + gamma * expected_future)
        return max(q_values)
