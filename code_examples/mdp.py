"""
Markov Decision Process (MDP) Definitions

This module provides base classes for representing MDPs and a concrete
implementation of a Grid World environment commonly used in reinforcement
learning education.

Classes:
    MDP: Base class for Markov Decision Processes
    GridWorld: Stochastic grid world with configurable rewards and dynamics
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class MDP:
    """
    Base class for Markov Decision Process representation.
    
    An MDP is defined by the tuple (S, A, T, R, γ) where:
        - S: State space (finite set of states)
        - A: Action space (finite set of actions)
        - T: Transition function T(s'|s,a) - probability of reaching s' from s via a
        - R: Reward function R(s,a) or R(s) - expected reward
        - γ: Discount factor (not stored here, used by algorithms)
    
    Attributes:
        S (int): Number of states
        A (int): Number of actions
        T (np.ndarray): Transition probabilities, shape (S, A, S) 
                        where T[s, a, s'] = P(s' | s, a)
        R (np.ndarray): Reward function, shape (S,) for R(s) or (S, A) for R(s,a)
        actions (List[str]): Human-readable action names
    
    Example:
        >>> mdp = MDP(T=transitions, S=16, R=rewards, A=4, act_list=['UP', 'DOWN', 'LEFT', 'RIGHT'])
        >>> print(f"MDP with {mdp.S} states and {mdp.A} actions")
    """
    
    def __init__(
        self,
        T: np.ndarray,
        S: int,
        R: np.ndarray,
        A: int,
        act_list: List[str]
    ):
        """
        Initialize an MDP.
        
        Args:
            T: Transition probability matrix, shape (S, A, S)
            S: Number of states
            R: Reward vector/matrix, shape (S,) or (S, A)
            A: Number of actions
            act_list: List of action names for display purposes
        """
        self.S = S
        self.A = A
        self.T = np.array(T)
        self.R = np.array(R)
        self.actions = act_list
        
        # Validate dimensions
        assert self.T.shape == (S, A, S), f"T should be ({S}, {A}, {S}), got {self.T.shape}"
        assert self.R.shape in [(S,), (S, A)], f"R should be ({S},) or ({S}, {A}), got {self.R.shape}"
    
    def get_reward(self, state: int, action: int) -> float:
        """
        Get the reward for a state-action pair.
        
        Handles both R(s) and R(s,a) reward formulations.
        
        Args:
            state: Current state index
            action: Action index
            
        Returns:
            Expected immediate reward
        """
        if self.R.ndim == 1:
            return self.R[state]
        return self.R[state, action]


class GridWorld(MDP):
    """
    Grid World MDP with stochastic dynamics.
    
    A classic RL environment where an agent navigates a grid to reach a goal
    while avoiding hazards. The dynamics are stochastic: the agent may "slip"
    and move perpendicular to the intended direction.
    
    Coordinate System:
        - (x, y) where x is column (0 = left), y is row (0 = bottom)
        - State index: s = y * grid_size + x (row-major from bottom-left)
        - Visual display shows y=max at top (standard grid orientation)
    
    Actions:
        - 0: UP    (y + 1)
        - 1: DOWN  (y - 1)
        - 2: LEFT  (x - 1)
        - 3: RIGHT (x + 1)
    
    Dynamics:
        When taking an action:
        - With probability (1 - 2*slip_prob): move in intended direction
        - With probability slip_prob: slip to each perpendicular direction
        - Hitting a wall means staying in place
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        goal_pos: (x, y) position of goal state
        goal_reward: Reward for reaching goal
        hazard_positions: List of (x, y) positions of hazard states
        hazard_reward: Reward (typically negative) for hazard states
        step_cost: Cost per step (typically small negative value)
        slip_prob: Probability of slipping to each perpendicular direction
        goal_terminal: If True, goal is absorbing (episode ends there)
    
    Example:
        >>> gw = GridWorld(
        ...     grid_size=4,
        ...     goal_pos=(3, 3),
        ...     hazard_positions=[(1, 2)],
        ...     slip_prob=0.1
        ... )
        >>> print(f"Grid world with {gw.S} states")
        Grid world with 16 states
    """
    
    # Action indices
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    def __init__(
        self,
        grid_size: int = 4,
        goal_pos: Tuple[int, int] = (3, 3),
        goal_reward: float = 10.0,
        hazard_positions: Optional[List[Tuple[int, int]]] = None,
        hazard_reward: float = -5.0,
        step_cost: float = -0.04,
        slip_prob: float = 0.1,
        goal_terminal: bool = True
    ):
        self.grid_size = grid_size
        S = grid_size * grid_size
        A = 4
        
        act_list = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # Direction deltas: (dx, dy) for each action
        self.action_deltas: Dict[int, Tuple[int, int]] = {
            self.UP:    (0, 1),    # y increases (move up)
            self.DOWN:  (0, -1),   # y decreases (move down)
            self.LEFT:  (-1, 0),   # x decreases (move left)
            self.RIGHT: (1, 0),    # x increases (move right)
        }
        
        # Perpendicular actions for slip mechanics
        # UP/DOWN can slip LEFT/RIGHT, LEFT/RIGHT can slip UP/DOWN
        self.perpendicular: Dict[int, List[int]] = {
            self.UP:    [self.LEFT, self.RIGHT],
            self.DOWN:  [self.LEFT, self.RIGHT],
            self.LEFT:  [self.UP, self.DOWN],
            self.RIGHT: [self.UP, self.DOWN],
        }
        
        if hazard_positions is None:
            hazard_positions = []
        
        # Build reward function R[s, a] and transition matrix T[s, a, s']
        R = np.zeros((S, A))
        T = np.zeros((S, A, S))
        
        main_prob = 1.0 - 2 * slip_prob  # e.g., 80% if slip_prob = 0.1
        goal_state = goal_pos[1] * grid_size + goal_pos[0]
        
        for s in range(S):
            x, y = self._state_to_xy(s)
            
            # Terminal state handling: goal state is absorbing
            if goal_terminal and (x, y) == goal_pos:
                for a in range(A):
                    T[s, a, s] = 1.0  # Self-loop with probability 1
                    R[s, a] = 0.0     # No rewards in terminal state
                continue
            
            for a in range(A):
                # Calculate transition probabilities
                probs: Dict[int, float] = {}
                
                # Main direction
                next_s_main = self._get_next_state(x, y, a)
                probs[next_s_main] = probs.get(next_s_main, 0) + main_prob
                
                # Perpendicular slip directions
                for perp_a in self.perpendicular[a]:
                    next_s_perp = self._get_next_state(x, y, perp_a)
                    probs[next_s_perp] = probs.get(next_s_perp, 0) + slip_prob
                
                # Fill transition matrix
                for next_s, prob in probs.items():
                    T[s, a, next_s] = prob
                
                # Calculate expected reward: R(s,a) = Σ T(s'|s,a) × [step_cost + bonus(s')]
                expected_reward = 0.0
                for next_s, prob in probs.items():
                    nx, ny = self._state_to_xy(next_s)
                    state_bonus = 0.0
                    if (nx, ny) == goal_pos:
                        state_bonus = goal_reward
                    elif (nx, ny) in hazard_positions:
                        state_bonus = hazard_reward
                    expected_reward += prob * (step_cost + state_bonus)
                
                R[s, a] = expected_reward
        
        # Store configuration for visualization
        self.goal_pos = goal_pos
        self.hazard_positions = hazard_positions
        self.goal_state = goal_state
        self.hazard_states = [self._xy_to_state(*h) for h in hazard_positions]
        self.goal_terminal = goal_terminal
        
        super().__init__(T, S, R, A, act_list)
    
    def _xy_to_state(self, x: int, y: int) -> int:
        """Convert (x, y) coordinates to state index."""
        return y * self.grid_size + x
    
    def _state_to_xy(self, s: int) -> Tuple[int, int]:
        """Convert state index to (x, y) coordinates."""
        y = s // self.grid_size
        x = s % self.grid_size
        return x, y
    
    def _get_next_state(self, x: int, y: int, action: int) -> int:
        """
        Get next state after taking an action.
        
        Handles wall collisions by keeping agent in place.
        """
        dx, dy = self.action_deltas[action]
        nx, ny = x + dx, y + dy
        
        # Wall collision: stay in place
        if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
            nx, ny = x, y
        
        return self._xy_to_state(nx, ny)
    
    def print_values(self, V: np.ndarray, title: str = "Value Function") -> None:
        """
        Pretty print the value function as a grid.
        
        Args:
            V: Value function array of shape (S,)
            title: Title to display above the grid
        """
        print(f"\n{title}:")
        print("-" * (self.grid_size * 8 + 1))
        
        # Print from top row (y = grid_size-1) to bottom (y = 0)
        for y in range(self.grid_size - 1, -1, -1):
            row = "|"
            for x in range(self.grid_size):
                s = self._xy_to_state(x, y)
                row += f" {V[s]:6.2f}|"
            print(row)
            print("-" * (self.grid_size * 8 + 1))
    
    def print_policy(self, policy: Dict[int, int], title: str = "Policy") -> None:
        """
        Pretty print the policy as a grid with arrows.
        
        Args:
            policy: Dictionary or list mapping state -> action
            title: Title to display above the grid
        """
        arrows = {
            self.UP: '↑',
            self.DOWN: '↓',
            self.LEFT: '←',
            self.RIGHT: '→'
        }
        
        print(f"\n{title}:")
        print("-" * (self.grid_size * 4 + 1))
        
        for y in range(self.grid_size - 1, -1, -1):
            row = "|"
            for x in range(self.grid_size):
                s = self._xy_to_state(x, y)
                if (x, y) == self.goal_pos:
                    row += " G |"
                elif (x, y) in self.hazard_positions:
                    row += " H |"
                else:
                    action = policy[s] if isinstance(policy, dict) else policy[s]
                    row += f" {arrows[action]} |"
            print(row)
            print("-" * (self.grid_size * 4 + 1))
        
        print("\nLegend: G=Goal, H=Hazard, ↑↓←→=Actions")
