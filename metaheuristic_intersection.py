import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from dataclasses import dataclass

# ============================================================================
# PROBLEM PARAMETERS (Based on Literature 2018-2025)
# ============================================================================

@dataclass
class ProblemParameters:
    """
    Parameters for CAV intersection optimization
    """
    # Number of vehicles
    N: int = 4                  # Total vehicles
    N_EW: int = 1              # East to West
    N_WE: int = 1              # West to East
    N_NS: int = 1              # North to South
    N_SN: int = 1              # South to North

    # Time discretization - EXTENDED to remove artificial time constraint
    dt: float = 0.5            # Time step (seconds) - [0.1-0.5 typical]
    K: int = 100               # Number of time steps (INCREASED from 30)
    T_max: float = 50.0        # Time horizon (seconds) (INCREASED from 15)

    # NOTE: T_max is now an upper bound for simulation, not a hard constraint.
    # The objective function measures actual crossing time, which can be much less.
    # Time minimization in objective naturally pushes for fast crossings (~10-20s).
    # Extended horizon ensures solution space completeness without artificial limits.

    # Acceleration limits (m/s²)
    u_min: float = -4.0        # Max deceleration (AASHTO standard)
    u_max: float = 3.0         # Max acceleration (typical vehicle)

    # Velocity limits (m/s)
    v_min: float = 5.0         # Min velocity in intersection (no stopping)
    v_max: float = 20.0        # Max velocity
    v_min_approach: float = 0.0  # Min velocity before intersection

    # Intersection geometry (meters)
    L: float = 150.0           # Control zone length (approach distance)
    S: float = 12.0            # Merging/conflict zone size
    delta: float = 5.0         # Minimum separation distance (point mass)

    # Safety parameters
    dt_safe: float = 2.0       # Minimum time separation (seconds)
    M: float = 500.0           # Big-M constant for MILP formulation

    # Objective function weights
    alpha: float = 0.5         # Time weight (0 to 1)
    # Note: Energy weight is (1 - alpha)

    def __post_init__(self):
        """Validate parameters after initialization"""
        # Check vehicle count consistency
        if self.N_EW + self.N_WE + self.N_NS + self.N_SN != self.N:
            raise ValueError(
                f"N_EW ({self.N_EW}) + N_WE ({self.N_WE}) + N_NS ({self.N_NS}) + N_SN ({self.N_SN}) must equal N ({self.N})"
            )

        # Check time discretization consistency
        if abs(self.dt * self.K - self.T_max) > 1e-6:
            print(f"Warning: dt×K ({self.dt * self.K:.2f}s) ≠ T_max ({self.T_max}s)")
            print(f"         Using dt×K = {self.dt * self.K:.2f}s as effective time horizon")

        # Check velocity limits are sensible
        if self.v_min > self.v_max:
            raise ValueError(
                f"v_min ({self.v_min}) must be less than v_max ({self.v_max})"
            )

        if self.v_min_approach > self.v_min:
            raise ValueError(
                f"v_min_approach ({self.v_min_approach}) should not exceed v_min ({self.v_min})"
            )

        # Check acceleration limits
        if self.u_min > self.u_max:
            raise ValueError(
                f"u_min ({self.u_min}) must be less than u_max ({self.u_max})"
            )

        # Check alpha weight is valid
        if not (0 <= self.alpha <= 1):
            raise ValueError(
                f"alpha ({self.alpha}) must be between 0 and 1"
            )

        # Check geometric parameters
        if self.S > self.L:
            raise ValueError(
                f"Merging zone size S ({self.S}) should not exceed control zone length L ({self.L})"
            )

        if self.delta <= 0:
            raise ValueError(
                f"Safety distance delta ({self.delta}) must be positive"
            )

    def get_conflict_matrix(self) -> np.ndarray:
        """
        Conflict matrix for 4-direction intersection

        Vehicle indices:
        0: E→W, 1: W→E, 2: N→S, 3: S→N

        Conflicts (perpendicular crossings):
        - E→W (0) conflicts with N→S (2) and S→N (3)
        - W→E (1) conflicts with N→S (2) and S→N (3)
        - N→S (2) conflicts with E→W (0) and W→E (1)
        - S→N (3) conflicts with E→W (0) and W→E (1)
        """
        C = np.zeros((self.N, self.N), dtype=int)

        # Horizontal vehicles (E→W, W→E) conflict with vertical (N→S, S→N)
        horizontal_vehicles = list(range(self.N_EW + self.N_WE))  # [0, 1]
        vertical_vehicles = list(range(self.N_EW + self.N_WE, self.N))  # [2, 3]

        for h in horizontal_vehicles:
            for v in vertical_vehicles:
                C[h, v] = 1
                C[v, h] = 1

        return C

    def get_vehicle_direction(self, vehicle_id: int) -> str:
        """
        Get direction string for a vehicle

        Returns: 'EW', 'WE', 'NS', or 'SN'
        """
        if vehicle_id < self.N_EW:
            return 'EW'
        elif vehicle_id < self.N_EW + self.N_WE:
            return 'WE'
        elif vehicle_id < self.N_EW + self.N_WE + self.N_NS:
            return 'NS'
        else:
            return 'SN'

    def vehicles_in_same_lane(self, i: int, j: int) -> bool:
        """
        Check if two vehicles are in the exact same lane (same direction)
        Used for rear-end collision checking
        """
        dir_i = self.get_vehicle_direction(i)
        dir_j = self.get_vehicle_direction(j)
        return dir_i == dir_j

# ============================================================================
# VEHICLE DYNAMICS SIMULATION
# ============================================================================

def simulate_vehicle_trajectory(u: np.ndarray, x0: float, v0: float, 
                                dt: float, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate vehicle trajectory using discrete-time dynamics
    
    Dynamics (Constraint 1):
        x[k+1] = x[k] + v[k]·dt + 0.5·u[k]·dt²
        v[k+1] = v[k] + u[k]·dt
    
    Args:
        u: Acceleration profile (K,) array
        x0: Initial position (m)
        v0: Initial velocity (m/s)
        dt: Time step (s)
        K: Number of time steps
        
    Returns:
        x: Position trajectory (K+1,) - positions at k=0,1,...,K
        v: Velocity trajectory (K+1,) - velocities at k=0,1,...,K
    """
    x = np.zeros(K + 1)
    v = np.zeros(K + 1)
    
    # Initial conditions (Constraint 2)
    x[0] = x0
    v[0] = v0
    
    # Forward integration
    for k in range(K):
        v[k+1] = v[k] + u[k] * dt
        x[k+1] = x[k] + v[k] * dt + 0.5 * u[k] * dt**2
    
    return x, v

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def objective_function(x_decision: np.ndarray,
                       x0: np.ndarray,
                       v0: np.ndarray,
                       t0: np.ndarray,
                       params: ProblemParameters) -> Tuple[float, float, float, Dict]:
    """
    Compute objective function value

    UPDATED: With extended time horizon (K=100, T_max=50s), vehicles have ample time.
    The time minimization component naturally incentivizes rapid crossings.
    We detect actual exit times and compute energy only up to exit (not full horizon).

    Objective: minimize α·Σᵢ(tᵢ - tᵢ⁰) + (1-α)·Σᵢ Σₖ uᵢ²[k]·Δt

    Decision variables:
        x_decision = [u₁[0], u₁[1], ..., u₁[K-1],    # Vehicle 1 accelerations
                      u₂[0], u₂[1], ..., u₂[K-1],    # Vehicle 2 accelerations
                      ...
                      uₙ[0], uₙ[1], ..., uₙ[K-1],    # Vehicle N accelerations
                      Z₁₂, Z₁₃, ..., Zᵢⱼ]              # Binary priority variables

    Args:
        x_decision: Decision vector (N*K + n_conflicts,)
        x0: Initial positions (N,)
        v0: Initial velocities (N,)
        t0: Initial times (N,)
        params: Problem parameters

    Returns:
        f_total: Total objective value
        f_time: Time component
        f_energy: Energy component
        info: Dictionary with detailed information
    """
    N = params.N
    K = params.K
    dt = params.dt

    # Extract acceleration profiles from decision vector
    u_profiles = x_decision[:N*K].reshape(N, K)

    # Initialize metrics
    travel_times = np.zeros(N)
    energies = np.zeros(N)
    trajectories = []
    actual_K_needed = 0  # Track maximum steps actually needed

    # Simulate each vehicle and compute metrics
    for i in range(N):
        u_i = u_profiles[i, :]
        x_traj, v_traj = simulate_vehicle_trajectory(u_i, x0[i], v0[i], dt, K)
        trajectories.append((x_traj, v_traj))

        # Find time when vehicle exits control zone (x >= L)
        exit_indices = np.where(x_traj >= params.L)[0]
        if len(exit_indices) > 0:
            k_exit = exit_indices[0]
            t_exit = t0[i] + k_exit * dt
            actual_K_needed = max(actual_K_needed, k_exit)
        else:
            # Vehicle doesn't reach exit - this will be caught by constraint checker
            # Use full time horizon as penalty
            k_exit = K - 1
            t_exit = t0[i] + K * dt
            actual_K_needed = K

        # Travel time: time to exit - initial time
        travel_times[i] = t_exit - t0[i]

        # Energy: integral of acceleration squared
        # Only count energy up to exit time (not full K)
        # This prevents penalizing unused time steps
        energies[i] = np.sum(u_i[:k_exit+1]**2) * dt

    # Compute objective components
    f_time = np.sum(travel_times)
    f_energy = np.sum(energies)

    # Total objective (weighted sum)
    f_total = params.alpha * f_time + (1 - params.alpha) * f_energy

    # Additional information for analysis
    info = {
        'travel_times': travel_times,
        'energies': energies,
        'trajectories': trajectories,
        'total_time': f_time,
        'total_energy': f_energy,
        'actual_K_needed': actual_K_needed,  # How many steps actually used
        'time_efficiency': actual_K_needed / K,  # e.g., 0.3 means only used 30%
        'avg_crossing_time': f_time / N  # Average time per vehicle
    }
    
    return f_total, f_time, f_energy, info

# ============================================================================
# CONSTRAINT CHECKERS
# ============================================================================

def check_constraint_1_dynamics(trajectories: List, u_profiles: np.ndarray,
                                x0: np.ndarray, v0: np.ndarray, 
                                params: ProblemParameters) -> Dict:
    """
    Constraint 1: Vehicle dynamics
    
    xᵢ[k+1] = xᵢ[k] + vᵢ[k]·Δt + 0.5·uᵢ[k]·Δt²
    vᵢ[k+1] = vᵢ[k] + uᵢ[k]·Δt
    
    This is enforced by construction in simulate_vehicle_trajectory()
    """
    return {
        'satisfied': True,
        'type': 'dynamics',
        'note': 'Enforced by trajectory simulation',
        'violations': []
    }

def check_constraint_2_initial_conditions(trajectories: List, 
                                         x0: np.ndarray, v0: np.ndarray,
                                         params: ProblemParameters) -> Dict:
    """
    Constraint 2: Initial conditions
    
    xᵢ[0] = 0 (or xᵢ⁰)
    vᵢ[0] = vᵢ⁰
    """
    violations = []
    tol = 1e-6
    
    for i, (x_traj, v_traj) in enumerate(trajectories):
        if abs(x_traj[0] - x0[i]) > tol:
            violations.append({
                'vehicle': i,
                'type': 'position',
                'expected': x0[i],
                'actual': x_traj[0],
                'error': abs(x_traj[0] - x0[i])
            })
        
        if abs(v_traj[0] - v0[i]) > tol:
            violations.append({
                'vehicle': i,
                'type': 'velocity',
                'expected': v0[i],
                'actual': v_traj[0],
                'error': abs(v_traj[0] - v0[i])
            })
    
    return {
        'satisfied': len(violations) == 0,
        'type': 'initial_conditions',
        'violations': violations
    }

def check_constraint_3_acceleration_limits(u_profiles: np.ndarray, 
                                          params: ProblemParameters) -> Dict:
    """
    Constraint 3: Acceleration limits
    
    u_min ≤ uᵢ[k] ≤ u_max  ∀i ∈ N, k ∈ ET
    """
    violations = []
    
    for i in range(u_profiles.shape[0]):
        # Check lower bound
        below_min = u_profiles[i, :] < params.u_min
        if np.any(below_min):
            violations.append({
                'vehicle': i,
                'type': 'below_minimum',
                'min_value': np.min(u_profiles[i, :]),
                'limit': params.u_min,
                'count': np.sum(below_min),
                'timesteps': np.where(below_min)[0].tolist()
            })
        
        # Check upper bound
        above_max = u_profiles[i, :] > params.u_max
        if np.any(above_max):
            violations.append({
                'vehicle': i,
                'type': 'above_maximum',
                'max_value': np.max(u_profiles[i, :]),
                'limit': params.u_max,
                'count': np.sum(above_max),
                'timesteps': np.where(above_max)[0].tolist()
            })
    
    return {
        'satisfied': len(violations) == 0,
        'type': 'acceleration_limits',
        'violations': violations
    }

def check_constraint_4_velocity_limits(trajectories: List, params: ProblemParameters) -> Dict:
    """
    Constraint 4: Velocity constraints (no stopping in intersection)
    
    Before merging zone: v_min_approach ≤ vᵢ[k] ≤ v_max
    Inside merging zone: v_min ≤ vᵢ[k] ≤ v_max  (no stopping)
    """
    violations = []
    
    for i, (x_traj, v_traj) in enumerate(trajectories):
        # Find where vehicle enters merging zone (x >= L - S)
        in_merge_zone = x_traj >= (params.L - params.S)
        
        # Before merging zone
        before_merge = ~in_merge_zone
        v_before = v_traj[before_merge]
        
        if len(v_before) > 0:
            below_min = v_before < params.v_min_approach
            if np.any(below_min):
                violations.append({
                    'vehicle': i,
                    'zone': 'approach',
                    'type': 'below_minimum',
                    'min_velocity': np.min(v_before),
                    'limit': params.v_min_approach,
                    'count': np.sum(below_min)
                })
        
        # Inside merging zone (no stopping constraint)
        v_inside = v_traj[in_merge_zone]
        if len(v_inside) > 0:
            below_min = v_inside < params.v_min
            if np.any(below_min):
                violations.append({
                    'vehicle': i,
                    'zone': 'merging',
                    'type': 'stopping_violation',
                    'min_velocity': np.min(v_inside),
                    'limit': params.v_min,
                    'count': np.sum(below_min)
                })
        
        # Check upper bound everywhere
        above_max = v_traj > params.v_max
        if np.any(above_max):
            violations.append({
                'vehicle': i,
                'zone': 'all',
                'type': 'above_maximum',
                'max_velocity': np.max(v_traj),
                'limit': params.v_max,
                'count': np.sum(above_max)
            })
    
    return {
        'satisfied': len(violations) == 0,
        'type': 'velocity_limits',
        'violations': violations
    }

def check_constraint_5_reaching_zones(trajectories: List, params: ProblemParameters) -> Dict:
    """
    Constraint 5: Vehicles must reach merging zone AND exit control zone

    UPDATED: With extended time horizon (K=100), this constraint checks that vehicles
    eventually reach the exit. The time minimization objective ensures they don't
    dawdle - vehicles that take longer get worse objective values.

    Checks for:
    1. Vehicle reaches merging zone entry (x >= L - S)
    2. Vehicle reaches control zone exit (x >= L)
    3. No teleportation: position is monotonically increasing
    4. Vehicle properly enters merge zone (not just touches it)
    """
    violations = []

    for i, (x_traj, v_traj) in enumerate(trajectories):
        # Check 1: Reaches merging zone
        reaches_merge = np.any(x_traj >= (params.L - params.S))
        if not reaches_merge:
            violations.append({
                'vehicle': i,
                'type': 'does_not_reach_merging_zone',
                'max_position': np.max(x_traj),
                'target': params.L - params.S,
                'deficit': (params.L - params.S) - np.max(x_traj),
                'note': f'Vehicle did not reach merge zone even with {params.T_max}s horizon'
            })

        # Check 2: Reaches exit
        reaches_exit = np.any(x_traj >= params.L)
        if not reaches_exit:
            violations.append({
                'vehicle': i,
                'type': 'does_not_reach_exit',
                'max_position': np.max(x_traj),
                'target': params.L,
                'deficit': params.L - np.max(x_traj),
                'note': f'Vehicle did not complete crossing even with {params.T_max}s horizon'
            })

        # Check 3: No teleportation (monotonic increasing)
        # Position should never decrease (vehicles only move forward)
        position_decreases = np.diff(x_traj) < -1e-6  # Small tolerance for numerical error
        if np.any(position_decreases):
            violations.append({
                'vehicle': i,
                'type': 'position_not_monotonic',
                'note': 'Vehicle position decreased (impossible for forward motion)',
                'first_violation_index': np.where(position_decreases)[0][0]
            })

        # Check 4: Vehicle actually ENTERS merging zone (not just touches it)
        # Must spend at least 2 time steps inside merging zone
        in_merge_zone = (x_traj >= params.L - params.S) & (x_traj < params.L)
        time_steps_in_merge = np.sum(in_merge_zone)

        if reaches_merge and time_steps_in_merge < 2:
            violations.append({
                'vehicle': i,
                'type': 'insufficient_time_in_merge_zone',
                'time_steps': time_steps_in_merge,
                'required': 2,
                'note': 'Vehicle barely touches merge zone instead of properly entering'
            })

    return {
        'satisfied': len(violations) == 0,
        'type': 'reaching_zones',
        'violations': violations
    }

def check_constraint_6_rear_end_collision(trajectories: List,
                                         t0: np.ndarray,
                                         params: ProblemParameters) -> Dict:
    """
    Constraint 6: Rear-end collision avoidance (SAME LANE only)

    Only check vehicles traveling in the EXACT SAME DIRECTION:
    - E→W vehicles only check against other E→W
    - W→E vehicles only check against other W→E
    - N→S vehicles only check against other N→S
    - S→N vehicles only check against other S→N

    Note: Same road but opposite direction = no rear-end collision possible
    """
    violations = []
    dt = params.dt

    # Check all vehicle pairs
    for i in range(params.N - 1):
        for j in range(i + 1, params.N):
            # Only check if in same lane (same direction)
            if not params.vehicles_in_same_lane(i, j):
                continue

            x_i, v_i = trajectories[i]
            x_j, v_j = trajectories[j]

            # Create time arrays
            t_i = t0[i] + np.arange(len(x_i)) * dt
            t_j = t0[j] + np.arange(len(x_j)) * dt

            # Find overlapping time range
            t_start = max(t_i[0], t_j[0])
            t_end = min(t_i[-1], t_j[-1])

            if t_start >= t_end:
                continue

            # Sample common time points
            common_times = np.arange(t_start, t_end, dt)

            # Interpolate positions
            x_i_common = np.interp(common_times, t_i, x_i)
            x_j_common = np.interp(common_times, t_j, x_j)

            # Check separation
            separation = np.abs(x_i_common - x_j_common)
            min_separation = np.min(separation)
            min_sep_idx = np.argmin(separation)
            min_sep_time = common_times[min_sep_idx]

            if min_separation < params.delta:
                violations.append({
                    'vehicle_pair': (i, j),
                    'direction': params.get_vehicle_direction(i),
                    'min_separation': min_separation,
                    'required': params.delta,
                    'deficit': params.delta - min_separation,
                    'time_of_min_separation': min_sep_time,
                    'position_i': x_i_common[min_sep_idx],
                    'position_j': x_j_common[min_sep_idx]
                })

    return {
        'satisfied': len(violations) == 0,
        'type': 'rear_end_collision',
        'violations': violations
    }

def check_constraint_6b_lateral_physical_collision(trajectories: List,
                                                   t0: np.ndarray,
                                                   params: ProblemParameters) -> Dict:
    """
    Constraint 6B: Physical lateral collision (CONFLICT-POINT BASED)

    NEW IMPLEMENTATION: Allows multiple vehicles in intersection simultaneously,
    but checks if they reach the CONFLICT POINT (center of intersection) at
    dangerously close times.

    For perpendicular vehicles (EW/WE vs NS/SN):
    - Compute when each reaches the conflict point (L - S/2)
    - Check if arrival times differ by at least safety margin
    - Safety margin = delta / v_min (time needed to clear conflict point)
    """
    violations = []
    dt = params.dt
    conflict_matrix = params.get_conflict_matrix()

    # Conflict point is at the CENTER of the merging zone
    conflict_position = params.L - params.S / 2.0

    # Minimum time separation at conflict point (seconds)
    # Based on: time = distance / velocity
    min_time_separation = params.delta / params.v_min

    for i in range(params.N):
        for j in range(i + 1, params.N):
            if conflict_matrix[i, j] != 1:
                continue  # Not perpendicular, skip

            x_i, v_i = trajectories[i]
            x_j, v_j = trajectories[j]

            t_i = t0[i] + np.arange(len(x_i)) * dt
            t_j = t0[j] + np.arange(len(x_j)) * dt

            # Find when each vehicle reaches conflict point
            # Use linear interpolation for accuracy

            # Vehicle i: find crossing time
            if x_i[-1] < conflict_position:
                # Doesn't reach conflict point
                continue

            idx_before_i = np.where(x_i < conflict_position)[0]
            idx_after_i = np.where(x_i >= conflict_position)[0]

            if len(idx_after_i) == 0:
                continue

            k_cross_i = idx_after_i[0]
            if k_cross_i == 0:
                t_cross_i = t_i[0]
                v_cross_i = v_i[0]
            else:
                # Linear interpolation
                x_before = x_i[k_cross_i - 1]
                x_after = x_i[k_cross_i]
                t_before = t_i[k_cross_i - 1]
                t_after = t_i[k_cross_i]

                alpha = (conflict_position - x_before) / (x_after - x_before)
                t_cross_i = t_before + alpha * (t_after - t_before)
                v_cross_i = v_i[k_cross_i - 1] + alpha * (v_i[k_cross_i] - v_i[k_cross_i - 1])

            # Vehicle j: find crossing time
            if x_j[-1] < conflict_position:
                continue

            idx_after_j = np.where(x_j >= conflict_position)[0]
            if len(idx_after_j) == 0:
                continue

            k_cross_j = idx_after_j[0]
            if k_cross_j == 0:
                t_cross_j = t_j[0]
                v_cross_j = v_j[0]
            else:
                x_before = x_j[k_cross_j - 1]
                x_after = x_j[k_cross_j]
                t_before = t_j[k_cross_j - 1]
                t_after = t_j[k_cross_j]

                alpha = (conflict_position - x_before) / (x_after - x_before)
                t_cross_j = t_before + alpha * (t_after - t_before)
                v_cross_j = v_j[k_cross_j - 1] + alpha * (v_j[k_cross_j] - v_j[k_cross_j - 1])

            # Check time separation at conflict point
            time_separation = abs(t_cross_i - t_cross_j)

            # Required separation depends on speeds
            # Use conservative estimate: slower vehicle needs more clearance
            v_slower = min(abs(v_cross_i), abs(v_cross_j))
            if v_slower < 0.1:  # Near-zero velocity
                v_slower = params.v_min

            required_separation = params.delta / v_slower

            if time_separation < required_separation:
                violations.append({
                    'vehicle_pair': (i, j),
                    'direction_i': params.get_vehicle_direction(i),
                    'direction_j': params.get_vehicle_direction(j),
                    't_cross_i': t_cross_i,
                    't_cross_j': t_cross_j,
                    'time_separation': time_separation,
                    'required_separation': required_separation,
                    'deficit': required_separation - time_separation,
                    'conflict_position': conflict_position
                })

    return {
        'satisfied': len(violations) == 0,
        'type': 'lateral_physical_collision',
        'note': 'Conflict-point based: checks arrival time at intersection center',
        'violations': violations
    }

def check_constraint_7_lateral_collision(trajectories: List,
                                        Z_binary: np.ndarray,
                                        t0: np.ndarray,
                                        params: ProblemParameters) -> Dict:
    """
    Constraint 7: Lateral collision avoidance (priority timing rules)

    FIXED: Properly handles binary variables by ROUNDING to nearest integer
    Z values from SA might be continuous (0.3, 0.7), but we interpret as binary:
    - Z < 0.5 → treat as 0 (vehicle j has priority)
    - Z >= 0.5 → treat as 1 (vehicle i has priority)
    """
    violations = []

    conflict_matrix = params.get_conflict_matrix()

    z_index = 0
    for i in range(params.N):
        for j in range(i + 1, params.N):
            if conflict_matrix[i, j] == 1:
                # These vehicles conflict (perpendicular paths)

                # Get Z_ij and ROUND to binary
                if z_index < len(Z_binary):
                    z_ij_raw = Z_binary[z_index]
                    z_ij = 1.0 if z_ij_raw >= 0.5 else 0.0  # ROUND to binary
                else:
                    z_ij = 0.5  # Unspecified

                z_index += 1

                # Find merging times
                x_i, v_i = trajectories[i]
                x_j, v_j = trajectories[j]

                # Entry time to merging zone (x >= L - S)
                idx_i = np.where(x_i >= params.L - params.S)[0]
                t_m_i = t0[i] + (idx_i[0] * params.dt if len(idx_i) > 0 else np.inf)

                idx_j = np.where(x_j >= params.L - params.S)[0]
                t_m_j = t0[j] + (idx_j[0] * params.dt if len(idx_j) > 0 else np.inf)

                # Exit time from merging zone (x >= L)
                idx_exit_i = np.where(x_i >= params.L)[0]
                t_f_i = t0[i] + (idx_exit_i[0] * params.dt if len(idx_exit_i) > 0 else np.inf)

                idx_exit_j = np.where(x_j >= params.L)[0]
                t_f_j = t0[j] + (idx_exit_j[0] * params.dt if len(idx_exit_j) > 0 else np.inf)

                # Safety time buffer
                crossing_time = params.dt_safe

                # Check priority constraint based on ROUNDED Z value
                if z_ij >= 0.5:
                    # Vehicle i has priority (should exit before j enters)
                    required = t_f_i + crossing_time <= t_m_j
                    gap = t_m_j - (t_f_i + crossing_time)

                    if not required:
                        violations.append({
                            'vehicle_pair': (i, j),
                            'priority': f'Vehicle {i} should go first (Z={z_ij_raw:.2f}→{z_ij:.0f})',
                            'z_ij_raw': z_ij_raw,
                            'z_ij_rounded': z_ij,
                            't_m_i': t_m_i,
                            't_f_i': t_f_i,
                            't_m_j': t_m_j,
                            'time_gap': gap,
                            'required_gap': 0.0,
                            'type': 'insufficient_separation'
                        })
                else:
                    # Vehicle j has priority (z_ij = 0, meaning z_ji = 1)
                    required = t_f_j + crossing_time <= t_m_i
                    gap = t_m_i - (t_f_j + crossing_time)

                    if not required:
                        violations.append({
                            'vehicle_pair': (i, j),
                            'priority': f'Vehicle {j} should go first (Z={z_ij_raw:.2f}→{z_ij:.0f})',
                            'z_ij_raw': z_ij_raw,
                            'z_ij_rounded': z_ij,
                            't_m_j': t_m_j,
                            't_f_j': t_f_j,
                            't_m_i': t_m_i,
                            'time_gap': gap,
                            'required_gap': 0.0,
                            'type': 'insufficient_separation'
                        })

    return {
        'satisfied': len(violations) == 0,
        'type': 'lateral_collision',
        'note': 'Binary variables rounded: Z<0.5→0, Z≥0.5→1',
        'violations': violations
    }

# ============================================================================
# MASTER FEASIBILITY CHECKER
# ============================================================================

def feasibility_check(x_decision: np.ndarray,
                      x0: np.ndarray,
                      v0: np.ndarray,
                      t0: np.ndarray,
                      params: ProblemParameters) -> Tuple[bool, Dict]:
    """
    Master feasibility checker - evaluates all 7 constraints
    
    Args:
        x_decision: Decision vector [u_profiles, Z_binary]
        x0: Initial positions (N,)
        v0: Initial velocities (N,)
        t0: Initial times (N,)
        params: Problem parameters
        
    Returns:
        is_feasible: Boolean indicating if solution is feasible
        all_violations: Dictionary containing results from all constraint checks
    """
    N = params.N
    K = params.K
    dt = params.dt
    
    # Extract decision variables
    u_profiles = x_decision[:N*K].reshape(N, K)
    Z_binary = x_decision[N*K:]  # Binary variables
    
    # Simulate all vehicles
    trajectories = []
    for i in range(N):
        x_traj, v_traj = simulate_vehicle_trajectory(u_profiles[i], x0[i], v0[i], dt, K)
        trajectories.append((x_traj, v_traj))
    
    # Check all constraints
    all_violations = {}
    
    all_violations['constraint_1_dynamics'] = check_constraint_1_dynamics(
        trajectories, u_profiles, x0, v0, params)
    
    all_violations['constraint_2_initial_conditions'] = check_constraint_2_initial_conditions(
        trajectories, x0, v0, params)
    
    all_violations['constraint_3_acceleration_limits'] = check_constraint_3_acceleration_limits(
        u_profiles, params)
    
    all_violations['constraint_4_velocity_limits'] = check_constraint_4_velocity_limits(
        trajectories, params)
    
    all_violations['constraint_5_reaching_zones'] = check_constraint_5_reaching_zones(
        trajectories, params)

    all_violations['constraint_6_rear_end_collision'] = check_constraint_6_rear_end_collision(
        trajectories, t0, params)

    all_violations['constraint_6b_lateral_physical'] = check_constraint_6b_lateral_physical_collision(
        trajectories, t0, params)

    all_violations['constraint_7_lateral_collision'] = check_constraint_7_lateral_collision(
        trajectories, Z_binary, t0, params)
    
    # Determine overall feasibility
    is_feasible = all(
        result['satisfied'] 
        for result in all_violations.values()
    )
    
    return is_feasible, all_violations

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_random_solution(params: ProblemParameters, scale: float = 0.3) -> np.ndarray:
    """
    Generate a random solution for testing

    Args:
        params: Problem parameters
        scale: Fraction of acceleration bounds to use (0-1)
               scale=0.3 for safe testing, scale=1.0 for full exploration

    Returns:
        x: Decision vector [u_profiles, Z_binary]
    """
    N = params.N
    K = params.K

    # Generate random acceleration profiles
    u_profiles = np.random.uniform(
        params.u_min * scale,
        params.u_max * scale,
        size=(N, K)
    )

    # Generate random binary variables
    n_conflicts = np.sum(params.get_conflict_matrix()) // 2
    Z_binary = np.random.randint(0, 2, size=n_conflicts).astype(float)

    # Concatenate
    x = np.concatenate([u_profiles.flatten(), Z_binary])

    return x

def print_constraint_summary(violations: Dict):
    """Simple constraint summary - robust to different violation payloads."""
    import numpy as np

    print("\n" + "="*70)
    print("CONSTRAINT CHECKING SUMMARY")
    print("="*70)

    constraint_names = {
        'constraint_1_dynamics': 'Constraint 1: Vehicle Dynamics',
        'constraint_2_initial_conditions': 'Constraint 2: Initial Conditions',
        'constraint_3_acceleration_limits': 'Constraint 3: Acceleration Limits',
        'constraint_4_velocity_limits': 'Constraint 4: Velocity Limits',
        'constraint_5_reaching_zones': 'Constraint 5: Reaching Zones',
        'constraint_6_rear_end_collision': 'Constraint 6: Rear-End Physical Collision',
        'constraint_6b_lateral_physical': 'Constraint 6B: Lateral Physical Collision',
        'constraint_7_lateral_collision': 'Constraint 7: Priority Timing Rules'
    }

    for key, name in constraint_names.items():
        result = violations[key]
        status = "[PASS]" if result['satisfied'] else "[FAIL]"
        print(f"{name}: {status}")

        if result['satisfied']:
            continue

        for v in result['violations']:
            # ---------- Constraint 5: several subtypes ----------
            if key == 'constraint_5_reaching_zones':
                vtype = v.get('type', '')
                if vtype in ('does_not_reach_merging_zone', 'does_not_reach_exit'):
                    max_pos = v.get('max_position', float('nan'))
                    target = v.get('target', float('nan'))
                    print(f"  V{v.get('vehicle','?')}: Reached {max_pos:.1f}m, needed {target:.1f}m")
                elif vtype == 'position_not_monotonic':
                    idx = v.get('first_violation_index', '?')
                    print(f"  V{v.get('vehicle','?')}: position decreased (first at k={idx})")
                elif vtype == 'insufficient_time_in_merge_zone':
                    steps = v.get('time_steps', 0)
                    req = v.get('required', 2)
                    print(f"  V{v.get('vehicle','?')}: spent {steps} steps in merge (min {req})")
                else:
                    print(f"  V{v.get('vehicle','?')}: {vtype} – details: {v}")

            # ---------- Constraint 6: rear-end (same lane) ----------
            elif key == 'constraint_6_rear_end_collision':
                i, j = v['vehicle_pair']
                print(f"  V{i} & V{j}: {v['min_separation']:.2f}m apart (need {v['required']:.1f}m)")

            # ---------- Constraint 6B: lateral physical (conflict point) ----------
            elif key == 'constraint_6b_lateral_physical':
                i, j = v['vehicle_pair']
                sep = v.get('time_separation', float('nan'))
                req = v.get('required_clearance', float('nan'))
                print(f"  V{i} ({v.get('direction_i','?')}) & V{j} ({v.get('direction_j','?')}): "
                      f"Δt={sep:.2f}s < req {req:.2f}s (PHYSICAL COLLISION)")

            # ---------- Constraint 7: priority timing ----------
            elif key == 'constraint_7_lateral_collision':
                i, j = v['vehicle_pair']
                gap = v.get('time_gap', float('nan'))
                if isinstance(gap, (int, float)) and np.isfinite(gap):
                    print(f"  V{i} & V{j}: time gap {gap:.1f}s (violates assigned priority)")
                else:
                    print(f"  V{i} & V{j}: time gap -inf (vehicle didn’t reach merge/exit)")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_combined_visualization(x_decision: np.ndarray, x0: np.ndarray,
                                v0: np.ndarray, t0: np.ndarray,
                                params: ProblemParameters):
    """
    Combined visualization: Animation + Stats + Timeline (NO position plot)
    Layout: Top row = Animation + Stats, Bottom row = Timeline
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle

    N = params.N
    K = params.K
    dt = params.dt
    u_profiles = x_decision[:N*K].reshape(N, K)

    # Simulate all vehicles
    trajectories = []
    max_time = 0
    for i in range(N):
        x_traj, v_traj = simulate_vehicle_trajectory(u_profiles[i], x0[i], v0[i], dt, K)
        t_traj = t0[i] + np.arange(len(x_traj)) * dt
        trajectories.append((t_traj, x_traj, v_traj))
        max_time = max(max_time, t_traj[-1])

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 10))
    fig.canvas.manager.set_window_title('Multi-Vehicle Intersection Simulation')

    # GridSpec for custom layout: 2 rows, 2 columns
    # Top row: intersection (left) and stats (right)
    # Bottom row: timeline (spans both columns)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1])

    ax_intersection = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_timeline = fig.add_subplot(gs[1, :])  # Spans both columns

    # ========================================================================
    # TOP-LEFT: ANIMATED INTERSECTION
    # ========================================================================

    road_width = 8
    lane_offset = road_width / 4
    view_range = params.L * 0.6

    ax_intersection.set_xlim(-view_range, view_range)
    ax_intersection.set_ylim(-view_range, view_range)
    ax_intersection.set_aspect('equal')
    ax_intersection.set_facecolor('#E8F5E9')

    # Draw roads
    road_ew = Rectangle((-view_range, -road_width/2), 2*view_range, road_width,
                        facecolor='#424242', edgecolor='black', linewidth=2, zorder=1)
    ax_intersection.add_patch(road_ew)

    road_ns = Rectangle((-road_width/2, -view_range), road_width, 2*view_range,
                        facecolor='#424242', edgecolor='black', linewidth=2, zorder=1)
    ax_intersection.add_patch(road_ns)

    # Lane markings
    ax_intersection.plot([-view_range, view_range], [0, 0],
                color='yellow', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
    ax_intersection.plot([0, 0], [-view_range, view_range],
                color='yellow', linestyle='--', linewidth=2, alpha=0.8, zorder=2)

    # Conflict zone
    conflict_size = params.S
    conflict_zone = Rectangle((-conflict_size/2, -conflict_size/2),
                              conflict_size, conflict_size,
                              facecolor='red', alpha=0.3,
                              edgecolor='red', linewidth=3, linestyle='--', zorder=3)
    ax_intersection.add_patch(conflict_zone)

    ax_intersection.text(0, conflict_size/2 + 2, 'DANGER', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='red', zorder=10)

    ax_intersection.set_xlabel('East-West (m)', fontsize=11)
    ax_intersection.set_ylabel('North-South (m)', fontsize=11)
    ax_intersection.set_title('Live Intersection View', fontsize=12, fontweight='bold')
    ax_intersection.grid(True, alpha=0.3, linestyle=':', zorder=0)

    # Vehicle setup
    colors = ['#2196F3', '#00BCD4', '#F44336', '#FF9800']

    vehicle_patches = []
    vehicle_labels = []
    trail_lines = []
    trail_data = [{'x': [], 'y': []} for _ in range(N)]

    ew_lane = lane_offset
    we_lane = -lane_offset
    ns_lane = -lane_offset
    sn_lane = lane_offset

    for i in range(N):
        direction = params.get_vehicle_direction(i)

        if direction in ['EW', 'WE']:
            vehicle = Rectangle((0, 0), 5, 2, facecolor=colors[i],
                              edgecolor='white', linewidth=2, zorder=5)
        else:
            vehicle = Rectangle((0, 0), 2, 5, facecolor=colors[i],
                              edgecolor='white', linewidth=2, zorder=5)

        vehicle_patches.append(ax_intersection.add_patch(vehicle))

        label = ax_intersection.text(0, 0, f'V{i}', ha='center', va='center',
                           fontsize=9, fontweight='bold', color='white', zorder=6)
        vehicle_labels.append(label)

        trail, = ax_intersection.plot([], [], color=colors[i], linewidth=2, alpha=0.4, zorder=4)
        trail_lines.append(trail)

    # ========================================================================
    # TOP-RIGHT: STATS PANEL
    # ========================================================================

    ax_stats.axis('off')
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)

    time_text = ax_stats.text(0.5, 0.98, '', ha='center', va='top',
                             fontsize=14, fontweight='bold',
                             transform=ax_stats.transAxes)

    stats_box = ax_stats.text(0.05, 0.92, '', ha='left', va='top',
                             fontsize=9, family='monospace',
                             transform=ax_stats.transAxes,
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    collision_text = ax_stats.text(0.5, 0.1, '', ha='center', va='center',
                                  fontsize=12, fontweight='bold',
                                  transform=ax_stats.transAxes)

    # ========================================================================
    # BOTTOM: TIMELINE (full width)
    # ========================================================================

    directions = ['E->W', 'W->E', 'N->S', 'S->N']

    for i in range(N):
        x_traj = trajectories[i][1]

        idx_enter = np.where(x_traj >= params.L - params.S)[0]
        idx_exit = np.where(x_traj >= params.L)[0]

        if len(idx_enter) > 0 and len(idx_exit) > 0:
            t_enter = t0[i] + idx_enter[0] * dt
            t_exit = t0[i] + idx_exit[0] * dt

            ax_timeline.barh(i, t_exit - t_enter, left=t_enter, height=0.6,
                           color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5)

            ax_timeline.text(t_enter - 0.5, i, f'V{i} {directions[i]}',
                           va='center', ha='right', fontsize=10, fontweight='bold')

    ax_timeline.set_xlabel('Time (s)', fontsize=11)
    ax_timeline.set_ylabel('Vehicle', fontsize=11)
    ax_timeline.set_title('Conflict Zone Timeline', fontsize=12, fontweight='bold')
    ax_timeline.set_yticks(range(N))
    ax_timeline.set_yticklabels([f'V{i}' for i in range(N)], fontsize=10)
    ax_timeline.grid(True, axis='x', alpha=0.3)
    ax_timeline.set_xlim(left=0)

    # ========================================================================
    # ANIMATION FUNCTION
    # ========================================================================

    def animate(frame):
        current_time = frame * dt * 0.5
        time_text.set_text(f'TIME: {current_time:.2f}s')

        stats_str = ''
        collision_warning = False
        vehicles_in_conflict = []

        for i in range(N):
            t_traj, x_traj, v_traj = trajectories[i]

            if current_time < t_traj[0] or current_time > t_traj[-1]:
                vehicle_patches[i].set_visible(False)
                vehicle_labels[i].set_visible(False)
                continue

            idx = np.searchsorted(t_traj, current_time)
            if idx == 0:
                idx = 1
            if idx >= len(t_traj):
                idx = len(t_traj) - 1

            t1, t2 = t_traj[idx-1], t_traj[idx]
            x1, x2 = x_traj[idx-1], x_traj[idx]
            v1, v2 = v_traj[idx-1], v_traj[idx]

            alpha_interp = (current_time - t1) / (t2 - t1) if t2 > t1 else 0
            x_current = x1 + alpha_interp * (x2 - x1)
            v_current = v1 + alpha_interp * (v2 - v1)

            direction = params.get_vehicle_direction(i)

            if direction == 'EW':
                x_pos = -params.L + x_current
                y_pos = ew_lane
                vehicle_patches[i].set_xy((x_pos - 2.5, y_pos - 1))
            elif direction == 'WE':
                x_pos = params.L - x_current
                y_pos = we_lane
                vehicle_patches[i].set_xy((x_pos - 2.5, y_pos - 1))
            elif direction == 'NS':
                x_pos = ns_lane
                y_pos = -params.L + x_current
                vehicle_patches[i].set_xy((x_pos - 1, y_pos - 2.5))
            else:  # SN
                x_pos = sn_lane
                y_pos = params.L - x_current
                vehicle_patches[i].set_xy((x_pos - 1, y_pos - 2.5))

            vehicle_labels[i].set_position((x_pos, y_pos))
            trail_data[i]['x'].append(x_pos)
            trail_data[i]['y'].append(y_pos)
            trail_lines[i].set_data(trail_data[i]['x'], trail_data[i]['y'])

            vehicle_patches[i].set_visible(True)
            vehicle_labels[i].set_visible(True)

            if x_current < params.L - params.S:
                zone = 'APPROACH'
            elif x_current < params.L:
                zone = 'CONFLICT'
                if (-conflict_size/2 <= x_pos <= conflict_size/2 and
                    -conflict_size/2 <= y_pos <= conflict_size/2):
                    vehicles_in_conflict.append(i)
            else:
                zone = 'EXIT'

            stats_str += f'V{i} [{direction}]: {v_current:4.1f}m/s | {zone}\n'

        # Check collision
        if len(vehicles_in_conflict) >= 2:
            horizontal = [v for v in vehicles_in_conflict
                         if params.get_vehicle_direction(v) in ['EW', 'WE']]
            vertical = [v for v in vehicles_in_conflict
                       if params.get_vehicle_direction(v) in ['NS', 'SN']]

            if len(horizontal) > 0 and len(vertical) > 0:
                collision_warning = True
                stats_str += '\nCOLLISION!\n'

        stats_box.set_text(stats_str)

        if collision_warning:
            stats_box.set_bbox(dict(boxstyle='round', facecolor='red', alpha=0.8))
            collision_text.set_text('COLLISION!')
            collision_text.set_color('red')
        else:
            stats_box.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            collision_text.set_text('Safe')
            collision_text.set_color('green')

        return vehicle_patches + vehicle_labels + trail_lines + [time_text, stats_box, collision_text]

    n_frames = int(max_time / (dt * 0.5)) + 30
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True, repeat=True)

    plt.tight_layout()

    # Store animation in figure object to prevent garbage collection
    fig._animation = anim

    return fig, anim


def plot_animated_intersection(x_decision: np.ndarray, x0: np.ndarray,
                               v0: np.ndarray, t0: np.ndarray,
                               params: ProblemParameters):
    """
    FIXED: Proper top-down intersection animation

    Coordinate system:
    - Intersection CENTER at (0, 0)
    - EW vehicles: move from (-L, lane_y) to (0, lane_y) horizontally
    - NS vehicles: move from (lane_x, -L) to (lane_x, 0) vertically
    - Conflict zone: Square from (-S/2, -S/2) to (+S/2, +S/2)
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle

    N = params.N
    K = params.K
    dt = params.dt
    u_profiles = x_decision[:N*K].reshape(N, K)

    # Simulate all vehicles
    trajectories = []
    max_time = 0
    for i in range(N):
        x_traj, v_traj = simulate_vehicle_trajectory(u_profiles[i], x0[i], v0[i], dt, K)
        t_traj = t0[i] + np.arange(len(x_traj)) * dt
        trajectories.append((t_traj, x_traj, v_traj))
        max_time = max(max_time, t_traj[-1])

    # Setup figure
    fig = plt.figure(figsize=(18, 9))
    fig.canvas.manager.set_window_title('Intersection Simulation - Top View')

    ax_main = plt.subplot(1, 2, 1)
    ax_stats = plt.subplot(1, 2, 2)
    ax_stats.axis('off')
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)

    # Intersection dimensions
    road_width = 8
    lane_offset = road_width / 4
    view_range = params.L * 0.6

    ax_main.set_xlim(-view_range, view_range)
    ax_main.set_ylim(-view_range, view_range)
    ax_main.set_aspect('equal')
    ax_main.set_facecolor('#E8F5E9')

    # Draw roads
    road_ew = Rectangle((-view_range, -road_width/2), 2*view_range, road_width,
                        facecolor='#424242', edgecolor='black', linewidth=2, zorder=1)
    ax_main.add_patch(road_ew)

    road_ns = Rectangle((-road_width/2, -view_range), road_width, 2*view_range,
                        facecolor='#424242', edgecolor='black', linewidth=2, zorder=1)
    ax_main.add_patch(road_ns)

    # Lane markings
    ax_main.plot([-view_range, view_range], [0, 0],
                color='yellow', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
    ax_main.plot([0, 0], [-view_range, view_range],
                color='yellow', linestyle='--', linewidth=2, alpha=0.8, zorder=2)

    # Conflict zone
    conflict_size = params.S
    conflict_zone = Rectangle((-conflict_size/2, -conflict_size/2),
                              conflict_size, conflict_size,
                              facecolor='red', alpha=0.3,
                              edgecolor='red', linewidth=3, linestyle='--',
                              zorder=3, label='CONFLICT ZONE')
    ax_main.add_patch(conflict_zone)

    # Control zone boundaries
    entry_distance = params.L - params.S
    ax_main.axvline(x=-entry_distance, color='orange', linestyle=':',
                   linewidth=2, alpha=0.6, zorder=2)
    ax_main.axhline(y=-entry_distance, color='orange', linestyle=':',
                   linewidth=2, alpha=0.6, zorder=2)

    ax_main.text(0, conflict_size/2 + 2, 'DANGER', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='red', zorder=10)

    # Vehicle setup
    colors = ['#2196F3', '#00BCD4', '#F44336', '#FF9800']
    labels = ['V0 (E->W)', 'V1 (W->E)', 'V2 (N->S)', 'V3 (S->N)']
    direction_arrows = ['->', '<-', 'v', '^']

    vehicle_patches = []
    vehicle_labels = []
    trail_lines = []
    trail_data = [{'x': [], 'y': []} for _ in range(N)]

    # Lane positions for 4 directions
    ew_lane = lane_offset      # E->W uses upper lane
    we_lane = -lane_offset     # W->E uses lower lane
    ns_lane = -lane_offset     # N->S uses left lane
    sn_lane = lane_offset      # S->N uses right lane

    for i in range(N):
        direction = params.get_vehicle_direction(i)

        if direction in ['EW', 'WE']:  # Horizontal vehicles
            vehicle = Rectangle((0, 0), 5, 2,
                              facecolor=colors[i], edgecolor='white',
                              linewidth=2, zorder=5)
        else:  # Vertical vehicles (NS, SN)
            vehicle = Rectangle((0, 0), 2, 5,
                              facecolor=colors[i], edgecolor='white',
                              linewidth=2, zorder=5)

        vehicle_patches.append(ax_main.add_patch(vehicle))

        label = ax_main.text(0, 0, f'V{i}', ha='center', va='center',
                           fontsize=10, fontweight='bold', color='white', zorder=6)
        vehicle_labels.append(label)

        trail, = ax_main.plot([], [], color=colors[i], linewidth=3,
                            alpha=0.4, zorder=4)
        trail_lines.append(trail)

    ax_main.set_xlabel('East-West Position (m)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('North-South Position (m)', fontsize=12, fontweight='bold')
    ax_main.set_title('4-Way Intersection - Top View', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3, linestyle=':', zorder=0)
    ax_main.legend(loc='upper left', fontsize=9)

    # Stats panel
    time_text = ax_stats.text(0.5, 0.98, '', ha='center', va='top',
                             fontsize=16, fontweight='bold',
                             transform=ax_stats.transAxes)

    stats_box = ax_stats.text(0.05, 0.92, '', ha='left', va='top',
                             fontsize=10, family='monospace',
                             transform=ax_stats.transAxes,
                             bbox=dict(boxstyle='round', facecolor='lightgreen',
                                     alpha=0.8, pad=1))

    collision_text = ax_stats.text(0.5, 0.15, '', ha='center', va='center',
                                  fontsize=14, fontweight='bold',
                                  transform=ax_stats.transAxes)

    # Animation function
    def animate(frame):
        current_time = frame * dt * 0.5
        time_text.set_text(f'TIME: {current_time:.2f} s')

        stats_str = '=' * 45 + '\n'
        vehicles_in_conflict = []
        collision_warning = False

        for i in range(N):
            t_traj, x_traj, v_traj = trajectories[i]

            if current_time < t_traj[0] or current_time > t_traj[-1]:
                vehicle_patches[i].set_visible(False)
                vehicle_labels[i].set_visible(False)
                continue

            idx = np.searchsorted(t_traj, current_time)
            if idx == 0:
                idx = 1
            if idx >= len(t_traj):
                idx = len(t_traj) - 1

            t1, t2 = t_traj[idx-1], t_traj[idx]
            x1, x2 = x_traj[idx-1], x_traj[idx]
            v1, v2 = v_traj[idx-1], v_traj[idx]

            alpha = (current_time - t1) / (t2 - t1) if t2 > t1 else 0
            x_current = x1 + alpha * (x2 - x1)
            v_current = v1 + alpha * (v2 - v1)

            direction = params.get_vehicle_direction(i)

            if direction == 'EW':  # East -> West
                x_pos = -params.L + x_current
                y_pos = ew_lane
                vehicle_patches[i].set_xy((x_pos - 2.5, y_pos - 1))
                vehicle_labels[i].set_position((x_pos, y_pos))

            elif direction == 'WE':  # West -> East (reverse)
                x_pos = params.L - x_current  # Moving backwards
                y_pos = we_lane
                vehicle_patches[i].set_xy((x_pos - 2.5, y_pos - 1))
                vehicle_labels[i].set_position((x_pos, y_pos))

            elif direction == 'NS':  # North -> South
                x_pos = ns_lane
                y_pos = -params.L + x_current
                vehicle_patches[i].set_xy((x_pos - 1, y_pos - 2.5))
                vehicle_labels[i].set_position((x_pos, y_pos))

            else:  # 'SN': South -> North (reverse)
                x_pos = sn_lane
                y_pos = params.L - x_current  # Moving backwards
                vehicle_patches[i].set_xy((x_pos - 1, y_pos - 2.5))
                vehicle_labels[i].set_position((x_pos, y_pos))

            trail_data[i]['x'].append(x_pos)
            trail_data[i]['y'].append(y_pos)

            trail_lines[i].set_data(trail_data[i]['x'], trail_data[i]['y'])
            vehicle_patches[i].set_visible(True)
            vehicle_labels[i].set_visible(True)

            if x_current < params.L - params.S:
                zone = 'APPROACH'
            elif x_current < params.L:
                zone = 'CONFLICT'
                if (-conflict_size/2 <= x_pos <= conflict_size/2 and
                    -conflict_size/2 <= y_pos <= conflict_size/2):
                    vehicles_in_conflict.append((i, x_pos, y_pos))
            else:
                zone = 'EXIT'

            direction_symbol = direction_arrows[i]
            stats_str += f'Vehicle {i} [{direction} {direction_symbol}]\n'
            stats_str += f'  Speed:    {v_current:5.1f} m/s\n'
            stats_str += f'  Progress: {x_current:5.1f}/{params.L:.0f} m\n'
            stats_str += f'  Zone:     {zone}\n\n'

        if len(vehicles_in_conflict) >= 2:
            # Check for perpendicular conflicts
            horizontal_in_conflict = [v for v in vehicles_in_conflict
                                     if params.get_vehicle_direction(v[0]) in ['EW', 'WE']]
            vertical_in_conflict = [v for v in vehicles_in_conflict
                                   if params.get_vehicle_direction(v[0]) in ['NS', 'SN']]

            if len(horizontal_in_conflict) > 0 and len(vertical_in_conflict) > 0:
                collision_warning = True
                stats_str += '=' * 45 + '\n'
                stats_str += 'COLLISION RISK!\n'
                horiz_ids = [v[0] for v in horizontal_in_conflict]
                vert_ids = [v[0] for v in vertical_in_conflict]
                stats_str += f'Horizontal: {horiz_ids}\n'
                stats_str += f'Vertical: {vert_ids}\n'

        stats_box.set_text(stats_str)

        if collision_warning:
            stats_box.set_bbox(dict(boxstyle='round', facecolor='red', alpha=0.8, pad=1))
            collision_text.set_text('COLLISION WARNING')
            collision_text.set_color('red')
        else:
            stats_box.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=1))
            collision_text.set_text('Safe - No Conflicts')
            collision_text.set_color('green')

        return vehicle_patches + vehicle_labels + trail_lines + [time_text, stats_box, collision_text]

    n_frames = int(max_time / (dt * 0.5)) + 30
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True, repeat=True)

    plt.tight_layout()
    return fig, anim


def plot_simplified_trajectories(x_decision: np.ndarray, x0: np.ndarray, v0: np.ndarray,
                                 t0: np.ndarray, params: ProblemParameters):
    """
    2. Simplified Trajectories - Easy to understand position and velocity plots
    """
    N = params.N
    K = params.K
    dt = params.dt
    u_profiles = x_decision[:N*K].reshape(N, K)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title('Simplified Trajectories')

    colors = ['blue', 'cyan', 'red', 'orange']
    labels = ['V0 (E->W)', 'V1 (W->E)', 'V2 (N->S)', 'V3 (S->N)']

    for i in range(N):
        x_traj, v_traj = simulate_vehicle_trajectory(u_profiles[i], x0[i], v0[i], dt, K)
        t = t0[i] + np.arange(len(x_traj)) * dt

        # Left: Distance traveled
        ax1.plot(t, x_traj, color=colors[i], label=labels[i], linewidth=2.5, marker='o',
                markersize=3, markevery=5)

        # Right: Speed over time
        ax2.plot(t, v_traj, color=colors[i], label=labels[i], linewidth=2.5, marker='s',
                markersize=3, markevery=5)

    # Configure left plot (Distance)
    ax1.axhspan(params.L - params.S, params.L, color='red', alpha=0.2, label='Conflict Zone')
    ax1.axhline(y=params.L - params.S, color='red', linestyle='--', linewidth=2,
                label='Zone Entry')
    ax1.axhline(y=params.L, color='green', linestyle='--', linewidth=2, label='Zone Exit')
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Distance from Start (meters)', fontsize=12, fontweight='bold')
    ax1.set_title('How Far Has Each Vehicle Traveled?', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_ylim(0, params.L + 10)

    # Configure right plot (Speed)
    ax2.axhline(y=params.v_min, color='red', linestyle='--', linewidth=2,
                label=f'Min Speed ({params.v_min} m/s)', alpha=0.7)
    ax2.axhline(y=params.v_max, color='red', linestyle='--', linewidth=2,
                label=f'Max Speed ({params.v_max} m/s)', alpha=0.7)
    ax2.axhspan(0, params.v_min, color='red', alpha=0.1)
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speed (m/s)', fontsize=12, fontweight='bold')
    ax2.set_title('How Fast Is Each Vehicle Going?', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.set_ylim(0, params.v_max + 5)

    plt.tight_layout()


def plot_intersection_timeline(x_decision: np.ndarray, x0: np.ndarray,
                               v0: np.ndarray, t0: np.ndarray,
                               params: ProblemParameters):
    """
    3. Timeline - Shows when vehicles are in conflict zone (bars must NOT overlap!)
    """
    N = params.N
    K = params.K
    dt = params.dt
    u_profiles = x_decision[:N*K].reshape(N, K)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.canvas.manager.set_window_title('Conflict Zone Timeline')

    colors = ['blue', 'cyan', 'red', 'orange']
    directions = ['E->W', 'W->E', 'N->S', 'S->N']

    overlap_detected = False
    time_intervals = []

    for i in range(N):
        x_traj, v_traj = simulate_vehicle_trajectory(u_profiles[i], x0[i], v0[i], dt, K)

        # Find conflict zone entry and exit times
        idx_enter = np.where(x_traj >= params.L - params.S)[0]
        idx_exit = np.where(x_traj >= params.L)[0]

        if len(idx_enter) > 0 and len(idx_exit) > 0:
            t_enter = t0[i] + idx_enter[0] * dt
            t_exit = t0[i] + idx_exit[0] * dt
            duration = t_exit - t_enter

            time_intervals.append((i, t_enter, t_exit))

            # Draw bar
            bar = ax.barh(i, duration, left=t_enter, height=0.7,
                         color=colors[i], alpha=0.8, edgecolor='black', linewidth=2)

            # Add time labels
            ax.text(t_enter, i, f'{t_enter:.1f}s', ha='right', va='center',
                   fontsize=9, fontweight='bold', color='darkgreen')
            ax.text(t_exit, i, f'{t_exit:.1f}s', ha='left', va='center',
                   fontsize=9, fontweight='bold', color='darkred')

            # Add vehicle label
            ax.text(t_enter - 0.8, i, f'  V{i} {directions[i]}  ',
                   va='center', ha='right', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.6))

    # Check for overlaps - only perpendicular directions can collide
    for i in range(len(time_intervals)):
        for j in range(i + 1, len(time_intervals)):
            v1_id, t1_enter, t1_exit = time_intervals[i]
            v2_id, t2_enter, t2_exit = time_intervals[j]

            # Get directions
            dir1 = params.get_vehicle_direction(v1_id)
            dir2 = params.get_vehicle_direction(v2_id)

            # Only check if perpendicular (horizontal vs vertical)
            is_perpendicular = ((dir1 in ['EW', 'WE'] and dir2 in ['NS', 'SN']) or
                               (dir1 in ['NS', 'SN'] and dir2 in ['EW', 'WE']))

            if is_perpendicular:
                # Check time overlap
                if not (t1_exit <= t2_enter or t2_exit <= t1_enter):
                    overlap_detected = True
                    overlap_start = max(t1_enter, t2_enter)
                    overlap_end = min(t1_exit, t2_exit)

                    # Highlight overlap
                    ax.axvspan(overlap_start, overlap_end, color='red', alpha=0.3, zorder=0)

                    # Add warning
                    mid_time = (overlap_start + overlap_end) / 2
                    ax.text(mid_time, N + 0.5, f'[WARNING] COLLISION!\nV{v1_id} vs V{v2_id}',
                           ha='center', fontsize=11, fontweight='bold', color='red',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

    # Configure plot
    ax.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Vehicle ID', fontsize=13, fontweight='bold')

    if overlap_detected:
        title = '[WARNING] CONFLICT ZONE TIMELINE - COLLISION DETECTED!'
        title_color = 'red'
    else:
        title = '[SAFE] CONFLICT ZONE TIMELINE - No Overlaps'
        title_color = 'green'

    ax.set_title(title, fontsize=14, fontweight='bold', color=title_color,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.set_yticks(range(N))
    ax.set_yticklabels([f'Vehicle {i}' for i in range(N)], fontsize=11)
    ax.grid(True, axis='x', alpha=0.4, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(-0.5, N)

    # Add legend
    legend_text = "KEY RULE: Bars must NOT overlap for perpendicular directions!\n"
    legend_text += "[SAFE] No overlap = Safe  |  [DANGER] Overlap = COLLISION"
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()

# ============================================================================
# MAIN DEMONSTRATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Create parameters
    params = ProblemParameters()

    print("="*70)
    print("PROBLEM SETUP")
    print("="*70)
    print(f"\nVehicles: {params.N} total ({params.N_EW} E->W, {params.N_WE} W->E, {params.N_NS} N->S, {params.N_SN} S->N)")
    print(f"Time discretization: K={params.K} steps x dt={params.dt}s = {params.K*params.dt}s horizon")
    print(f"Acceleration limits: [{params.u_min}, {params.u_max}] m/s^2")
    print(f"Velocity limits: [{params.v_min}, {params.v_max}] m/s (no stopping in intersection)")
    print(f"Control zone: L={params.L}m, Merging zone: S={params.S}m")
    print(f"Safety distance: delta={params.delta}m, Time separation: dt_safe={params.dt_safe}s")
    print(f"\nObjective: alpha={params.alpha} (time weight)")
    print(f"  - Time component weight: {params.alpha}")
    print(f"  - Energy component weight: {1-params.alpha}")

    # Decision vector dimensions
    n_conflicts = np.sum(params.get_conflict_matrix()) // 2
    dim_accel = params.N * params.K
    dim_binary = n_conflicts
    dim_total = dim_accel + dim_binary

    print(f"\nDecision Vector Dimensions:")
    print(f"  - Acceleration variables: {dim_accel} (NxK = {params.N}x{params.K})")
    print(f"  - Binary priority variables: {dim_binary}")
    print(f"  - Total: {dim_total} decision variables")

    # Initial conditions
    # All vehicles start at position 0 in their trajectory
    # (trajectory position, not world position)
    x0 = np.zeros(params.N)

    # Initial velocities for each vehicle
    v0 = np.array([
        12.0,  # Vehicle 0: E->W
        11.0,  # Vehicle 1: W->E
        13.0,  # Vehicle 2: N->S
        10.0   # Vehicle 3: S->N
    ])

    # Staggered arrival times
    t0 = np.array([
        0.0,   # Vehicle 0: E->W arrives first
        0.5,   # Vehicle 1: W->E
        1.0,   # Vehicle 2: N->S
        1.5    # Vehicle 3: S->N arrives last
    ])

    print(f"\nVehicle Directions:")
    for i in range(params.N):
        direction_map = {
            'EW': 'East -> West',
            'WE': 'West -> East',
            'NS': 'North -> South',
            'SN': 'South -> North'
        }
        dir_code = params.get_vehicle_direction(i)
        print(f"  Vehicle {i}: {direction_map[dir_code]} | x0={x0[i]:.1f}m, v0={v0[i]:.1f}m/s, t0={t0[i]:.1f}s")

    # Generate test solution
    x_test = generate_random_solution(params)

    # Evaluate objective function
    print("\n" + "="*70)
    print("EVALUATING OBJECTIVE FUNCTION")
    print("="*70)
    f_total, f_time, f_energy, info = objective_function(x_test, x0, v0, t0, params)

    print(f"\nObjective Function Values:")
    print(f"  Total objective: f = {f_total:.4f}")
    print(f"  Time component: {f_time:.4f} seconds")
    print(f"  Energy component: {f_energy:.4f} (m²/s³)")
    print(f"\nIndividual Vehicle Metrics:")
    for i in range(params.N):
        print(f"  Vehicle {i}: Travel time = {info['travel_times'][i]:.2f}s, "
              f"Energy = {info['energies'][i]:.2f}")

    # Check feasibility
    print("\n" + "="*70)
    print("CHECKING FEASIBILITY")
    print("="*70)
    is_feasible, violations = feasibility_check(x_test, x0, v0, t0, params)

    print(f"\nOverall Feasibility: {'[FEASIBLE]' if is_feasible else '[INFEASIBLE]'}")
    print_constraint_summary(violations)

    # Generate combined visualization
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)
    fig, anim = plot_combined_visualization(x_test, x0, v0, t0, params)
    print("Animation created. Window should open with moving vehicles.")
    print("Close the window when done viewing.")
    plt.show()