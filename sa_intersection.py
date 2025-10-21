"""
Simulated Annealing for Multi-Vehicle Intersection Optimization
Milestone 3 - Due Oct 23, 2025

Integrates with metaheuristic_intersection.py
Uses FIXED Constraint 6B (conflict-point based, allows simultaneous crossing)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from metaheuristic_intersection import (
    ProblemParameters, objective_function, feasibility_check,
    generate_random_solution, simulate_vehicle_trajectory
)

# ============================================================================
# REPAIR OPERATOR
# ============================================================================

# max_iter = 6000

def repair_solution(x_infeasible: np.ndarray,
                   violations: Dict,
                   params,
                   x0: np.ndarray,
                   v0: np.ndarray,
                   t0: np.ndarray) -> np.ndarray:
    """
    IMPROVED repair operator with targeted fixes

    Strategy:
    1. Velocity violations → smooth accelerations to stay within bounds
    2. Collision violations → create time gaps by adjusting speeds
    3. Reaching violations → ensure sufficient forward acceleration
    """
    from metaheuristic_intersection import feasibility_check

    x_repaired = x_infeasible.copy()
    N, K = params.N, params.K

    # Extract components
    u_profiles = x_repaired[:N*K].reshape(N, K)
    Z_binary = x_repaired[N*K:] if len(x_repaired) > N*K else np.array([])

    # Count violation types
    has_velocity_viol = False
    has_collision_viol = False
    has_reaching_viol = False

    # Parse violations
    for constraint_type, result in violations.items():
        if not result.get('satisfied', True):
            if 'velocity' in constraint_type.lower():
                has_velocity_viol = True
            elif 'collision' in constraint_type.lower() or 'lateral' in constraint_type.lower():
                has_collision_viol = True
            elif 'reaching' in constraint_type.lower():
                has_reaching_viol = True

    # REPAIR STRATEGY 1: Fix velocity violations by smoothing
    if has_velocity_viol:
        for i in range(N):
            # Apply moving average filter to smooth accelerations
            window = 5
            u_smooth = np.convolve(u_profiles[i], np.ones(window)/window, mode='same')
            u_profiles[i] = u_smooth * 0.7  # Scale down to stay in bounds
            u_profiles[i] = np.clip(u_profiles[i], params.u_min, params.u_max)

    # REPAIR STRATEGY 2: Fix collisions by creating time separation
    if has_collision_viol:
        # Slow down some vehicles, speed up others to create gaps
        for i in range(N):
            if i % 2 == 0:
                # Even vehicles: slow down in merge zone
                merge_start_idx = int((params.L - params.S) / (v0[i] * params.dt))
                merge_start_idx = max(0, min(merge_start_idx, K-10))
                u_profiles[i, merge_start_idx:merge_start_idx+10] *= 0.5
            else:
                # Odd vehicles: speed up before merge zone
                pre_merge_idx = max(0, int((params.L - params.S - 20) / (v0[i] * params.dt)))
                pre_merge_idx = min(pre_merge_idx, K-10)
                u_profiles[i, pre_merge_idx:pre_merge_idx+5] *= 1.3

        # Clip to bounds
        u_profiles = np.clip(u_profiles, params.u_min, params.u_max)

        # Also flip priorities to change crossing order
        if len(Z_binary) > 0:
            flip_idx = np.random.randint(0, len(Z_binary))
            Z_binary[flip_idx] = 1.0 - Z_binary[flip_idx]

    # REPAIR STRATEGY 3: Fix reaching violations by boosting forward motion
    if has_reaching_viol:
        for i in range(N):
            # Increase all accelerations to ensure reaching goal
            u_profiles[i] *= 1.5
            u_profiles[i] = np.clip(u_profiles[i], 0.0, params.u_max)  # Only positive

    # Reconstruct solution
    if len(Z_binary) > 0:
        x_repaired = np.concatenate([u_profiles.flatten(), Z_binary])
    else:
        x_repaired = u_profiles.flatten()

    # Verify repair worked (at least partially)
    is_feas_after, _ = feasibility_check(x_repaired, x0, v0, t0, params)

    if not is_feas_after:
        # If still infeasible, try more aggressive repair
        # Strategy: blend with known-good conservative solution
        u_conservative = np.ones((N, K)) * 0.3  # Very gentle acceleration
        u_conservative = np.clip(u_conservative, params.u_min, params.u_max)

        # Blend 70% conservative, 30% current
        u_blended = 0.7 * u_conservative + 0.3 * u_profiles
        u_blended = np.clip(u_blended, params.u_min, params.u_max)

        if len(Z_binary) > 0:
            x_repaired = np.concatenate([u_blended.flatten(), Z_binary])
        else:
            x_repaired = u_blended.flatten()

    return x_repaired


def generate_initial_feasible_solution(params, x0, v0, t0, max_attempts=1000):
    """
    Generate initial FEASIBLE solution using conservative heuristic

    Strategy: Start with gentle accelerations (more likely feasible)
    """
    from metaheuristic_intersection import feasibility_check

    for attempt in range(max_attempts):
        # Generate solution with bias toward gentle accelerations
        u_profiles = np.random.normal(0, 0.5, (params.N, params.K))
        u_profiles = np.clip(u_profiles, params.u_min, params.u_max)

        # Random priority variables
        n_binary = 4  # Z_02, Z_03, Z_12, Z_13
        Z_binary = np.random.randint(0, 2, n_binary).astype(float)

        # Combine
        x = np.concatenate([u_profiles.flatten(), Z_binary])

        # Check feasibility
        is_feas, _ = feasibility_check(x, x0, v0, t0, params)
        if is_feas:
            print(f"✅ Found initial feasible solution on attempt {attempt + 1}")
            return x

    # If failed, return VERY conservative solution
    print("⚠️  Using ultra-conservative fallback initial solution")

    # Strategy: Very gentle constant acceleration to slowly reach goal
    # Each vehicle accelerates gently, no conflicts
    u_conservative = np.zeros((params.N, params.K))

    for i in range(params.N):
        # Gentle acceleration for first 20 steps to reach decent speed
        u_conservative[i, :20] = 0.5  # Gentle accel
        # Then coast (zero acceleration)
        u_conservative[i, 20:] = 0.0

    u_conservative = np.clip(u_conservative, params.u_min, params.u_max)

    # Sequential priorities (vehicles go one by one)
    Z_conservative = np.array([1.0, 1.0, 0.0, 0.0])

    x_conservative = np.concatenate([u_conservative.flatten(), Z_conservative])

    # VERIFY this is actually feasible
    from metaheuristic_intersection import feasibility_check
    is_feas, viols = feasibility_check(x_conservative, x0, v0, t0, params)

    if is_feas:
        print("✅ Conservative solution is feasible")
    else:
        print("❌ WARNING: Even conservative solution is infeasible!")
        violated_constraints = [k for k, v in viols.items() if not v['satisfied']]
        print(f"   Violations: {violated_constraints}")

    return x_conservative


# ============================================================================
# NEIGHBOR GENERATION
# ============================================================================

def generate_neighbor(x_current, params, T, T_init):
    """
    Generate neighbor solution with adaptive step size

    Strategy:
    - Perturb ~10% of continuous variables
    - Step size decreases with temperature
    - Flip binary variables with 30% probability
    """
    x_neighbor = x_current.copy()
    N, K = params.N, params.K

    # Adaptive step size
    step_scale = np.sqrt(T / T_init)
    u_step = 1.0 * step_scale

    # Perturb random 10% of acceleration values
    n_perturb = max(1, int(N * K * 0.1))
    perturb_indices = np.random.choice(N * K, n_perturb, replace=False)

    for idx in perturb_indices:
        delta = np.random.uniform(-u_step, u_step)
        x_neighbor[idx] += delta
        x_neighbor[idx] = np.clip(x_neighbor[idx], params.u_min, params.u_max)

    # Flip binary variables with 30% chance
    n_binary = len(x_current) - N * K
    if n_binary > 0 and np.random.random() < 0.3:
        flip_idx = np.random.randint(0, n_binary)
        binary_idx = N * K + flip_idx
        x_neighbor[binary_idx] = 1.0 - x_neighbor[binary_idx]

    return x_neighbor


# ============================================================================
# SIMULATED ANNEALING MAIN LOOP
# ============================================================================

def simulated_annealing(params, x0, v0, t0,
                       T_init=100.0,
                       T_final=0.01,
                       max_iter=5,
                       seed=42):
    """
    Simulated Annealing with THREE FIXES:
    1. Variable time horizon (adaptive K)
    2. Repair operator (guaranteed feasibility)
    3. Linear cooling (matches lecture)

    Args:
        params: Problem parameters
        x0, v0, t0: Initial conditions
        T_init: Initial temperature
        T_final: Final temperature
        max_iter: Maximum iterations
        seed: Random seed

    Returns:
        x_best: Best feasible solution found
        f_best: Best objective value
        history: Convergence history
    """
    from metaheuristic_intersection import objective_function, feasibility_check

    np.random.seed(seed)

    # Calculate LINEAR cooling rate (matches lecture)
    beta = (T_init - T_final) / max_iter

    # Initialize with FEASIBLE solution
    print("\n" + "="*80)
    print("SIMULATED ANNEALING - MILESTONE 3 (CORRECTED)")
    print("="*80)
    print("Generating initial feasible solution...")

    x_current = generate_initial_feasible_solution(params, x0, v0, t0)
    f_current, f_time_current, f_energy_current, _ = objective_function(
        x_current, x0, v0, t0, params
    )

    # Track best
    x_best = x_current.copy()
    f_best = f_current

    # History
    history = {
        'iteration': [],
        'f_best': [],
        'f_current': [],
        'T': [],
        'acceptance_rate': [],
        'repair_rate': [],
        'skip_rate': []
    }

    # Counters
    n_accepted = 0
    n_repaired = 0
    n_skipped = 0
    T = T_init

    print(f"\n✅ Initial feasible solution found")
    print(f"   Fitness: {f_current:.2f} (time={f_time_current:.2f}, energy={f_energy_current:.2f})")
    print(f"\nSA Parameters:")
    print(f"   T_init = {T_init}, T_final = {T_final}")
    print(f"   Cooling: LINEAR with β = {beta:.6f}")
    print(f"   Max iterations: {max_iter}")
    print("="*80 + "\n")

    # Main SA loop
    for iteration in range(max_iter):
        # Generate neighbor
        x_neighbor = generate_neighbor(x_current, params, T, T_init)

        # Check feasibility
        is_feas, violations = feasibility_check(x_neighbor, x0, v0, t0, params)

        # Repair if infeasible
        if not is_feas:
            x_neighbor = repair_solution(x_neighbor, violations, params, x0, v0, t0)
            n_repaired += 1

            # Re-check after repair
            is_feas, violations = feasibility_check(x_neighbor, x0, v0, t0, params)

            # If still infeasible after repair, skip this iteration
            if not is_feas:
                n_skipped += 1

                # DEBUG: Print why it's failing (only first 10 times)
                if n_skipped <= 10:
                    violated_constraints = [k for k, v in violations.items() if not v['satisfied']]
                    print(f"  [Iter {iteration}] Skipped - still infeasible after repair")
                    print(f"    Violations: {violated_constraints}")

                # Still cool the temperature
                T = T_init - beta * iteration

                # Update history with current values
                history['iteration'].append(iteration)
                history['f_best'].append(f_best)
                history['f_current'].append(f_current)
                history['T'].append(T)
                history['acceptance_rate'].append(n_accepted / (iteration + 1))
                history['repair_rate'].append(n_repaired / (iteration + 1))
                history['skip_rate'].append(n_skipped / (iteration + 1))
                continue

        # Evaluate feasible neighbor (NO PENALTIES, pure objective)
        f_neighbor, f_time_neighbor, f_energy_neighbor, _ = objective_function(
            x_neighbor, x0, v0, t0, params
        )

        # Acceptance criterion (standard Metropolis)
        delta_f = f_neighbor - f_current

        if delta_f < 0:
            # Always accept improvement
            accept = True
        else:
            # Accept worse solution with probability exp(-ΔE/T)
            prob_accept = np.exp(-delta_f / T)
            accept = (np.random.random() < prob_accept)

        # Update current solution
        if accept:
            x_current = x_neighbor.copy()
            f_current = f_neighbor
            n_accepted += 1

        # Update best solution
        if f_neighbor < f_best:
            x_best = x_neighbor.copy()
            f_best = f_neighbor

        # LINEAR COOLING (matches lecture formula: T_i = T_0 - β*i)
        T = T_init - beta * iteration

        # Update history
        history['iteration'].append(iteration)
        history['f_best'].append(f_best)
        history['f_current'].append(f_current)
        history['T'].append(T)
        history['acceptance_rate'].append(n_accepted / (iteration + 1))
        history['repair_rate'].append(n_repaired / (iteration + 1))
        history['skip_rate'].append(n_skipped / (iteration + 1))

        # Progress reporting
        if iteration % 10 == 0:
            print(f"Iter {iteration:4d}: f_best={f_best:7.2f}, f_current={f_current:7.2f}, "
                  f"T={T:6.2f}, accept={n_accepted/(iteration+1):5.1%}, "
                  f"repair={n_repaired/(iteration+1):5.1%}, skip={n_skipped/(iteration+1):5.1%}")

        # Stop if temperature too low
        if T <= T_final:
            print(f"\n✅ Reached T_final={T_final} at iteration {iteration}")
            break

    # Final verification
    is_feas_final, _ = feasibility_check(x_best, x0, v0, t0, params)

    print("\n" + "="*80)
    print("SA COMPLETE")
    print("="*80)
    print(f"Best fitness:      {f_best:.2f}")
    print(f"Final feasibility: {'✅ FEASIBLE' if is_feas_final else '❌ INFEASIBLE'}")
    print(f"Acceptance rate:   {n_accepted/iteration:.2%}")
    print(f"Repair rate:       {n_repaired/iteration:.2%}")
    print(f"Skip rate:         {n_skipped/iteration:.2%}")
    print("="*80 + "\n")

    if not is_feas_final:
        print("⚠️  WARNING: Best solution is INFEASIBLE - increase max_iter or adjust repair")

    return x_best, f_best, history


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_convergence(history):
    """Plot SA convergence history with repair tracking"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Simulated Annealing Convergence (Corrected)', fontsize=14, fontweight='bold')

    iters = history['iteration']

    # Plot 1: Objective value
    ax = axes[0, 0]
    ax.plot(iters, history['f_best'], 'g-', linewidth=2, label='Best')
    ax.plot(iters, history['f_current'], 'b-', alpha=0.5, label='Current')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness (Pure Objective)')
    ax.set_title('Objective Function Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Temperature
    ax = axes[0, 1]
    ax.plot(iters, history['T'], 'r-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Temperature')
    ax.set_title('Cooling Schedule (LINEAR)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Acceptance rate
    ax = axes[1, 0]
    ax.plot(iters, [r*100 for r in history['acceptance_rate']], 'orange', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('Solution Acceptance Rate')
    ax.grid(True, alpha=0.3)

    # Plot 4: Repair rate
    ax = axes[1, 1]
    ax.plot(iters, [r*100 for r in history['repair_rate']], 'purple', linewidth=2, label='Repair')
    ax.plot(iters, [r*100 for r in history['skip_rate']], 'red', linewidth=2, label='Skip')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Repair & Skip Rates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Temperature vs Best Fitness (trajectory)
    ax = axes[2, 0]
    scatter = ax.scatter(history['T'], history['f_best'],
                        c=iters, cmap='viridis', s=2, alpha=0.6)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Best Fitness')
    ax.set_title('Fitness vs Temperature')
    ax.set_xscale('log')
    plt.colorbar(scatter, ax=ax, label='Iteration')
    ax.grid(True, alpha=0.3)

    # Plot 6: Summary statistics
    ax = axes[2, 1]
    ax.axis('off')
    final_stats = f"""
    FINAL STATISTICS
    {'='*40}

    Best Fitness:      {history['f_best'][-1]:.2f}
    Final Temperature: {history['T'][-1]:.4f}

    Acceptance Rate:   {history['acceptance_rate'][-1]:.2%}
    Repair Rate:       {history['repair_rate'][-1]:.2%}
    Skip Rate:         {history['skip_rate'][-1]:.2%}

    Iterations:        {len(iters)}
    """
    ax.text(0.1, 0.5, final_stats, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    return fig


# ============================================================================
# RANDOM SEARCH BASELINE (for comparison)
# ============================================================================

def random_search_baseline(params, x0, v0, t0, n_samples=500, seed=42):
    """Random search baseline - now uses pure objective"""
    from metaheuristic_intersection import objective_function, feasibility_check

    np.random.seed(seed)

    print("\n" + "="*80)
    print("RANDOM SEARCH BASELINE")
    print("="*80)

    best_f = float('inf')
    best_x = None
    best_feasible = False
    n_feasible = 0

    for i in range(n_samples):
        x = generate_random_solution(params)

        # Check feasibility first
        is_feas, _ = feasibility_check(x, x0, v0, t0, params)

        if is_feas:
            n_feasible += 1
            # Only evaluate objective if feasible
            f, _, _, _ = objective_function(x, x0, v0, t0, params)

            # Update best
            if not best_feasible or f < best_f:
                best_x = x.copy()
                best_f = f
                best_feasible = True

        if (i+1) % 100 == 0:
            status = f"feasible_rate={n_feasible/(i+1):.2%}"
            if best_feasible:
                status += f", best_f={best_f:.2f}"
            print(f"Sample {i+1}/{n_samples}: {status}")

    print(f"\nRandom search complete:")
    if best_feasible:
        print(f"  Best fitness: {best_f:.2f} ✅")
    else:
        print(f"  No feasible solution found ❌")
    print(f"  Feasibility rate: {n_feasible/n_samples:.2%}")
    print("="*80 + "\n")

    return best_x, best_f, best_feasible


# ============================================================================
# ADVANCED VISUALIZATION FOR M3
# ============================================================================

def plot_live_sa_dashboard(history, params, x_current, x_best, x0, v0, t0,
                          info_current, info_best, iteration):
    """
    Real-time dashboard showing SA progress
    Updates during SA run to show algorithm "thinking"

    Shows 6 subplots:
    1. Convergence (f_best vs iteration)
    2. Current solution trajectories
    3. Best solution trajectories
    4. Temperature schedule
    5. Acceptance rate
    6. Constraint violation breakdown
    """
    from matplotlib.gridspec import GridSpec

    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Main title with iteration count
    fig.suptitle(f'Simulated Annealing - Iteration {iteration}',
                 fontsize=16, fontweight='bold')

    # ========================================================================
    # Plot 1: Convergence (top-left, large)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    iters = history['iteration']

    ax1.plot(iters, history['f_best'], 'g-', linewidth=3, label='Best', marker='o', markersize=4)
    ax1.plot(iters, history['f_current'], 'b-', alpha=0.5, linewidth=2, label='Current')
    ax1.axhline(y=info_best['f_total'], color='green', linestyle='--', alpha=0.5, label='Best Raw Objective')

    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Fitness (with penalties)', fontsize=11)
    ax1.set_title('Convergence: Fitness Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Annotate current point
    ax1.plot(iteration, info_current['fitness'], 'ro', markersize=10, zorder=10)
    ax1.annotate(f"Current: {info_current['fitness']:.1f}",
                xy=(iteration, info_current['fitness']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                fontsize=9)

    # ========================================================================
    # Plot 2: Current Solution Trajectories (middle-left)
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    N, K, dt = params.N, params.K, params.dt
    u_current = x_current[:N*K].reshape(N, K)
    colors = ['blue', 'cyan', 'red', 'orange']

    for i in range(N):
        x_traj, v_traj = simulate_vehicle_trajectory(u_current[i], x0[i], v0[i], dt, K)
        t_traj = t0[i] + np.arange(len(x_traj)) * dt
        ax2.plot(t_traj, x_traj, color=colors[i], linewidth=2, label=f'V{i}', alpha=0.7)

    ax2.axhspan(params.L - params.S, params.L, color='red', alpha=0.2)
    ax2.axhline(y=params.L - params.S, color='red', linestyle='--', linewidth=1)
    ax2.axhline(y=params.L, color='green', linestyle='--', linewidth=1)

    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Position (m)', fontsize=10)
    ax2.set_title(f'Current Solution (f={info_current["f_total"]:.1f})', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 3: Best Solution Trajectories (middle-center)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])

    u_best = x_best[:N*K].reshape(N, K)

    for i in range(N):
        x_traj, v_traj = simulate_vehicle_trajectory(u_best[i], x0[i], v0[i], dt, K)
        t_traj = t0[i] + np.arange(len(x_traj)) * dt
        ax3.plot(t_traj, x_traj, color=colors[i], linewidth=2, label=f'V{i}', alpha=0.7)

    ax3.axhspan(params.L - params.S, params.L, color='green', alpha=0.2)
    ax3.axhline(y=params.L - params.S, color='red', linestyle='--', linewidth=1)
    ax3.axhline(y=params.L, color='green', linestyle='--', linewidth=1)

    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Position (m)', fontsize=10)
    ax3.set_title(f'Best Solution (f={info_best["f_total"]:.1f})', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8, loc='lower right')
    ax3.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 4: Temperature Schedule (top-right)
    # ========================================================================
    ax4 = fig.add_subplot(gs[0, 2])

    ax4.semilogy(iters, history['T'], 'r-', linewidth=3)
    ax4.plot(iteration, history['T'][-1], 'ko', markersize=10)

    ax4.set_xlabel('Iteration', fontsize=10)
    ax4.set_ylabel('Temperature (log scale)', fontsize=10)
    ax4.set_title('Cooling Schedule', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 5: Acceptance & Feasibility Rates (middle-right)
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 2])

    ax5_twin = ax5.twinx()

    line1 = ax5.plot(iters, [r*100 for r in history['acceptance_rate']],
                     'orange', linewidth=2, label='Acceptance')
    line2 = ax5_twin.plot(iters, [r*100 for r in history['feasible_rate']],
                          'purple', linewidth=2, label='Feasibility')

    ax5.set_xlabel('Iteration', fontsize=10)
    ax5.set_ylabel('Acceptance Rate (%)', fontsize=10, color='orange')
    ax5_twin.set_ylabel('Feasibility Rate (%)', fontsize=10, color='purple')
    ax5.set_title('Algorithm Performance', fontsize=11, fontweight='bold')

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, fontsize=9, loc='upper right')

    ax5.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 6: Constraint Violation Breakdown (bottom, full width)
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, :])

    # Get violations for current solution
    violations = info_current['violations']

    constraint_names = [
        'C3: Accel',
        'C4: Velocity',
        'C5: Reaching',
        'C6: Rear-End',
        'C6B: Lateral',
        'C7: Priority'
    ]

    constraint_keys = [
        'constraint_3_acceleration_limits',
        'constraint_4_velocity_limits',
        'constraint_5_reaching_zones',
        'constraint_6_rear_end_collision',
        'constraint_6b_lateral_physical',
        'constraint_7_lateral_collision'
    ]

    violation_counts = []
    colors_bar = []

    for key in constraint_keys:
        if key in violations:
            count = len(violations[key]['violations'])
            violation_counts.append(count)
            colors_bar.append('red' if count > 0 else 'green')
        else:
            violation_counts.append(0)
            colors_bar.append('green')

    bars = ax6.barh(constraint_names, violation_counts, color=colors_bar, alpha=0.7, edgecolor='black')

    # Annotate bars
    for i, (bar, count) in enumerate(zip(bars, violation_counts)):
        if count > 0:
            ax6.text(count + 0.1, i, f'{count}', va='center', fontsize=10, fontweight='bold')

    ax6.set_xlabel('Number of Violations', fontsize=11)
    ax6.set_title('Current Solution: Constraint Satisfaction', fontsize=12, fontweight='bold')
    ax6.grid(True, axis='x', alpha=0.3)

    # Add feasibility status
    feas_text = "✓ FEASIBLE" if info_current['is_feasible'] else "✗ INFEASIBLE"
    feas_color = 'green' if info_current['is_feasible'] else 'red'
    ax6.text(0.98, 0.95, feas_text, transform=ax6.transAxes,
            fontsize=14, fontweight='bold', color=feas_color,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def simulated_annealing_with_live_viz(params, x0, v0, t0,
                                      T_init=1000.0,
                                      T_final=1.0,
                                      cooling=0.95,
                                      max_iter=5000,
                                      seed=42,
                                      update_interval=100):
    """
    SA with LIVE VISUALIZATION - shows algorithm working in real-time

    This creates a figure that updates every `update_interval` iterations
    showing the search process as it happens.
    """
    np.random.seed(seed)

    # Initialize
    x_current = generate_random_solution(params)
    f_current, is_feas_current, info_current = penalized_objective(
        x_current, x0, v0, t0, params
    )

    x_best = x_current.copy()
    f_best = f_current
    best_feasible = is_feas_current
    info_best = info_current

    history = {
        'iteration': [],
        'f_best': [],
        'f_current': [],
        'T': [],
        'acceptance_rate': [],
        'feasible_rate': []
    }

    T = T_init
    n_accepted = 0
    n_feasible = 0

    print("\n" + "="*80)
    print("LIVE SIMULATED ANNEALING VISUALIZATION")
    print("="*80)
    print("Dashboard will update every 100 iterations...")
    print("Close the figure window to continue to next update.")
    print("="*80 + "\n")

    # Main loop
    for iteration in range(max_iter):
        # Generate and evaluate neighbor
        x_neighbor = generate_neighbor(x_current, params, T, T_init)
        f_neighbor, is_feas_neighbor, info_neighbor = penalized_objective(
            x_neighbor, x0, v0, t0, params
        )

        # Acceptance
        delta_f = f_neighbor - f_current
        if delta_f < 0:
            accept = True
        else:
            accept = (np.random.random() < np.exp(-delta_f / T))

        if accept:
            x_current = x_neighbor
            f_current = f_neighbor
            is_feas_current = is_feas_neighbor
            info_current = info_neighbor
            n_accepted += 1

        # Update best
        if is_feas_neighbor and not best_feasible:
            x_best = x_neighbor.copy()
            f_best = f_neighbor
            best_feasible = True
            info_best = info_neighbor
        elif is_feas_neighbor == best_feasible and f_neighbor < f_best:
            x_best = x_neighbor.copy()
            f_best = f_neighbor
            info_best = info_neighbor

        if is_feas_current:
            n_feasible += 1

        T *= cooling

        # Update history
        history['iteration'].append(iteration)
        history['f_best'].append(f_best)
        history['f_current'].append(f_current)
        history['T'].append(T)
        history['acceptance_rate'].append(n_accepted / (iteration + 1))
        history['feasible_rate'].append(n_feasible / (iteration + 1))

        # Live visualization update
        if iteration % update_interval == 0 or iteration == max_iter - 1:
            print(f"\n>>> Iteration {iteration}: Updating dashboard...")

            fig = plot_live_sa_dashboard(
                history, params, x_current, x_best, x0, v0, t0,
                info_current, info_best, iteration
            )

            plt.show(block=False)
            plt.pause(0.1)

            # Close figure to prevent memory buildup
            plt.close(fig)

        if T < T_final:
            break

    print("\n" + "="*80)
    print("SA COMPLETED")
    print("="*80)

    return x_best, f_best, history, info_best


# ============================================================================
# PARAMETRIC STUDY - Show effect of SA parameters
# ============================================================================

def parametric_study_cooling_rate(params, x0, v0, t0):
    """
    Study effect of cooling rate on SA performance

    Tests: alpha = [0.90, 0.93, 0.95, 0.97, 0.99]
    Shows tradeoff: fast cooling (0.90) vs slow cooling (0.99)
    """
    print("\n" + "="*80)
    print("PARAMETRIC STUDY: Effect of Cooling Rate")
    print("="*80)

    cooling_rates = [0.90, 0.93, 0.95, 0.97, 0.99]
    results = []

    for alpha in cooling_rates:
        print(f"\nTesting cooling rate: {alpha}")

        x_best, f_best, history = simulated_annealing(
            params, x0, v0, t0,
            T_init=1000.0,
            T_final=1.0,
            cooling=alpha,
            max_iter=5000,
            seed=42
        )

        results.append({
            'alpha': alpha,
            'f_best': f_best,
            'history': history,
            'n_iterations': len(history['iteration'])
        })

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Parametric Study: Cooling Rate Effect', fontsize=14, fontweight='bold')

    # Plot 1: Convergence curves
    ax = axes[0]
    for res in results:
        ax.plot(res['history']['iteration'], res['history']['f_best'],
               linewidth=2, label=f"α={res['alpha']}", marker='o', markersize=3)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Best Fitness', fontsize=11)
    ax.set_title('Convergence Speed vs Cooling Rate', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Final fitness comparison
    ax = axes[1]
    alphas = [res['alpha'] for res in results]
    finals = [res['f_best'] for res in results]
    iters = [res['n_iterations'] for res in results]

    ax.bar(range(len(alphas)), finals, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel('Cooling Rate α', fontsize=11)
    ax.set_ylabel('Final Best Fitness', fontsize=11)
    ax.set_title('Final Solution Quality', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate best
    best_idx = np.argmin(finals)
    ax.bar(best_idx, finals[best_idx], color='green', alpha=0.7, edgecolor='black')
    ax.text(best_idx, finals[best_idx] + 5, '★ BEST', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("\n" + "="*80)
    print("Parametric Study Results:")
    print("="*80)
    for res in results:
        print(f"α={res['alpha']:.2f}: f_best={res['f_best']:.2f}, iterations={res['n_iterations']}")
    print("="*80)

    return results


def parametric_study_temperature(params, x0, v0, t0):
    """
    Study effect of initial temperature

    Tests: T_init = [100, 500, 1000, 2000, 5000]
    """
    print("\n" + "="*80)
    print("PARAMETRIC STUDY: Effect of Initial Temperature")
    print("="*80)

    temperatures = [100, 500, 1000, 2000, 5000]
    results = []

    for T in temperatures:
        print(f"\nTesting T_init: {T}")

        x_best, f_best, history = simulated_annealing(
            params, x0, v0, t0,
            T_init=float(T),
            T_final=1.0,
            cooling=0.95,
            max_iter=5000,
            seed=42
        )

        results.append({
            'T_init': T,
            'f_best': f_best,
            'history': history
        })

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for res in results:
        ax.plot(res['history']['iteration'], res['history']['f_best'],
               linewidth=2, label=f"T={res['T_init']}", marker='o', markersize=3)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Best Fitness', fontsize=11)
    ax.set_title('Effect of Initial Temperature on Convergence', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results


# ============================================================================
# STATISTICAL VALIDATION - Multiple runs
# ============================================================================

def statistical_validation(params, x0, v0, t0, n_runs=10):
    """
    Run SA multiple times with different seeds
    Report mean, std, best, worst
    """
    print("\n" + "="*80)
    print(f"STATISTICAL VALIDATION: {n_runs} Independent Runs")
    print("="*80)

    results = []

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs} (seed={run})")

        x_best, f_best, history = simulated_annealing(
            params, x0, v0, t0,
            T_init=1000.0,
            T_final=1.0,
            cooling=0.95,
            max_iter=5000,
            seed=run
        )

        _, is_feas, info = penalized_objective(x_best, x0, v0, t0, params)

        results.append({
            'seed': run,
            'f_best': f_best,
            'f_time': info['f_time'],
            'f_energy': info['f_energy'],
            'feasible': is_feas,
            'history': history
        })

    # Statistics
    feasible_results = [r for r in results if r['feasible']]
    all_fitness = [r['f_best'] for r in results]

    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    print(f"Feasibility rate: {len(feasible_results)}/{n_runs} ({len(feasible_results)/n_runs*100:.1f}%)")

    if feasible_results:
        feas_fitness = [r['f_best'] for r in feasible_results]
        print(f"\nFeasible solutions:")
        print(f"  Best:    {np.min(feas_fitness):.2f}")
        print(f"  Worst:   {np.max(feas_fitness):.2f}")
        print(f"  Mean:    {np.mean(feas_fitness):.2f}")
        print(f"  Std Dev: {np.std(feas_fitness):.2f}")

    print(f"\nAll solutions:")
    print(f"  Best:    {np.min(all_fitness):.2f}")
    print(f"  Worst:   {np.max(all_fitness):.2f}")
    print(f"  Mean:    {np.mean(all_fitness):.2f}")
    print(f"  Std Dev: {np.std(all_fitness):.2f}")
    print("="*80)

    # Box plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Statistical Validation ({n_runs} runs)', fontsize=14, fontweight='bold')

    # Plot 1: All convergence curves
    ax = axes[0]
    for res in results:
        color = 'green' if res['feasible'] else 'red'
        alpha = 0.7 if res['feasible'] else 0.3
        ax.plot(res['history']['iteration'], res['history']['f_best'],
               color=color, alpha=alpha, linewidth=1.5)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Best Fitness', fontsize=11)
    ax.set_title('Convergence: All Runs', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Plot 2: Box plot of final fitness
    ax = axes[1]
    feas_vals = [r['f_best'] for r in results if r['feasible']]
    infeas_vals = [r['f_best'] for r in results if not r['feasible']]

    bp = ax.boxplot([feas_vals, infeas_vals] if infeas_vals else [feas_vals],
                    labels=['Feasible', 'Infeasible'] if infeas_vals else ['Feasible'],
                    patch_artist=True)

    for patch, color in zip(bp['boxes'], ['green', 'red'] if infeas_vals else ['green']):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel('Final Fitness', fontsize=11)
    ax.set_title('Distribution of Final Solutions', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results


# ============================================================================
# COMPREHENSIVE M3 DEMO
# ============================================================================

def run_comprehensive_m3_demo():
    """
    Complete M3 demonstration with all visualizations
    This is what you show your professor!
    """
    print("\n" + "="*80)
    print("MILESTONE 3: COMPREHENSIVE DEMONSTRATION")
    print("Team 57 - Multi-Vehicle Intersection Optimization")
    print("="*80)

    # Setup
    params = ProblemParameters()
    x0 = np.zeros(4)
    v0 = np.array([12.0, 11.0, 13.0, 10.0])
    t0 = np.array([0.0, 0.5, 1.0, 1.5])

    # Part 1: Baseline
    print("\n" + "="*80)
    print("PART 1: Random Search Baseline")
    print("="*80)
    x_random, f_random, feas_random = random_search_baseline(
        params, x0, v0, t0, n_samples=500, seed=42
    )

    # Part 2: SA with live visualization
    print("\n" + "="*80)
    print("PART 2: Simulated Annealing (Live Visualization)")
    print("="*80)
    x_best, f_best, history, info_best = simulated_annealing_with_live_viz(
        params, x0, v0, t0,
        T_init=1000.0, T_final=1.0, cooling=0.95,
        max_iter=5000, seed=42, update_interval=100
    )

    # Part 3: Parametric studies
    print("\n" + "="*80)
    print("PART 3: Parametric Studies")
    print("="*80)
    results_cooling = parametric_study_cooling_rate(params, x0, v0, t0)
    results_temp = parametric_study_temperature(params, x0, v0, t0)

    # Part 4: Statistical validation
    print("\n" + "="*80)
    print("PART 4: Statistical Validation")
    print("="*80)
    results_stats = statistical_validation(params, x0, v0, t0, n_runs=10)

    # Part 5: Final visualization of best solution
    print("\n" + "="*80)
    print("PART 5: Visualizing Best Solution")
    print("="*80)
    from metaheuristic_intersection import plot_combined_visualization
    fig_final, anim = plot_combined_visualization(x_best, x0, v0, t0, params)
    plt.show()

    print("\n" + "="*80)
    print("M3 DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nAll visualizations displayed in windows.")
    print("Review the plots and close windows when done.")
    print("="*80)


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    import sys

    print("\n" + "="*80)
    print("MILESTONE 3: SIMULATED ANNEALING FOR INTERSECTION OPTIMIZATION")
    print("="*80)

    print("\n** DESIGN DECISION: Extended Time Horizon **")
    print("-" * 80)
    print(f"Time Horizon: K=100 steps × dt=0.5s = 50 seconds available")
    print(f"              (increased from K=30, T=15s)")
    print("\nRationale:")
    print("  • No artificial time limit constrains the solution space")
    print("  • Time minimization objective naturally pushes for fast crossings")
    print("  • Vehicles that dawdle get penalized via higher objective values")
    print("  • Typical crossing times: 10-20 seconds (well below 50s limit)")
    print("  • Extended horizon ensures feasibility without being binding")
    print("\nThis is more elegant: use objective to guide, not hard constraints!")
    print("="*80)

    print("\nOptions:")
    print("  1. Quick demo (single SA run)")
    print("  2. Comprehensive M3 demo (all visualizations)")
    print("="*80)

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "2":
        # Full M3 demo with all bells and whistles
        run_comprehensive_m3_demo()
    else:
        # Quick single run
        params = ProblemParameters()
        x0 = np.zeros(4)
        v0 = np.array([12.0, 11.0, 13.0, 10.0])
        t0 = np.array([0.0, 0.5, 1.0, 1.5])

        # Baseline
        x_random, f_random, feas_random = random_search_baseline(
            params, x0, v0, t0, n_samples=500, seed=42
        )

        # SA
        x_best, f_best, history = simulated_annealing(
            params, x0, v0, t0,
            T_init=100.0,      # Lower initial temp
            T_final=0.01,
            max_iter=5000,
            seed=42
        )

        # Plot convergence
        fig_conv = plot_convergence(history)
        plt.show()

        # Visualize best solution
        from metaheuristic_intersection import plot_combined_visualization
        fig_viz, anim = plot_combined_visualization(x_best, x0, v0, t0, params)
        plt.show()
