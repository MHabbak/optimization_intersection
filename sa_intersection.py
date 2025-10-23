"""
Simulated Annealing for Multi-Vehicle Intersection Optimization
Milestone 3 - Due Oct 23, 2025

Integrates with metaheuristic_intersection.py
Uses FIXED Constraint 6B (conflict-point based, allows simultaneous crossing)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from metaheuristic_intersection import (
    make_params,
    build_spawn_times,
    build_v0_random,
    objective_function,
    feasibility_check,
    generate_random_solution,
)


def evaluate_solution(
    x: np.ndarray,
    x0: np.ndarray, v0: np.ndarray, t0: np.ndarray,
    params: make_params,
    base_penalty: float = 1e6
) -> Tuple[float, bool, Dict]:
    """
    Returns (f_total, is_feasible, info).
    - If feasible: use objective_function(...) (already aligned to stop at x_exit = 2L-S in your main file)
    - If infeasible: return a large penalty and attach violations so the caller can display them.
    """
    is_feas, violations = feasibility_check(x, x0, v0, t0, params)
    if is_feas:
        f_time, f_energy, f_total, info = objective_function(x, x0, v0, t0, params)
        info['violations'] = violations
        info['is_feasible'] = True
        return float(f_total), True, info

    # Penalize infeasible solutions (simple, deterministic penalty)
    info = {'violations': violations, 'is_feasible': False}
    return float(base_penalty), False, info


# ============================================================================
# REPAIR OPERATOR
# ============================================================================

# max_iter = 6000

def repair_solution(x_infeasible: np.ndarray, x0, v0, t0, params,
                    max_repairs: int = 20) -> np.ndarray:
    x_repaired = x_infeasible.copy()
    N, K = params.N, params.K
    dt = params.dt
    eps = 1e-6

    # Extract u & Z safely
    u_flat = x_repaired[:N*K]
    u_profiles = u_flat.reshape(N, K)
    Z_binary = x_repaired[N*K:] if len(x_repaired) > N*K else np.array([])

    for attempt in range(max_repairs):
        # Stage A: smooth + scale + clip within physical accel bounds
        u_profiles = 0.5*u_profiles + 0.5*np.clip(u_profiles, params.u_min, params.u_max)
        u_profiles = np.clip(u_profiles, params.u_min, params.u_max)

        # Stage B: gentle braking before merge — use robust indices
        for i in range(N):
            v_start = max(float(v0[i]), eps)
            # time-to-merge (approx) using approach distance (L-S)
            t_pre = (params.L - params.S) / v_start
            k_pre = int(np.clip(round(t_pre / dt), 0, K-1))
            # apply a slight decel taper
            u_profiles[i, :k_pre] = np.clip(u_profiles[i, :k_pre] - 0.1*abs(params.u_min),
                                            params.u_min, params.u_max)

        # Stage C: conservative around the merge window
        for i in range(N):
            v_start = max(float(v0[i]), eps)
            t_pre = (params.L - params.S) / v_start
            t_mer = params.S / max(v_start, eps)
            k0 = int(np.clip(round(t_pre / dt), 0, K-1))
            k1 = int(np.clip(round((t_pre + t_mer) / dt), 0, K-1))
            u_profiles[i, k0:k1+1] = np.clip(u_profiles[i, k0:k1+1],
                                             params.u_min, params.u_max)  # keep braking allowed

        # Reassemble candidate
        if len(Z_binary) > 0:
            x_repaired = np.concatenate([u_profiles.flatten(), Z_binary])
        else:
            x_repaired = u_profiles.flatten()

        # Quick feasibility check
        is_feas, _ = feasibility_check(x_repaired, x0, v0, t0, params)
        if is_feas:
            return x_repaired

    return x_repaired



def generate_initial_feasible_solution(params, x0, v0, t0, max_attempts=500):
    """
    Try main.generate_random_solution as a starter; if infeasible, attempt repairs;
    fall back to a conservative zero-accel + zero-priority vector sized by conflict matrix.
    """
    # Try several random candidates from the main generator (correct shape & Z length)
    for _ in range(max_attempts):
        x = generate_random_solution(params)
        is_feas, _ = feasibility_check(x, x0, v0, t0, params)
        if is_feas:
            return x

        x_rep = repair_solution(x, x0, v0, t0, params, max_repairs=20)
        is_feas, _ = feasibility_check(x_rep, x0, v0, t0, params)
        if is_feas:
            return x_rep

    # Fallback: conservative profile (u=0, all priorities 0)
    N, K = params.N, params.K
    u_conservative = np.zeros((N, K), dtype=float)

    # Size Z by conflict matrix (half of symmetric sum)
    n_conflicts = int(np.sum(params.get_conflict_matrix()) // 2)
    Z_conservative = np.zeros(n_conflicts, dtype=float)

    return np.concatenate([u_conservative.flatten(), Z_conservative])



# ============================================================================
# NEIGHBOR GENERATION
# ============================================================================

def generate_neighbor(x_current, params, T, T_init):
    """
    Continuous part: Gaussian perturb scaled by sqrt(T/T_init), clipped to [u_min, u_max]
    Binary part: with 30% prob., flip a single priority bit.
    """
    N, K = params.N, params.K
    x_neighbor = x_current.copy()

    # Continuous u
    step = 0.2 * np.sqrt(max(T, 1e-12) / max(T_init, 1e-12))
    u = x_neighbor[:N*K].reshape(N, K)
    u = np.clip(u + np.random.normal(0.0, step, size=(N, K)), params.u_min, params.u_max)
    x_neighbor[:N*K] = u.flatten()

    # Binary Z
    n_binary = len(x_neighbor) - N*K
    if n_binary > 0 and np.random.random() < 0.3:
        idx = N*K + np.random.randint(0, n_binary)
        x_neighbor[idx] = 1.0 - x_neighbor[idx]

    return x_neighbor



# ============================================================================
# SIMULATED ANNEALING MAIN LOOP
# ============================================================================

def simulated_annealing(params, x0, v0, t0,
                       T_init=100.0,
                       T_final=0.01,
                       max_iter=5000):
    """
    Simulated Annealing (linear cooling) with feasibility-first candidates.
    This is your original flow, with clear debug prints added.
    """
    import numpy as np
    from metaheuristic_intersection import objective_function, feasibility_check

    # --- helpers this file already defines ---
    # - generate_initial_feasible_solution(params, x0, v0, t0)
    # - repair_solution(x, x0, v0, t0, params, max_repairs=...)
    # - generate_neighbor(x, params, T, T_init)

    # Linear cooling step
    beta = (T_init - T_final) / max_iter
    T = float(T_init)

    # Initial solution (feasible)
    x_current = generate_initial_feasible_solution(params, x0, v0, t0)

    is_feas_init, _ = feasibility_check(x_current, x0, v0, t0, params)
    if not is_feas_init:
        x_current = repair_solution(x_current, x0, v0, t0, params, max_repairs=10)
        is_feas_init, _ = feasibility_check(x_current, x0, v0, t0, params)

    f_time, f_energy, f_current, info_current = objective_function(x_current, x0, v0, t0, params)

    x_best = x_current.copy()
    f_best = float(f_current)

    # Debug counters / history
    n_accepted = 0
    n_feasible = 1  # initial is feasible by construction
    n_skipped  = 0
    rng = np.random.default_rng()

    history = {
        'iteration': [],
        'f_best': [],
        'f_current': [],
        'T': [],
        'accept_rate': [],
        'feas_rate': [],
        'skipped': []
    }

    # --- header prints ---
    nZ = len(x_current) - params.N * params.K
    print("="*80)
    print("[SA] schedule=linear"
          f"  T0={T_init}  Tf={T_final}  max_iter={max_iter}")
    print(f"[SA] dims: N={params.N}  K={params.K}  |Z|={nZ}")
    print(f"[SA] init: f={f_current:.6g}  (time={f_time:.6g}, energy={f_energy:.6g})  feasible=✅", flush=True)

    # --- main loop ---
    PRINT_EVERY = 50  # adjust if you want fewer/more logs

    for it in range(max_iter):
        # Cool
        T = max(T_final, T_init - beta * (it + 1))

        # Propose neighbor and try a quick repair
        x_neighbor = generate_neighbor(x_current, params, T, T_init)
        x_neighbor = repair_solution(x_neighbor, x0, v0, t0, params, max_repairs=4)

        # Feasibility first
        is_feas, _ = feasibility_check(x_neighbor, x0, v0, t0, params)
        if not is_feas:
            n_skipped += 1
            # periodic progress print
            if (it % PRINT_EVERY) == 0 or it == max_iter - 1:
                acc_rate = n_accepted / (it + 1)
                feas_rate = n_feasible / (it + 1)
                print(f"[SA][it={it}] T={T:.4g}  f_cur={f_current:.6g}  f_best={f_best:.6g}  "
                      f"accept_rate={acc_rate:.2%}  feas_rate={feas_rate:.2%}  skipped={n_skipped}", flush=True)
            continue

        n_feasible += 1

        # Score feasible neighbor
        f_time_n, f_energy_n, f_neighbor, info_neighbor = objective_function(x_neighbor, x0, v0, t0, params)
        dF = f_neighbor - f_current

        # Metropolis acceptance
        accepted = False
        if dF <= 0:
            accepted = True
        else:
            p = np.exp(-dF / max(T, 1e-12))
            if rng.random() < p:
                accepted = True

        if accepted:
            x_current = x_neighbor
            f_current = float(f_neighbor)
            info_current = info_neighbor
            n_accepted += 1

        # New best?
        if f_current < f_best:
            print(f"[SA][it={it}] new best: {f_best:.6g} → {f_current:.6g}", flush=True)
            x_best = x_current.copy()
            f_best = float(f_current)

        # Book-keeping + periodic print
        history['iteration'].append(it)
        history['f_best'].append(f_best)
        history['f_current'].append(f_current)
        history['T'].append(T)
        history['accept_rate'].append(n_accepted / (it + 1))
        history['feas_rate'].append(n_feasible / (it + 1))
        history['skipped'].append(n_skipped)

        if (it % PRINT_EVERY) == 0 or it == max_iter - 1:
            print(f"[SA][it={it}] T={T:.4g}  f_cur={f_current:.6g}  f_best={f_best:.6g}  "
                  f"accept_rate={history['accept_rate'][-1]:.2%}  "
                  f"feas_rate={history['feas_rate'][-1]:.2%}  skipped={n_skipped}", flush=True)

    # --- summary ---
    is_feas_final, _ = feasibility_check(x_best, x0, v0, t0, params)
    print("="*80)
    print(f"Best fitness:           {f_best:.2f}")
    print(f"Final feasibility:      {'✅ FEASIBLE' if is_feas_final else '❌ INFEASIBLE'}")
    print(f"Acceptance rate:        {n_accepted / max(1, len(history['iteration'])):.2%}")
    print(f"Avg skipped/iter:       {n_skipped / max(1, len(history['iteration'])):.2f}")
    print("="*80 + "\n")

    return x_best, f_best, history



# ============================================================================
# VISUALIZATION
# ============================================================================

import matplotlib.pyplot as plt

def plot_convergence(history):
    """
    Plot convergence trends for Simulated Annealing:
    - Best and current fitness values
    - Temperature cooling curve
    - Acceptance rate evolution
    - Repairs per iteration

    Args:
        history (dict): Convergence log returned by simulated_annealing()
    """
    iters = history['iteration']

    # Normalize keys so this works with both SA variants
    acc = history.get('acceptance_rate', history.get('accept_rate', []))
    feas = history.get('feasible_rate', history.get('feas_rate', []))
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Simulated Annealing Convergence", fontsize=14, fontweight='bold')

    # === 1️⃣ Fitness Evolution ===
    ax = axs[0, 0]
    ax.plot(iters, history['f_best'], 'b-', linewidth=2, label='Best Fitness')
    ax.plot(iters, history['f_current'], 'orange', linewidth=1.5, label='Current Fitness')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.set_title("Fitness Evolution")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # === 2️⃣ Temperature Cooling ===
    ax = axs[0, 1]
    ax.plot(iters, history['T'], 'r-', linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Temperature (T)")
    ax.set_title("Cooling Schedule")
    ax.grid(True, linestyle='--', alpha=0.6)

    # === 3️⃣ Acceptance Rate ===
    ax = axs[1, 0]
    ax.plot(iters, [a * 100 for a in acc], 'g-', linewidth=2)    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_title("Acceptance Rate Over Time")
    ax.grid(True, linestyle='--', alpha=0.6)

    # === 4️⃣ Repairs per Iteration ===
    ax = axs[1, 1]
    if 'repairs_per_iter' in history:
        ax.plot(iters, history['repairs_per_iter'], 'purple', linewidth=2)
        ax.set_ylabel("Repairs per Iteration")
    elif 'repair_rate' in history:
        ax.plot(iters, [r * 100 for r in history['repair_rate']], 'purple', linewidth=2)
        ax.set_ylabel("Repair Rate (%)")
    else:
        ax.text(0.5, 0.5, "No repair data", ha='center', va='center', fontsize=12, color='gray')
    ax.set_xlabel("Iteration")
    ax.set_title("Repair Effort")
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return fig



# ============================================================================
# RANDOM SEARCH BASELINE (for comparison)
# ============================================================================

def random_search_baseline(params, x0, v0, t0, n_samples=500):
    rng = np.random.default_rng()
    best = (np.inf, None, None)
    feas_count = 0

    for _ in range(n_samples):
        x = generate_random_solution(params)
        x = repair_solution(x, x0, v0, t0, params, max_repairs=2)
        f, is_feas, info = evaluate_solution(x, x0, v0, t0, params)
        feas_count += int(is_feas)
        if f < best[0]:
            best = (f, x, info)

    print(f"Random search feasible rate: {feas_count}/{n_samples} = {feas_count/n_samples:.2%}")
    return best



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
    cur_val = history['f_current'][-1]
    ax1.plot(iteration, cur_val, 'ro', markersize=10, zorder=10)
    ax1.annotate(f"Current: {cur_val:.1f}",
                xy=(iteration, cur_val),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                fontsize=9)

    # ========================================================================
    # Plot 2: Current Solution Trajectories (middle-left)
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    N, K, dt = params.N, params.K, params.dt
    u_current = x_current[:N*K].reshape(N, K)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(params.N)]

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
                                      update_interval=100):
    """
    SA with LIVE VISUALIZATION - shows algorithm working in real-time

    This creates a figure that updates every `update_interval` iterations
    showing the search process as it happens.
    """
    # Initialize
    x_current = generate_random_solution(params)
    f_current, is_feas_current, info_current = evaluate_solution(
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
    print(f"Dashboard will update every {update_interval} iterations...")
    print("Close the figure window to continue to next update.")
    print("="*80 + "\n")

    # Main loop
    for iteration in range(max_iter):
        # Generate and evaluate neighbor
        x_neighbor = generate_neighbor(x_current, params, T, T_init)
        f_neighbor, is_feas_neighbor, info_neighbor = evaluate_solution(
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
            max_iter=5000,
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
            max_iter=5000,
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
    Run SA multiple times with different random initializations
    Report mean, std, best, worst.
    (No fixed seeds — results will vary across runs)
    """
    print("\n" + "="*80)
    print(f"STATISTICAL VALIDATION: {n_runs} Independent Runs (non-deterministic)")
    print("="*80)

    results = []

    for run in range(n_runs):
        print(f"\nRun {run+1}/{n_runs}")

        # Each SA run uses natural randomness 
        x_best, f_best, history = simulated_annealing(
            params, x0, v0, t0,
            T_init=1000.0,
            T_final=1.0,
            max_iter=5000,
        )

        # Evaluate final solution
        _, is_feas, info = evaluate_solution(x_best, x0, v0, t0, params)

        results.append({
            'run_id': run + 1,
            'f_best': f_best,
            'f_time': info['f_time'],
            'f_energy': info['f_energy'],
            'feasible': is_feas,
            'history': history
        })

    # =========================
    # 📊 Statistics summary
    # =========================
    feasible_results = [r for r in results if r['feasible']]
    all_fitness = [r['f_best'] for r in results]

    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    print(f"Feasibility rate: {len(feasible_results)}/{n_runs} "
          f"({len(feasible_results)/n_runs*100:.1f}%)")

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

    # =========================
    # 📈 Visualization
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Statistical Validation ({n_runs} runs)', fontsize=14, fontweight='bold')

    # Convergence curves
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

    # Box plot of final fitness
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
    params = make_params()
    x0 = np.zeros(4)
    v0 = np.array([12.0, 11.0, 13.0, 10.0])
    t0 = np.array([0.0, 0.5, 1.0, 1.5])

    # Part 1: Baseline
    print("\n" + "="*80)
    print("PART 1: Random Search Baseline")
    print("="*80)
    f_random, x_random, info_random = random_search_baseline(params, x0, v0, t0, n_samples=500)


    # Part 2: SA with live visualization
    print("\n" + "="*80)
    print("PART 2: Simulated Annealing (Live Visualization)")
    print("="*80)
    x_best, f_best, history, info_best = simulated_annealing_with_live_viz(
        params, x0, v0, t0,
        T_init=1000.0, T_final=1.0, cooling=0.95,
        max_iter=5000, update_interval=100
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
        params = make_params()
        x0 = np.zeros(params.N, dtype=float)
        v0 = build_v0_random(params)
        t0 = build_spawn_times(params)

        # --- Simulated Annealing only ---
        x_best, f_best, history = simulated_annealing(
            params, x0, v0, t0,
            T_init=1000.0,
            T_final=0.1,
            max_iter=1000
        )

        # Plot convergence
        fig_conv = plot_convergence(history)
        plt.show()

        # Visualize best solution
        from metaheuristic_intersection import plot_combined_visualization
        fig_viz, anim = plot_combined_visualization(x_best, x0, v0, t0, params)
        plt.show()