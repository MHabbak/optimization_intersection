"""
Complete example: Run PSO intersection optimization with any number of vehicles

Usage:
    python run_flexible_pso.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the flexible configuration module
from flexible_vehicle_setup import (
    create_vehicle_config,
    print_vehicle_summary,
    config_4_way_balanced,
    config_6_vehicles,
    config_10_vehicles,
    config_custom
)

# Import PSO optimization and visualization
from hybrid_pso_intersection import (
    HybridPSO,
    plot_pso_convergence
)

from metaheuristic_intersection import (
    plot_combined_visualization,
    feasibility_check,
    print_constraint_summary,
    objective_function
)


def run_pso_optimization_scenario(scenario_name, vehicles, 
                                   swarm_size=30, max_iter=100, 
                                   show_plots=True):
    """
    Run complete PSO optimization for a vehicle scenario
    
    Args:
        scenario_name: Name for display
        vehicles: List of vehicle specifications
        swarm_size: Number of particles in swarm
        max_iter: PSO iterations
        show_plots: Whether to display plots
    """
    print("\n" + "="*80)
    print(f"SCENARIO: {scenario_name}")
    print("="*80)
    
    # Create configuration
    params, x0, v0, t0, specs = create_vehicle_config(vehicles)
    print_vehicle_summary(params, x0, v0, t0, specs)
    
    # Adjust time horizon based on number of vehicles
    # More vehicles = need more time
    if params.N <= 4:
        params.K = 100
    elif params.N <= 8:
        params.K = 120
    else:
        params.K = 150
    
    params.T_max = params.K * params.dt
    
    print(f"\nTime Horizon: K={params.K} steps × dt={params.dt}s = {params.T_max}s")
    print(f"Decision Variables: {params.N * params.K} (accel) + "
          f"{np.sum(params.get_conflict_matrix())//2} (binary) = "
          f"{params.N * params.K + np.sum(params.get_conflict_matrix())//2}")
    
    # Adjust PSO parameters based on problem size
    if params.N <= 4:
        default_swarm = 100
        default_iter = 200
    elif params.N <= 8:
        default_swarm = 100
        default_iter = 200
    else:
        default_swarm = 100
        default_iter = 200
    
    swarm_size = swarm_size or default_swarm
    max_iter = max_iter or default_iter
    
    print(f"PSO Parameters: Swarm={swarm_size}, Iterations={max_iter}")
    
    # Create PSO optimizer
    print("\n[1/2] Running Particle Swarm Optimization...")
    pso = HybridPSO(
        params=params,
        x0=x0, v0=v0, t0=t0,
        swarm_size=swarm_size,
        max_iter=max_iter,
        w_start=0.9, w_end=0.2,
        c1=2.0, c2=2.0
    )
    
    # Run optimization
    x_best, f_best, history = pso.optimize()
    
    if x_best is None:
        print("\n❌ PSO failed to find a feasible solution!")
        print("Try increasing swarm_size or max_iter.")
        return None
    
    # Check final solution
    print("\n[2/2] Validating final solution...")
    is_feasible, violations = feasibility_check(x_best, x0, v0, t0, params)
    
    # Get detailed info
    _, _, _, info = objective_function(x_best, x0, v0, t0, params)
    
    print_constraint_summary(violations)
    
    print("\n" + "="*80)
    print("FINAL SOLUTION METRICS")
    print("="*80)
    print(f"✓ Feasible:        {'YES ✅' if is_feasible else 'NO ❌'}")
    print(f"✓ Total Objective: {f_best:.4f}")
    print(f"✓ Total Time:      {info['f_time']:.2f} s")
    print(f"✓ Total Energy:    {info['f_energy']:.2f}")
    print(f"✓ Avg Crossing:    {info['avg_crossing_time']:.2f} s/vehicle")
    print(f"✓ All Completed:   {info['all_completed']}")
    
    if info['all_completed']:
        print(f"\nPer-Vehicle Travel Times:")
        for i in range(params.N):
            direction = params.get_vehicle_direction(i)
            print(f"  V{i} ({direction}): {info['travel_times'][i]:.2f}s")
    
    # Display results
    if show_plots:
        print("\nGenerating visualizations...")
        
        # Convergence plot
        fig_conv = plot_pso_convergence(history)
        fig_conv.suptitle(f"{scenario_name} - PSO Convergence", 
                         fontweight='bold', fontsize=14)
        
        # Animated visualization
        fig_anim, anim = plot_combined_visualization(x_best, x0, v0, t0, params)
        fig_anim.suptitle(f"{scenario_name} - Optimized Solution", 
                         fontweight='bold', fontsize=14)
        
        plt.show()
    
    return {
        'params': params,
        'x_best': x_best,
        'f_best': f_best,
        'history': history,
        'is_feasible': is_feasible,
        'info': info
    }


# ============================================================================
# PRE-DEFINED SCENARIOS
# ============================================================================

def scenario_minimal():
    """2 vehicles - simplest case"""
    vehicles = [
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
    ]
    return "Minimal (2 vehicles)", vehicles


def scenario_light_traffic():
    """4 vehicles - standard case"""
    vehicles = [
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        {'direction': 'SN', 'v0': 10.0, 't0': 1.5},
    ]
    return "Light Traffic (4 vehicles)", vehicles


def scenario_moderate_traffic():
    """6 vehicles - moderate complexity"""
    vehicles = [
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'EW', 'v0': 14.0, 't0': 2.5},
        {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        {'direction': 'NS', 'v0': 12.0, 't0': 3.0},
        {'direction': 'SN', 'v0': 10.0, 't0': 1.5},
    ]
    return "Moderate Traffic (6 vehicles)", vehicles


def scenario_heavy_eastwest():
    """8 vehicles - heavy EW/WE traffic"""
    vehicles = [
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'EW', 'v0': 13.5, 't0': 1.5},
        {'direction': 'EW', 'v0': 11.5, 't0': 3.0},
        {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
        {'direction': 'WE', 'v0': 12.5, 't0': 2.0},
        {'direction': 'WE', 'v0': 13.0, 't0': 3.5},
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        {'direction': 'SN', 'v0': 10.0, 't0': 2.5},
    ]
    return "Heavy East-West (8 vehicles)", vehicles


def scenario_heavy_traffic():
    """10 vehicles - challenging"""
    vehicles = [
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'EW', 'v0': 13.0, 't0': 2.0},
        {'direction': 'EW', 'v0': 11.5, 't0': 4.0},
        {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
        {'direction': 'WE', 'v0': 12.5, 't0': 2.5},
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        {'direction': 'NS', 'v0': 14.0, 't0': 3.0},
        {'direction': 'SN', 'v0': 10.0, 't0': 1.5},
        {'direction': 'SN', 'v0': 11.5, 't0': 3.5},
        {'direction': 'SN', 'v0': 13.0, 't0': 5.0},
    ]
    return "Heavy Traffic (10 vehicles)", vehicles


def scenario_rush_hour():
    """12 vehicles - very challenging"""
    vehicles = []
    t_current = 0.0
    
    # Simulate rush hour: vehicles arriving frequently
    directions = ['EW', 'WE', 'NS', 'SN', 'EW', 'NS', 'WE', 'SN', 
                  'EW', 'NS', 'WE', 'SN']
    
    for direction in directions:
        v0 = np.random.uniform(10.5, 14.5)
        vehicles.append({
            'direction': direction,
            'v0': v0,
            't0': t_current
        })
        t_current += np.random.uniform(0.4, 0.9)
    
    return "Rush Hour (12 vehicles)", vehicles


def scenario_random_n(n_vehicles=8):
    """Random configuration with N vehicles"""
    vehicles = []
    directions = ['EW', 'WE', 'NS', 'SN']
    t_current = 0.0
    
    for i in range(n_vehicles):
        direction = directions[i % 4]  # Cycle through directions
        v0 = np.random.uniform(10.0, 15.0)
        vehicles.append({
            'direction': direction,
            'v0': v0,
            't0': t_current
        })
        t_current += np.random.uniform(0.3, 1.0)
    
    return f"Random {n_vehicles}-Vehicle", vehicles


def scenario_custom_input():
    """Interactive: user specifies vehicles"""
    print("\n" + "="*80)
    print("CUSTOM SCENARIO BUILDER")
    print("="*80)
    print("Build your own vehicle configuration interactively.")
    print("=" *80)
    
    vehicles = []
    
    while True:
        print(f"\n--- Vehicle {len(vehicles) + 1} ---")
        print("Directions: EW (East→West), WE (West→East), NS (North→South), SN (South→North)")
        
        direction = input("Enter direction (or 'done' to finish): ").strip().upper()
        
        if direction == 'DONE':
            break
        
        if direction not in ['EW', 'WE', 'NS', 'SN']:
            print("❌ Invalid direction! Use EW, WE, NS, or SN")
            continue
        
        try:
            v0 = float(input("Initial velocity (m/s, e.g., 12.0): "))
            if v0 < 1.0 or v0 > 20.0:
                print("⚠️  Warning: velocity outside typical range [1, 20] m/s")
            
            t0 = float(input("Arrival time (s, e.g., 0.0): "))
            if t0 < 0:
                print("⚠️  Warning: negative arrival time")
            
            vehicles.append({
                'direction': direction,
                'v0': v0,
                't0': t0
            })
            
            print(f"✓ Added V{len(vehicles)-1}: {direction}, v={v0:.1f}m/s, t={t0:.1f}s")
            
        except ValueError:
            print("❌ Invalid input! Please enter numeric values.")
    
    if len(vehicles) == 0:
        print("No vehicles added! Using default 4-vehicle config.")
        return scenario_light_traffic()
    
    return f"Custom ({len(vehicles)} vehicles)", vehicles


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    print("\n" + "="*80)
    print("FLEXIBLE N-VEHICLE INTERSECTION OPTIMIZATION WITH PSO")
    print("="*80)
    print("\nSelect a scenario:")
    print("  1. Minimal (2 vehicles) - Quick test")
    print("  2. Light Traffic (4 vehicles) - Standard case")
    print("  3. Moderate Traffic (6 vehicles) - Balanced")
    print("  4. Heavy East-West (8 vehicles) - Asymmetric load")
    print("  5. Heavy Traffic (10 vehicles) - Challenging")
    print("  6. Rush Hour (12 vehicles) - Very challenging")
    print("  7. Random N vehicles (specify N)")
    print("  8. Custom input (build your own)")
    print("="*80)
    
    choice = input("\nEnter choice (1-8): ").strip()
    
    # Handle choice 7 (random N)
    if choice == '7':
        try:
            n = int(input("How many vehicles (2-20)? "))
            if n < 2 or n > 20:
                print("Using default N=8")
                n = 8
            name, vehicles = scenario_random_n(n)
        except ValueError:
            print("Invalid input, using N=8")
            name, vehicles = scenario_random_n(8)
    else:
        # Map other choices to scenarios
        scenarios = {
            '1': scenario_minimal,
            '2': scenario_light_traffic,
            '3': scenario_moderate_traffic,
            '4': scenario_heavy_eastwest,
            '5': scenario_heavy_traffic,
            '6': scenario_rush_hour,
            '8': scenario_custom_input,
        }
        
        if choice not in scenarios:
            print("Invalid choice! Using default light traffic scenario.")
            choice = '2'
        
        name, vehicles = scenarios[choice]()
    
    # Ask for PSO parameters
    print("\n" + "="*80)
    print("PSO CONFIGURATION")
    print("="*80)
    
    try:
        swarm_input = input("Swarm size (default: auto-select based on N): ").strip()
        swarm_size = int(swarm_input) if swarm_input else None
    except ValueError:
        swarm_size = None
    
    try:
        iter_input = input("Max iterations (default: auto-select based on N): ").strip()
        max_iter = int(iter_input) if iter_input else None
    except ValueError:
        max_iter = None
    
    # Run optimization
    result = run_pso_optimization_scenario(
        name, vehicles, 
        swarm_size=swarm_size, 
        max_iter=max_iter
    )
    
    if result is None:
        print("\n❌ Optimization failed!")
        return
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Scenario:       {name}")
    print(f"Vehicles:       {result['params'].N}")
    print(f"Best Objective: {result['f_best']:.4f}")
    print(f"Total Time:     {result['info']['f_time']:.2f} s")
    print(f"Total Energy:   {result['info']['f_energy']:.2f}")
    print(f"Feasible:       {'✓ YES' if result['is_feasible'] else '✗ NO'}")
    print("="*80)
    
    # Option to save results
    save = input("\nSave solution to file? (y/n): ").strip().lower()
    if save == 'y':
        filename = input("Filename (e.g., solution.npy): ").strip()
        if not filename:
            filename = f"pso_solution_{result['params'].N}vehicles.npy"
        
        np.save(filename, {
            'x_best': result['x_best'],
            'params': result['params'],
            'x0': result['params'].x0 if hasattr(result['params'], 'x0') else None,
            'info': result['info']
        })
        print(f"✓ Saved to {filename}")


if __name__ == "__main__":
    main()