"""
Complete example: Run intersection optimization with any number of vehicles

Usage:
    python run_flexible_optimization.py
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

# Import optimization and visualization
from sa_intersection import (
    simulated_annealing,
    plot_convergence,
    random_search_baseline
)

from metaheuristic_intersection import (
    plot_combined_visualization,
    feasibility_check,
    print_constraint_summary
)


def run_optimization_scenario(scenario_name, vehicles, 
                              max_iter=2000, show_plots=True):
    """
    Run complete optimization for a vehicle scenario
    
    Args:
        scenario_name: Name for display
        vehicles: List of vehicle specifications
        max_iter: SA iterations
        show_plots: Whether to display plots
    """
    print("\n" + "="*80)
    print(f"SCENARIO: {scenario_name}")
    print("="*80)
    
    # Create configuration
    params, x0, v0, t0, specs = create_vehicle_config(vehicles)
    print_vehicle_summary(params, x0, v0, t0, specs)
    
    # Run baseline (optional, comment out for speed)
    print("\n[1/3] Running random search baseline...")
    x_random, f_random, feas_random = random_search_baseline(
        params, x0, v0, t0, n_samples=200
    )
    
    # Run SA optimization
    print("\n[2/3] Running Simulated Annealing...")
    x_best, f_best, history = simulated_annealing(
        params, x0, v0, t0,
        T_init=1000.0,
        T_final=0.1,
        max_iter=max_iter
    )
    
    # Check final solution
    print("\n[3/3] Validating final solution...")
    is_feasible, violations = feasibility_check(x_best, x0, v0, t0, params)
    print_constraint_summary(violations)
    
    # Display results
    if show_plots:
        print("\nGenerating visualizations...")
        
        # Convergence plot
        fig_conv = plot_convergence(history)
        fig_conv.suptitle(f"{scenario_name} - Convergence", fontweight='bold')
        
        # Animated visualization
        fig_anim, anim = plot_combined_visualization(x_best, x0, v0, t0, params)
        fig_anim.suptitle(f"{scenario_name} - Solution", fontweight='bold')
        
        plt.show()
    
    return {
        'params': params,
        'x_best': x_best,
        'f_best': f_best,
        'history': history,
        'is_feasible': is_feasible,
        'baseline': f_random if feas_random else None
    }


# ============================================================================
# PRE-DEFINED SCENARIOS
# ============================================================================

def scenario_light_traffic():
    """2 vehicles - simplest case"""
    vehicles = [
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
    ]
    return "Light Traffic (2 vehicles)", vehicles


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
    """15 vehicles - very challenging"""
    vehicles = []
    t_current = 0.0
    
    # Simulate rush hour: vehicles arriving frequently
    directions = ['EW', 'WE', 'NS', 'SN', 'EW', 'NS', 'WE', 'SN', 
                  'EW', 'NS', 'WE', 'SN', 'EW', 'NS', 'SN']
    
    for direction in directions:
        v0 = np.random.uniform(10.5, 14.5)
        vehicles.append({
            'direction': direction,
            'v0': v0,
            't0': t_current
        })
        t_current += np.random.uniform(0.3, 0.8)
    
    return "Rush Hour (15 vehicles)", vehicles


def scenario_custom_input():
    """Interactive: user specifies vehicles"""
    print("\n" + "="*80)
    print("CUSTOM SCENARIO BUILDER")
    print("="*80)
    
    vehicles = []
    
    while True:
        print(f"\nVehicle {len(vehicles) + 1}:")
        print("  Directions: EW (East→West), WE (West→East), NS (North→South), SN (South→North)")
        
        direction = input("  Enter direction (or 'done' to finish): ").strip().upper()
        
        if direction == 'DONE':
            break
        
        if direction not in ['EW', 'WE', 'NS', 'SN']:
            print("  Invalid direction! Use EW, WE, NS, or SN")
            continue
        
        try:
            v0 = float(input("  Initial velocity (m/s, e.g., 12.0): "))
            t0 = float(input("  Arrival time (s, e.g., 0.0): "))
            
            vehicles.append({
                'direction': direction,
                'v0': v0,
                't0': t0
            })
            
            print(f"  ✓ Added V{len(vehicles)-1}: {direction}, v={v0}m/s, t={t0}s")
            
        except ValueError:
            print("  Invalid input! Please enter numeric values.")
    
    if len(vehicles) == 0:
        print("No vehicles added! Using default 4-vehicle config.")
        return scenario_light_traffic()
    
    return f"Custom ({len(vehicles)} vehicles)", vehicles


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    print("\n" + "="*80)
    print("FLEXIBLE N-VEHICLE INTERSECTION OPTIMIZATION")
    print("="*80)
    print("\nSelect a scenario:")
    print("  1. Light Traffic (2 vehicles) - Quick test")
    print("  2. Moderate Traffic (6 vehicles) - Balanced complexity")
    print("  3. Heavy Traffic (10 vehicles) - Challenging")
    print("  4. Rush Hour (15 vehicles) - Very challenging")
    print("  5. Original 4-vehicle config")
    print("  6. Random 8-vehicle config")
    print("  7. Custom input (specify your own vehicles)")
    print("="*80)
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    # Map choices to scenarios
    scenarios = {
        '1': scenario_light_traffic,
        '2': scenario_moderate_traffic,
        '3': scenario_heavy_traffic,
        '4': scenario_rush_hour,
        '5': lambda: ("Original 4-Vehicle", [
            {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
            {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
            {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
            {'direction': 'SN', 'v0': 10.0, 't0': 1.5},
        ]),
        '6': lambda: ("Random 8-Vehicle", [
            {'direction': d, 'v0': np.random.uniform(10, 15), 't0': i*0.6}
            for i, d in enumerate(['EW', 'WE', 'NS', 'SN', 'EW', 'NS', 'WE', 'SN'])
        ]),
        '7': scenario_custom_input,
    }
    
    if choice not in scenarios:
        print("Invalid choice! Using default light traffic scenario.")
        choice = '1'
    
    # Get scenario
    name, vehicles = scenarios[choice]()
    
    # Ask for iteration count
    try:
        max_iter = int(input("\nMax iterations for SA (default 2000): ") or "2000")
    except ValueError:
        max_iter = 2000
    
    # Run optimization
    result = run_optimization_scenario(name, vehicles, max_iter=max_iter)
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Scenario: {name}")
    print(f"Vehicles: {result['params'].N}")
    print(f"Best fitness: {result['f_best']:.2f}")
    print(f"Feasible: {'✓ YES' if result['is_feasible'] else '✗ NO'}")
    if result['baseline']:
        improvement = (result['baseline'] - result['f_best']) / result['baseline'] * 100
        print(f"Improvement over random: {improvement:.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()
