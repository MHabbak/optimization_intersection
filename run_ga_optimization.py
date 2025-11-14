"""
Complete example: Run intersection optimization with GA

Usage:
    python run_ga_optimization.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the flexible configuration module
from flexible_vehicle_setup import (
    create_vehicle_config,
    print_vehicle_summary,
)

# Import GA optimization and visualization
from ga_intersection import (
    genetic_algorithm,
    plot_ga_convergence,
    GAConfig
)

# Import base functions
from metaheuristic_intersection import (
    plot_combined_visualization,
    feasibility_check,
    print_constraint_summary,
    objective_function
)


def run_ga_scenario(scenario_name, vehicles, config, show_plots=True):
    """
    Run complete GA optimization for a vehicle scenario
    
    Args:
        scenario_name: Name for display
        vehicles: List of vehicle specifications
        config: GAConfig object with all GA parameters
        show_plots: Whether to display plots
    """
    print("\n" + "="*80)
    print(f"SCENARIO: {scenario_name}")
    print("="*80)
    
    # Create configuration using flexible setup
    params, x0, v0, t0, specs = create_vehicle_config(vehicles)
    print_vehicle_summary(params, x0, v0, t0, specs)
    
    # Display GA configuration
    print(f"\nGA Configuration:")
    print(f"  Population size: {config.population_size}")
    print(f"  Max generations: {config.max_generations}")
    print(f"  Elitism: {config.p_elitism*100:.0f}%")
    print(f"  Crossover: {config.p_crossover*100:.0f}%")
    print(f"  Mutation: {config.p_mutation*100:.0f}%")
    
    # Run GA optimization
    print("\n[1/2] Running Genetic Algorithm...")
    x_best, f_best, f_obj, history = genetic_algorithm(
        x0, v0, t0, params, config
    )
    
    # Check final solution
    print("\n[2/2] Validating final solution...")
    is_feasible, violations = feasibility_check(x_best, x0, v0, t0, params)
    print_constraint_summary(violations)
    
    # Aggregate metrics
    _, _, _, info = objective_function(x_best, x0, v0, t0, params)

    print("\nSUMMARY METRICS")
    print("-" * 80)
    print(f"Avg time to reach L (s):      {info['avg_time_to_L']:.3f}")
    print(f"Avg energy per vehicle:       {info['avg_energy_per_vehicle']:.3f}")
    print(f"Combined avg speed (m/s):     {info['avg_speed_all']:.3f}")

    # Display results
    if show_plots:
        print("\nGenerating visualizations...")
        
        # Convergence plot
        fig_conv = plot_ga_convergence(history)
        fig_conv.suptitle(f"{scenario_name} - GA Convergence", fontweight='bold')
        
        # Animated visualization
        fig_anim, anim = plot_combined_visualization(x_best, x0, v0, t0, params)
        fig_anim.suptitle(f"{scenario_name} - Best Solution", fontweight='bold')
        
        plt.show()
    
    return {
        'params': params,
        'x_best': x_best,
        'f_best': f_best,
        'f_obj': f_obj,
        'history': history,
        'is_feasible': is_feasible
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


def scenario_original_4():
    """Original 4-vehicle config"""
    vehicles = [
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        {'direction': 'SN', 'v0': 10.0, 't0': 1.5},
    ]
    return "Original 4-Vehicle", vehicles


def scenario_random_8():
    """Random 8-vehicle configuration"""
    vehicles = [
        {'direction': d, 'v0': np.random.uniform(10, 15), 't0': i*0.6}
        for i, d in enumerate(['EW', 'WE', 'NS', 'SN', 'EW', 'NS', 'WE', 'SN'])
    ]
    return "Random 8-Vehicle", vehicles


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
        return scenario_original_4()
    
    return f"Custom ({len(vehicles)} vehicles)", vehicles


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    print("\n" + "="*80)
    print("FLEXIBLE N-VEHICLE INTERSECTION OPTIMIZATION - GENETIC ALGORITHM")
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
        '5': scenario_original_4,
        '6': scenario_random_8,
        '7': scenario_custom_input,
    }
    
    if choice not in scenarios:
        print("Invalid choice! Using default light traffic scenario.")
        choice = '1'
    
    # Get scenario
    name, vehicles = scenarios[choice]()
    
    # Configure GA parameters
    print("\n" + "="*80)
    print("GA PARAMETER CONFIGURATION")
    print("="*80)
    config = GAConfig()
    
    try:
        print("\nBasic Parameters:")
        pop_size = input(f"  Population size (default {config.population_size}): ").strip()
        if pop_size:
            config.population_size = int(pop_size)
        
        max_gen = input(f"  Max generations (default {config.max_generations}): ").strip()
        if max_gen:
            config.max_generations = int(max_gen)
        
        print("\nOperator Percentages (must sum to 1.0):")
        p_e = input(f"  Elitism percentage (default {config.p_elitism}): ").strip()
        if p_e:
            config.p_elitism = float(p_e)
        
        p_c = input(f"  Crossover percentage (default {config.p_crossover}): ").strip()
        if p_c:
            config.p_crossover = float(p_c)
        
        p_m = input(f"  Mutation percentage (default {config.p_mutation}): ").strip()
        if p_m:
            config.p_mutation = float(p_m)
        
        print("\nAdvanced Parameters (press Enter to keep defaults):")
        alpha = input(f"  Crossover alpha (default {config.crossover_alpha}): ").strip()
        if alpha:
            config.crossover_alpha = float(alpha)
        
        mut_std = input(f"  Mutation std dev (default {config.mutation_std}): ").strip()
        if mut_std:
            config.mutation_std = float(mut_std)
        
        mut_prob = input(f"  Mutation probability (default {config.mutation_probability}): ").strip()
        if mut_prob:
            config.mutation_probability = float(mut_prob)
            
    except ValueError:
        print("\n⚠️  Invalid input detected. Using default configuration.")
        config = GAConfig()  # Reset to defaults
    
    # Validate configuration
    total_percentage = config.p_elitism + config.p_crossover + config.p_mutation
    if abs(total_percentage - 1.0) > 0.01:
        print(f"\n⚠️  Warning: Operator percentages sum to {total_percentage:.2f}, not 1.0")
        print("    Continuing anyway, but results may be unexpected...")
    
    # Run optimization
    result = run_ga_scenario(name, vehicles, config)
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Scenario: {name}")
    print(f"Vehicles: {result['params'].N}")
    print(f"Best objective: {result['f_obj']:.2f}")
    print(f"Feasible: {'✓ YES' if result['is_feasible'] else '✗ NO'}")
    print("="*80)


if __name__ == "__main__":
    main()