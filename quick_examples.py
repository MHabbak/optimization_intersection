"""
Quick Examples: Run optimization with different vehicle counts

This script shows simple copy-paste examples for common scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
from flexible_vehicle_setup import create_vehicle_config, print_vehicle_summary
from sa_intersection import simulated_annealing, plot_convergence
from metaheuristic_intersection import plot_combined_visualization, feasibility_check


# ============================================================================
# EXAMPLE 4: UNBALANCED TRAFFIC (Heavy EW, Light NS)
# ============================================================================

def example_unbalanced_traffic():
    """Heavy traffic in one direction, light in others"""
    
    vehicles = [
        # Heavy EW traffic (5 vehicles)
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'EW', 'v0': 13.0, 't0': 1.0},
        {'direction': 'EW', 'v0': 11.5, 't0': 2.0},
        {'direction': 'EW', 'v0': 12.5, 't0': 3.0},
        {'direction': 'EW', 'v0': 13.5, 't0': 4.0},
        
        # Moderate WE traffic (2 vehicles)
        {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
        {'direction': 'WE', 'v0': 12.0, 't0': 2.5},
        
        # Light cross traffic (2 vehicles total)
        {'direction': 'NS', 'v0': 13.0, 't0': 1.5},
        {'direction': 'SN', 'v0': 10.0, 't0': 3.5},
    ]
    
    params, x0, v0, t0, specs = create_vehicle_config(vehicles)
    print_vehicle_summary(params, x0, v0, t0, specs)
    
    print("\n📊 Traffic Pattern: Heavy East-West corridor with light cross traffic")
    print("   This mimics a major highway with minor intersecting roads")
    
    # Run SA
    x_best, f_best, history = simulated_annealing(
        params, x0, v0, t0,
        T_init=1000.0,
        T_final=0.1,
        max_iter=3000
    )
    
    # Check feasibility
    is_feas, viols = feasibility_check(x_best, x0, v0, t0, params)
    print(f"\nFinal solution: f={f_best:.2f}, feasible={is_feas}")
    
    # Visualize
    plot_convergence(history)
    plot_combined_visualization(x_best, x0, v0, t0, params)
    plt.show()


# ============================================================================
# EXAMPLE 5: RANDOM N VEHICLES
# ============================================================================

def example_random_n_vehicles(n_vehicles=7):
    """Generate random scenario with N vehicles"""
    
    print(f"\n🎲 Generating random scenario with {n_vehicles} vehicles...")
    
    vehicles = []
    directions = ['EW', 'WE', 'NS', 'SN']
    
    for i in range(n_vehicles):
        vehicles.append({
            'direction': np.random.choice(directions),
            'v0': np.random.uniform(10.0, 15.0),
            't0': i * np.random.uniform(0.3, 1.0)
        })
    
    params, x0, v0, t0, specs = create_vehicle_config(vehicles)
    print_vehicle_summary(params, x0, v0, t0, specs)
    
    # Run SA
    x_best, f_best, history = simulated_annealing(
        params, x0, v0, t0,
        T_init=1000.0,
        T_final=0.1,
        max_iter=2500
    )
    
    # Check feasibility
    is_feas, viols = feasibility_check(x_best, x0, v0, t0, params)
    print(f"\nFinal solution: f={f_best:.2f}, feasible={is_feas}")
    
    # Visualize
    plot_convergence(history)
    plot_combined_visualization(x_best, x0, v0, t0, params)
    plt.show()


# ============================================================================
# EXAMPLE 6: CUSTOM MANUAL SPECIFICATION
# ============================================================================

def example_custom_manual():
    """
    Template for manually specifying your own vehicles
    Copy this function and modify the vehicles list!
    """
    
    # 🔧 CUSTOMIZE THIS LIST:
    vehicles = [
        # Format: {'direction': 'EW'|'WE'|'NS'|'SN', 'v0': speed, 't0': time}
        
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        {'direction': 'WE', 'v0': 11.0, 't0': 2.0},
        {'direction': 'SN', 'v0': 14.0, 't0': 3.0},
        {'direction': 'EW', 'v0': 12.5, 't0': 4.0},
        
        # Add more vehicles here...
        # {'direction': 'NS', 'v0': 13.5, 't0': 5.0},
    ]
    
    params, x0, v0, t0, specs = create_vehicle_config(vehicles)
    print_vehicle_summary(params, x0, v0, t0, specs)
    
    # 🔧 CUSTOMIZE SA PARAMETERS:
    x_best, f_best, history = simulated_annealing(
        params, x0, v0, t0,
        T_init=1000.0,      # Initial temperature
        T_final=0.1,        # Final temperature
        max_iter=2000       # Iterations
    )
    
    # Check feasibility
    is_feas, viols = feasibility_check(x_best, x0, v0, t0, params)
    print(f"\nFinal solution: f={f_best:.2f}, feasible={is_feas}")
    
    # Visualize
    plot_convergence(history)
    plot_combined_visualization(x_best, x0, v0, t0, params)
    plt.show()


# ============================================================================
# QUICK COMPARISON: 2 vs 6 vs 12 vehicles
# ============================================================================

def compare_scenarios():
    """Compare optimization performance across different scales"""
    
    scenarios = [
        ("2 vehicles", [
            {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
            {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        ]),
        ("6 vehicles", [
            {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
            {'direction': 'EW', 'v0': 14.0, 't0': 2.5},
            {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
            {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
            {'direction': 'NS', 'v0': 12.0, 't0': 3.0},
            {'direction': 'SN', 'v0': 10.0, 't0': 1.5},
        ]),
        ("12 vehicles", [
            {'direction': d, 'v0': np.random.uniform(11, 14), 't0': i*0.5}
            for i, d in enumerate(['EW', 'EW', 'EW', 'WE', 'WE', 'WE',
                                  'NS', 'NS', 'NS', 'SN', 'SN', 'SN'])
        ])
    ]
    
    results = []
    
    for name, vehicles in scenarios:
        print("\n" + "="*80)
        print(f"Testing: {name}")
        print("="*80)
        
        params, x0, v0, t0, specs = create_vehicle_config(vehicles)
        print_vehicle_summary(params, x0, v0, t0, specs)
        
        # Run SA with fixed iterations for fair comparison
        x_best, f_best, history = simulated_annealing(
            params, x0, v0, t0,
            T_init=1000.0,
            T_final=0.1,
            max_iter=2000
        )
        
        is_feas, _ = feasibility_check(x_best, x0, v0, t0, params)
        
        results.append({
            'name': name,
            'n_vehicles': len(vehicles),
            'f_best': f_best,
            'feasible': is_feas,
            'history': history
        })
        
        print(f"Result: f={f_best:.2f}, feasible={is_feas}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Scenario':<15} {'Vehicles':<10} {'Fitness':<12} {'Feasible':<10}")
    print("-"*80)
    for r in results:
        feas_str = "✓ YES" if r['feasible'] else "✗ NO"
        print(f"{r['name']:<15} {r['n_vehicles']:<10} {r['f_best']:<12.2f} {feas_str:<10}")
    print("="*80)
    
    # Plot comparison
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for ax, r in zip(axes, results):
        ax.plot(r['history']['iteration'], r['history']['f_best'], 
                linewidth=2, color='green' if r['feasible'] else 'red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Fitness')
        ax.set_title(f"{r['name']}\nf={r['f_best']:.1f}", fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN MENU
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUICK EXAMPLES - N-VEHICLE OPTIMIZATION")
    print("="*80)
    print("\nSelect an example to run:")
    print("  1. Two vehicles (simplest)")
    print("  2. Eight vehicles (balanced)")
    print("  3. Twelve vehicles (large scale)")
    print("  4. Unbalanced traffic (heavy EW, light NS)")
    print("  5. Random N vehicles (you choose N)")
    print("  6. Custom manual specification")
    print("  7. Compare 2 vs 6 vs 12 vehicles")
    print("="*80)
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice == '1':
        example_2_vehicles()
    elif choice == '2':
        example_8_vehicles()
    elif choice == '3':
        example_12_vehicles()
    elif choice == '4':
        example_unbalanced_traffic()
    elif choice == '5':
        try:
            n = int(input("How many vehicles? (e.g., 7): "))
            example_random_n_vehicles(n)
        except ValueError:
            print("Invalid input, using 7 vehicles")
            example_random_n_vehicles(7)
    elif choice == '6':
        example_custom_manual()
    elif choice == '7':
        compare_scenarios()
    else:
        print("Invalid choice! Running default 2-vehicle example...")
        example_2_vehicles()
#5============================================================================
# EXAMPLE 1: TWO VEHICLES (SIMPLEST)
# ============================================================================

def example_2_vehicles():
    """Just 2 vehicles crossing - simplest case"""
    
    vehicles = [
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
    ]
    
    params, x0, v0, t0, specs = create_vehicle_config(vehicles)
    print_vehicle_summary(params, x0, v0, t0, specs)
    
    # Run SA
    x_best, f_best, history = simulated_annealing(
        params, x0, v0, t0,
        T_init=500.0,
        T_final=0.1,
        max_iter=1000
    )
    
    # Check feasibility
    is_feas, viols = feasibility_check(x_best, x0, v0, t0, params)
    print(f"\nFinal solution: f={f_best:.2f}, feasible={is_feas}")
    
    # Visualize
    plot_convergence(history)
    plot_combined_visualization(x_best, x0, v0, t0, params)
    plt.show()


# ============================================================================
# EXAMPLE 2: EIGHT VEHICLES
# ============================================================================

def example_8_vehicles():
    """8 vehicles with staggered arrivals"""
    
    vehicles = [
        # EW direction (2 vehicles)
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'EW', 'v0': 13.5, 't0': 2.0},
        
        # WE direction (2 vehicles)
        {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
        {'direction': 'WE', 'v0': 12.5, 't0': 2.5},
        
        # NS direction (2 vehicles)
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        {'direction': 'NS', 'v0': 14.0, 't0': 3.0},
        
        # SN direction (2 vehicles)
        {'direction': 'SN', 'v0': 10.0, 't0': 1.5},
        {'direction': 'SN', 'v0': 11.5, 't0': 3.5},
    ]
    
    params, x0, v0, t0, specs = create_vehicle_config(vehicles)
    print_vehicle_summary(params, x0, v0, t0, specs)
    
    # Run SA with more iterations for harder problem
    x_best, f_best, history = simulated_annealing(
        params, x0, v0, t0,
        T_init=1000.0,
        T_final=0.1,
        max_iter=3000
    )
    
    # Check feasibility
    is_feas, viols = feasibility_check(x_best, x0, v0, t0, params)
    print(f"\nFinal solution: f={f_best:.2f}, feasible={is_feas}")
    
    # Visualize
    plot_convergence(history)
    plot_combined_visualization(x_best, x0, v0, t0, params)
    plt.show()


# ============================================================================
# EXAMPLE 3: TWELVE VEHICLES (LARGE SCALE)
# ============================================================================

def example_12_vehicles():
    """12 vehicles - large scale scenario"""
    
    vehicles = [
        # EW: 3 vehicles
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'EW', 'v0': 13.0, 't0': 2.0},
        {'direction': 'EW', 'v0': 11.5, 't0': 4.0},
        
        # WE: 3 vehicles
        {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
        {'direction': 'WE', 'v0': 12.5, 't0': 2.5},
        {'direction': 'WE', 'v0': 13.5, 't0': 4.5},
        
        # NS: 3 vehicles
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        {'direction': 'NS', 'v0': 14.0, 't0': 3.0},
        {'direction': 'NS', 'v0': 12.5, 't0': 5.0},
        
        # SN: 3 vehicles
        {'direction': 'SN', 'v0': 10.0, 't0': 1.5},
        {'direction': 'SN', 'v0': 11.5, 't0': 3.5},
        {'direction': 'SN', 'v0': 13.0, 't0': 5.5},
    ]
    
    params, x0, v0, t0, specs = create_vehicle_config(vehicles)
    print_vehicle_summary(params, x0, v0, t0, specs)
    
    # Run SA with even more iterations
    x_best, f_best, history = simulated_annealing(
        params, x0, v0, t0,
        T_init=1500.0,
        T_final=0.1,
        max_iter=5000
    )
    
    # Check feasibility
    is_feas, viols = feasibility_check(x_best, x0, v0, t0, params)
    print(f"\nFinal solution: f={f_best:.2f}, feasible={is_feas}")
    
    # Visualize
    plot_convergence(history)
    plot_combined_visualization(x_best, x0, v0, t0, params)
    plt.show()


# 