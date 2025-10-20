"""
Test script for corrected SA implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from metaheuristic_intersection import ProblemParameters
from sa_intersection import simulated_annealing, plot_convergence, random_search_baseline

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING CORRECTED SA IMPLEMENTATION")
    print("="*80)

    # Setup parameters
    params = ProblemParameters()
    x0 = np.zeros(4)
    v0 = np.array([12.0, 11.0, 13.0, 10.0])
    t0 = np.array([0.0, 0.5, 1.0, 1.5])

    print(f"\nProblem Setup:")
    print(f"  Time horizon: K={params.K} steps, T_max={params.T_max}s")
    print(f"  Vehicles: {params.N}")
    print(f"  Decision variables: {params.N * params.K + 4}")

    # Optional: Run baseline
    print("\n" + "="*80)
    choice = input("Run random search baseline first? (y/n): ").strip().lower()
    if choice == 'y':
        x_random, f_random, feas_random = random_search_baseline(
            params, x0, v0, t0, n_samples=500, seed=42
        )

    # Run corrected SA
    print("\n" + "="*80)
    print("Running Corrected SA...")
    print("="*80)

    x_best, f_best, history = simulated_annealing(
        params, x0, v0, t0,
        T_init=100.0,      # Lower temp = more selective
        T_final=0.01,
        max_iter=5000,
        seed=42
    )

    # Plot convergence
    print("\n" + "="*80)
    print("Generating convergence plots...")
    print("="*80)

    fig = plot_convergence(history)
    plt.savefig('/home/boko/Uni/Optimization/sa_convergence_corrected.png', dpi=150, bbox_inches='tight')
    print("✅ Saved convergence plot to sa_convergence_corrected.png")
    plt.show()

    # Visualize best solution
    from metaheuristic_intersection import plot_combined_visualization

    print("\nGenerating animation of best solution...")
    fig_viz, anim = plot_combined_visualization(x_best, x0, v0, t0, params)
    print("✅ Animation created - showing now...")
    plt.show()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  Best fitness: {f_best:.2f}")
    print(f"  Iterations: {len(history['iteration'])}")
    print(f"  Acceptance rate: {history['acceptance_rate'][-1]:.2%}")
    print(f"  Repair rate: {history['repair_rate'][-1]:.2%}")
    print(f"  Skip rate: {history['skip_rate'][-1]:.2%}")
    print("="*80)
