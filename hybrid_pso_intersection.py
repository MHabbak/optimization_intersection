import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy

# Import problem definitions
from metaheuristic_intersection import (
    ProblemParameters, 
    objective_function, 
    feasibility_check, 
    plot_combined_visualization
)
from flexible_vehicle_setup import create_vehicle_config, config_4_way_balanced

class HybridPSO:
    def __init__(self, params: ProblemParameters, x0, v0, t0, 
                 swarm_size=50, max_iter=100, 
                 w_start=0.9, w_end=0.4, 
                 c1=2.0, c2=2.0):
        """
        Hybrid Discrete-Continuous PSO
        """
        self.params = params
        self.x0 = x0
        self.v0 = v0
        self.t0 = t0
        
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        
        # PSO Hyperparameters
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        
        # Dimensions
        self.dim_acc = params.N * params.K
        self.num_conflicts = np.sum(params.get_conflict_matrix()) // 2
        self.dim_bin = self.num_conflicts
        self.dim_total = self.dim_acc + self.dim_bin
        
        # Boundaries
        self.u_min = params.u_min
        self.u_max = params.u_max
        self.v_max_clamp = (params.u_max - params.u_min) * 0.2  # Clamp velocity to 20% of range

    def sigmoid(self, x):
        """Sigmoid transfer function for binary PSO"""
        # Clip x to prevent overflow/underflow
        x = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-x))

    def optimize(self):
        print(f"Initializing Hybrid PSO with {self.swarm_size} particles...")
        print(f"Dimensions: {self.dim_acc} (Continuous Acc) + {self.dim_bin} (Binary Z)")

        # 1. Initialization
        # Position X = [X_acc (continuous), X_bin (binary)]
        # Velocity V = [V_acc, V_bin]
        
        # Random initialize acceleration within bounds (using smaller initial range for stability)
        X_acc = np.random.uniform(self.u_min * 0.3, self.u_max * 0.3, 
                                 (self.swarm_size, self.dim_acc))
        
        # Random initialize binary Z (0 or 1)
        X_bin = np.random.randint(0, 2, (self.swarm_size, self.dim_bin)).astype(float)
        
        # Initialize velocities to 0
        V_acc = np.zeros((self.swarm_size, self.dim_acc))
        V_bin = np.zeros((self.swarm_size, self.dim_bin))

        # Personal Best (Pbest)
        Pbest_acc = X_acc.copy()
        Pbest_bin = X_bin.copy()
        Pbest_scores = np.full(self.swarm_size, np.inf)
        
        # Global Best (Gbest)
        Gbest_acc = None
        Gbest_bin = None
        Gbest_score = np.inf
        Gbest_info = None

        history = []
        feasible_count_history = []
        best_time_history = []
        best_energy_history = []

        start_time = time.time()

        # 2. Main Loop
        for it in range(self.max_iter):
            # Dynamic inertia weight (Linear Decay)
            w = self.w_start - (self.w_start - self.w_end) * (it / self.max_iter)
            
            feasible_in_iter = 0
            
            # --- Evaluation Step ---
            for i in range(self.swarm_size):
                # Combine solution components
                x_full = np.concatenate([X_acc[i], X_bin[i]])
                
                # --- Evaluate Objective First (it includes completion check) ---
                try:
                    fitness, f_time, f_energy, info = objective_function(
                        x_full, self.x0, self.v0, self.t0, self.params
                    )
                    
                    # Check if solution completed successfully
                    if not info.get('all_completed', False):
                        fitness = 1e9  # Heavy penalty for incomplete solutions
                        is_feasible = False
                    else:
                        # Now check detailed constraints only if completed
                        is_feasible, _ = feasibility_check(x_full, self.x0, self.v0, self.t0, self.params)
                        
                        if not is_feasible:
                            fitness = 1e8  # Still penalize but less than incomplete
                        else:
                            feasible_in_iter += 1
                            
                except Exception as e:
                    # Handle any simulation errors
                    print(f"  Warning: Particle {i} failed evaluation: {str(e)[:50]}")
                    fitness = 1e9
                    is_feasible = False
                    f_time = np.inf
                    f_energy = np.inf
                    info = {}
                
                # Update Pbest
                if fitness < Pbest_scores[i]:
                    Pbest_scores[i] = fitness
                    Pbest_acc[i] = X_acc[i].copy()
                    Pbest_bin[i] = X_bin[i].copy()
                    
                    # Update Gbest
                    if fitness < Gbest_score:
                        Gbest_score = fitness
                        Gbest_acc = X_acc[i].copy()
                        Gbest_bin = X_bin[i].copy()
                        Gbest_info = info
                        
                        # Store component values for tracking
                        if is_feasible:
                            best_time_history.append(f_time)
                            best_energy_history.append(f_energy)
                            
                            print(f"  🎯 New Gbest at Iter {it}: {Gbest_score:.4f} "
                                  f"(Time: {f_time:.2f}s, Energy: {f_energy:.2f})")

            # --- Update Step ---
            if Gbest_acc is not None:  # Only update if we have a global best
                r1 = np.random.rand(self.swarm_size, self.dim_acc)
                r2 = np.random.rand(self.swarm_size, self.dim_acc)
                
                # 1. Update Continuous Part (Acceleration)
                V_acc = (w * V_acc + 
                         self.c1 * r1 * (Pbest_acc - X_acc) + 
                         self.c2 * r2 * (Gbest_acc - X_acc))
                
                # Clamp velocities
                V_acc = np.clip(V_acc, -self.v_max_clamp, self.v_max_clamp)
                
                # Update Position
                X_acc = X_acc + V_acc
                
                # Boundary handling for acceleration (Hard limit)
                X_acc = np.clip(X_acc, self.u_min, self.u_max)

                # 2. Update Discrete Part (Binary Z)
                r1_bin = np.random.rand(self.swarm_size, self.dim_bin)
                r2_bin = np.random.rand(self.swarm_size, self.dim_bin)
                
                # Standard velocity update
                V_bin = (w * V_bin + 
                         self.c1 * r1_bin * (Pbest_bin - X_bin) + 
                         self.c2 * r2_bin * (Gbest_bin - X_bin))
                
                # Sigmoid transfer for position update
                sigmoid_v = self.sigmoid(V_bin)
                rho = np.random.rand(self.swarm_size, self.dim_bin)
                
                # If sigmoid(v) > rand, set to 1, else 0
                X_bin = np.where(rho < sigmoid_v, 1.0, 0.0)

            # Logging
            history.append(Gbest_score)
            feasible_count_history.append(feasible_in_iter)
            
            if it % 1 == 0:
                feas_pct = 100 * feasible_in_iter / self.swarm_size
                print(f"Iter {it:3d}/{self.max_iter} | Best: {Gbest_score:.4f} | "
                      f"Feasible: {feasible_in_iter:2d}/{self.swarm_size} ({feas_pct:.0f}%) | w: {w:.3f}")

        elapsed = time.time() - start_time
        print(f"\n✅ Optimization Finished in {elapsed:.2f}s")
        
        if Gbest_acc is None:
            print("❌ WARNING: No feasible solution found!")
            return None, np.inf, {
                'cost': history, 
                'feasible': feasible_count_history,
                'time': [],
                'energy': []
            }
            
        x_final = np.concatenate([Gbest_acc, Gbest_bin])
        
        return x_final, Gbest_score, {
            'cost': history,
            'feasible': feasible_count_history,
            'time': best_time_history,
            'energy': best_energy_history,
            'final_info': Gbest_info
        }

def plot_pso_convergence(history_dict):
    """Plot comprehensive convergence history"""
    cost_history = history_dict['cost']
    feasible_history = history_dict['feasible']
    time_history = history_dict.get('time', [])
    energy_history = history_dict.get('energy', [])
    
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Cost convergence
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(cost_history, 'b-', linewidth=2)
    ax1.set_ylabel('Objective Cost')
    ax1.set_xlabel('Iteration')
    ax1.set_yscale('log')
    ax1.set_title('Overall Cost Convergence')
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    
    # 2. Feasibility count
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(feasible_history, 'g-', linewidth=2)
    ax2.set_ylabel('# Feasible Particles')
    ax2.set_xlabel('Iteration')
    ax2.set_title('Swarm Feasibility Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Time component
    if len(time_history) > 0:
        ax3 = plt.subplot(2, 2, 3)
        iterations = np.arange(len(time_history))
        ax3.plot(iterations, time_history, 'r-', linewidth=2, marker='o', markersize=4)
        ax3.set_ylabel('Total Time (s)')
        ax3.set_xlabel('Improvement Event')
        ax3.set_title('Best Solution Time Component')
        ax3.grid(True, alpha=0.3)
    
    # 4. Energy component
    if len(energy_history) > 0:
        ax4 = plt.subplot(2, 2, 4)
        iterations = np.arange(len(energy_history))
        ax4.plot(iterations, energy_history, 'm-', linewidth=2, marker='s', markersize=4)
        ax4.set_ylabel('Total Energy')
        ax4.set_xlabel('Improvement Event')
        ax4.set_title('Best Solution Energy Component')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # 1. Setup Problem
    params, x0, v0, t0, specs = config_4_way_balanced()
    
    # Use moderate horizon - balance between solution quality and computation
    params.K = 100
    params.T_max = params.K * params.dt
    
    print("="*60)
    print("HYBRID DISCRETE PSO FOR INTERSECTION CONTROL")
    print("Constraint Handling: Hard Rejection")
    print("="*60)
    print(f"Problem Size: {params.N} vehicles, K={params.K} timesteps")
    print(f"Time horizon: {params.T_max}s with dt={params.dt}s")
    print("="*60 + "\n")
    
    # 2. Configure PSO
    pso = HybridPSO(
        params=params,
        x0=x0, v0=v0, t0=t0,
        swarm_size=100,       # Moderate swarm size for balance
        max_iter=200,        # Iterations
        w_start=0.9, w_end=0.7,
        c1=2.0, c2=2.0
    )
    
    # 3. Run Optimization
    x_best, f_best, history = pso.optimize()
    
    if x_best is not None:
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        
        # 4. Verify final solution
        is_feasible, violations = feasibility_check(x_best, x0, v0, t0, params)
        _, _, _, info = objective_function(x_best, x0, v0, t0, params)
        
        print(f"✓ Feasible: {'YES ✅' if is_feasible else 'NO ❌'}")
        print(f"✓ Total Objective: {f_best:.4f}")
        print(f"✓ Total Time:      {info['f_time']:.2f} s")
        print(f"✓ Total Energy:    {info['f_energy']:.2f}")
        print(f"✓ Avg Crossing:    {info['avg_crossing_time']:.2f} s")
        print(f"✓ All Completed:   {info['all_completed']}")
        
        if not is_feasible:
            print("\n⚠️  Solution violates constraints:")
            for key, result in violations.items():
                if not result['satisfied']:
                    print(f"   - {key}: {len(result['violations'])} violations")
        
        # 5. Visualize convergence
        fig_conv = plot_pso_convergence(history)
        
        # 6. Visualize trajectories
        fig_traj, anim = plot_combined_visualization(x_best, x0, v0, t0, params)
        
        plt.show()
    else:
        print("\n❌ Optimization failed to find a feasible solution.")
        print("Try increasing max_iter or swarm_size, or adjusting initial parameters.")