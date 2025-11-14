"""
ga_intersection.py

Genetic Algorithm for CAV Intersection Optimization
"""

import numpy as np
from metaheuristic_intersection import (
    ProblemParameters,
    objective_function,
    feasibility_check,
    generate_random_solution,
    plot_combined_visualization
)
import matplotlib.pyplot as plt

# ============================================================================
# GA CONFIGURATION - ALL PARAMETERS IN ONE PLACE
# ============================================================================

class GAConfig:
    """
    All GA parameters in one configurable class
    Modify these values to experiment with different GA settings
    """
    def __init__(self):
        # -------------------- BASIC GA PARAMETERS --------------------
        self.population_size = 10          # m: Number of chromosomes in population
        self.max_generations = 100         # i_max: Stopping criterion
        
        # -------------------- OPERATOR PERCENTAGES -------------------
        self.p_elitism = 0.20              # 20% elites survive
        self.p_crossover = 0.60            # 60% from crossover
        self.p_mutation = 0.20             # 20% from mutation
        # Note: p_elitism + p_crossover + p_mutation must equal 1.0
        
        # -------------------- CROSSOVER PARAMETERS -------------------
        self.crossover_alpha = 0.7         # For whole arithmetic recombination
                                           # Child1 = α*P1 + (1-α)*P2
        
        # -------------------- MUTATION PARAMETERS --------------------
        self.mutation_std = 0.5            # Std dev of Gaussian noise for acceleration
        self.mutation_probability = 0.15   # Probability of mutating each gene
        self.max_repair_attempts = 5       # Max attempts to repair infeasible mutation
        
        # -------------------- FITNESS PARAMETERS ---------------------
        self.penalty_weight = 1e8          # Penalty for infeasible solutions
        
        # -------------------- CONVERGENCE TRACKING -------------------
        self.verbose = True                # Print progress every N generations
        self.print_every = 10              # Print frequency
        
    def get_operator_counts(self):
        """Calculate number of offspring from each operator"""
        n_elite = int(self.population_size * self.p_elitism)
        n_crossover = int(self.population_size * self.p_crossover)
        n_mutation = self.population_size - n_elite - n_crossover  # Ensure exact sum
        return n_elite, n_crossover, n_mutation


# ============================================================================
# FITNESS EVALUATION
# ============================================================================

def evaluate_fitness(chromosome, x0, v0, t0, params, config):
    """
    Evaluate fitness = objective + penalty for constraint violations
    Lower fitness is better (minimization)
    """
    # Calculate objective function
    f_total, f_time, f_energy, info = objective_function(chromosome, x0, v0, t0, params)
    
    # Check feasibility
    is_feasible, violations = feasibility_check(chromosome, x0, v0, t0, params)
    
    # Apply penalty if infeasible
    if not is_feasible:
        # Count total constraint violations
        n_violations = sum(len(v) for v in violations.values() if isinstance(v, list))
        penalty = config.penalty_weight * n_violations
        fitness = f_total + penalty
    else:
        fitness = f_total
    
    return fitness, is_feasible, f_total

# ============================================================================
# FEASIBLE SOLUTION GENERATION
# ============================================================================

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
# POPULATION INITIALIZATION
# ============================================================================

def initialize_population(config, x0, v0, t0, params):
    """
    Generate initial population using SA's working feasible solution generator
    """
    m = config.population_size
    population = []
    fitness_values = []
    feasibility_flags = []
    objective_values = []
    
    print(f"Generating initial population of {m} chromosomes...")
    print("  Using proven feasible solution generator from SA...")
    
    feasible_count = 0
    
    for i in range(m):
        print(f"    [{i+1}/{m}] Generating feasible solution...")
        
        # Use SA's working generator - it handles all constraints properly
        chromosome = generate_initial_feasible_solution(params, x0, v0, t0)
        
        # Verify feasibility
        is_feasible, violations = feasibility_check(chromosome, x0, v0, t0, params)
        
        if is_feasible:
            feasible_count += 1
            print(f"        ✓ Feasible")
        else:
            print(f"        ⚠️ Infeasible - keeping anyway for diversity")
        
        # Evaluate fitness
        fitness, is_feas_check, f_obj = evaluate_fitness(
            chromosome, x0, v0, t0, params, config
        )
        
        population.append(chromosome)
        fitness_values.append(fitness)
        feasibility_flags.append(is_feasible)
        objective_values.append(f_obj)
    
    # Convert to arrays
    population = np.array(population)
    fitness_values = np.array(fitness_values)
    feasibility_flags = np.array(feasibility_flags)
    objective_values = np.array(objective_values)
    
    print(f"\n  Initial population: {feasible_count}/{m} feasible")
    
    if feasible_count == 0:
        print("\n⚠️  WARNING: No feasible solutions generated.")
        print("This problem may be very difficult or have conflicting constraints.")
        print("GA will proceed with penalty method.")
    
    return population, fitness_values, feasibility_flags, objective_values


# ============================================================================
# SELECTION - FITNESS BASED (ELITISM)
# ============================================================================

def select_elites(population, fitness_values, n_elite):
    """
    Select n_elite best members (lowest fitness for minimization)
    """
    elite_indices = np.argsort(fitness_values)[:n_elite]
    return population[elite_indices].copy()


def select_crossover_parents(population, fitness_values, n_parents):
    """
    Select best n_parents for crossover (elite parents selection)
    """
    parent_indices = np.argsort(fitness_values)[:n_parents]
    return population[parent_indices]


def select_mutation_parent(population, fitness_values):
    """
    Select worst member for mutation
    """
    worst_idx = np.argmax(fitness_values)
    return population[worst_idx].copy()


# ============================================================================
# CROSSOVER OPERATORS
# ============================================================================

def crossover_acceleration_profiles(accel1, accel2, alpha, params):
    """
    Whole Arithmetic Recombination for continuous acceleration variables
    
    Child1 = alpha * Parent1 + (1-alpha) * Parent2
    Child2 = alpha * Parent2 + (1-alpha) * Parent1
    """
    child1_accel = alpha * accel1 + (1 - alpha) * accel2
    child2_accel = alpha * accel2 + (1 - alpha) * accel1
    
    # Clip to acceleration bounds
    child1_accel = np.clip(child1_accel, params.u_min, params.u_max)
    child2_accel = np.clip(child2_accel, params.u_min, params.u_max)
    
    return child1_accel, child2_accel


def crossover_binary_one_point(binary1, binary2):
    """
    One-Point Crossover for binary Z variables
    
    Randomly select crossover point and swap tails
    """
    if len(binary1) == 0:
        return binary1.copy(), binary2.copy()
    
    # Random crossover point
    point = np.random.randint(1, len(binary1))
    
    # Create children by swapping tails
    child1_binary = np.concatenate([binary1[:point], binary2[point:]])
    child2_binary = np.concatenate([binary2[:point], binary1[point:]])
    
    return child1_binary, child2_binary


def crossover(parent1, parent2, config, params):
    """
    Apply crossover to two parents to produce two children
    
    Split into acceleration part and binary part, apply appropriate operator
    """
    dim_accel = params.N * params.K
    
    # Split parents
    accel1 = parent1[:dim_accel]
    accel2 = parent2[:dim_accel]
    binary1 = parent1[dim_accel:]
    binary2 = parent2[dim_accel:]
    
    # Crossover acceleration profiles (Whole Arithmetic Recombination)
    child1_accel, child2_accel = crossover_acceleration_profiles(
        accel1, accel2, config.crossover_alpha, params
    )
    
    # Crossover binary variables (One-Point)
    child1_binary, child2_binary = crossover_binary_one_point(binary1, binary2)
    
    # Combine parts
    child1 = np.concatenate([child1_accel, child1_binary])
    child2 = np.concatenate([child2_accel, child2_binary])
    
    return child1, child2


def apply_crossover_operator(population, fitness_values, n_crossover, config, params, x0, v0, t0):
    """
    Apply crossover - if offspring is infeasible, accept anyway (penalty will handle it)
    """
    offspring = []
    n_pairs = int(np.ceil(n_crossover / 2))
    
    # Select best members as parent pool
    parents = select_crossover_parents(population, fitness_values, n_pairs * 2)
    
    # Generate offspring pairs
    for i in range(n_pairs):
        idx1 = i * 2
        idx2 = i * 2 + 1
        if idx2 >= len(parents):
            idx2 = 0
        
        parent1 = parents[idx1]
        parent2 = parents[idx2]
        
        # Apply crossover
        child1, child2 = crossover(parent1, parent2, config, params)
        
        # Check feasibility but accept anyway
        is_feas1, _ = feasibility_check(child1, x0, v0, t0, params)
        
        if not is_feas1:
            # Try simple repair: blend more with feasible parent
            child1 = 0.7 * parent1 + 0.3 * child1
        
        offspring.append(child1)
        
        if len(offspring) < n_crossover:
            is_feas2, _ = feasibility_check(child2, x0, v0, t0, params)
            
            if not is_feas2:
                child2 = 0.7 * parent2 + 0.3 * child2
            
            offspring.append(child2)
    
    return np.array(offspring[:n_crossover])

def repair_if_needed(chromosome, x0, v0, t0, params, config):
    """
    Check if chromosome is feasible, attempt simple repair if not
    """
    is_feasible, violations = feasibility_check(chromosome, x0, v0, t0, params)
    
    if is_feasible:
        return chromosome, True
    
    # Attempt simple repair: blend with conservative solution
    conservative, _ = generate_initial_feasible_solution(params, x0, v0, t0, max_attempts=1)
    
    # Try blending
    for blend in [0.7, 0.5, 0.3]:
        repaired = blend * chromosome + (1 - blend) * conservative
        is_feasible, _ = feasibility_check(repaired, x0, v0, t0, params)
        if is_feasible:
            return repaired, True
    
    # If repair failed, return original (will get penalized)
    return chromosome, False


# ============================================================================
# MUTATION OPERATORS
# ============================================================================

def mutate_acceleration_with_repair(accel, config, params, x0, v0, t0, parent_chromosome):
    """
    Mutate acceleration profile by adding Gaussian noise
    If result is infeasible, attempt repair by regenerating
    """
    dim_accel = params.N * params.K
    mutated_accel = accel.copy()
    
    # Add Gaussian noise to randomly selected genes
    for i in range(len(mutated_accel)):
        if np.random.rand() < config.mutation_probability:
            noise = np.random.normal(0, config.mutation_std)
            mutated_accel[i] += noise
            mutated_accel[i] = np.clip(mutated_accel[i], params.u_min, params.u_max)
    
    # Check if mutated solution is feasible
    # Create temporary chromosome to test
    temp_chromosome = parent_chromosome.copy()
    temp_chromosome[:dim_accel] = mutated_accel
    
    is_feasible, _ = feasibility_check(temp_chromosome, x0, v0, t0, params)
    
    # If infeasible, try to repair by reducing noise magnitude
    attempts = 0
    while not is_feasible and attempts < config.max_repair_attempts:
        # Blend with original to reduce disruption
        blend_factor = 0.5
        mutated_accel = blend_factor * accel + (1 - blend_factor) * mutated_accel
        mutated_accel = np.clip(mutated_accel, params.u_min, params.u_max)
        
        temp_chromosome[:dim_accel] = mutated_accel
        is_feasible, _ = feasibility_check(temp_chromosome, x0, v0, t0, params)
        attempts += 1
    
    # If still infeasible after repairs, keep original
    if not is_feasible:
        mutated_accel = accel.copy()
    
    return mutated_accel


def mutate_binary_bit_flip(binary, config):
    """
    Bit Flip mutation for binary Z variables
    Flip each bit with mutation probability
    """
    mutated_binary = binary.copy()
    for i in range(len(mutated_binary)):
        if np.random.rand() < config.mutation_probability:
            mutated_binary[i] = 1 - mutated_binary[i]  # Flip 0↔1
    return mutated_binary


def mutate(parent, config, params, x0, v0, t0):
    """
    Apply mutation - GENTLE changes to maintain feasibility better
    
    Args:
        parent: Parent chromosome
        config: GAConfig object
        params: Problem parameters
        x0, v0, t0: Initial conditions (not used but kept for consistency)
    """
    dim_accel = params.N * params.K
    child = parent.copy()
    
    # Mutate acceleration (GENTLE - only small changes)
    accel = child[:dim_accel]
    
    # Only mutate a few genes, not all
    n_genes_to_mutate = max(1, int(dim_accel * config.mutation_probability))
    mutation_indices = np.random.choice(dim_accel, n_genes_to_mutate, replace=False)
    
    # Add SMALL noise
    noise = np.random.normal(0, config.mutation_std * 0.5, n_genes_to_mutate)
    accel[mutation_indices] += noise
    accel = np.clip(accel, params.u_min, params.u_max)
    child[:dim_accel] = accel
    
    # Mutate binary (very low probability)
    binary = child[dim_accel:]
    if len(binary) > 0 and np.random.rand() < 0.1:  # Only 10% chance
        flip_idx = np.random.randint(len(binary))
        binary[flip_idx] = 1 - binary[flip_idx]
        child[dim_accel:] = binary
    
    return child


def apply_mutation_operator(population, fitness_values, n_mutation, config, params, x0, v0, t0):
    """
    Apply mutation - gentle changes
    """
    offspring = []
    
    for _ in range(n_mutation):
        parent = select_mutation_parent(population, fitness_values)
        child = mutate(parent, config, params, x0, v0, t0)
        
        # Check feasibility
        is_feasible, _ = feasibility_check(child, x0, v0, t0, params)
        
        if not is_feasible:
            # If infeasible, blend with parent (less disruptive)
            child = 0.8 * parent + 0.2 * child
        
        offspring.append(child)
    
    return np.array(offspring)


# ============================================================================
# MAIN GA ALGORITHM
# ============================================================================

def genetic_algorithm(x0, v0, t0, params, config):
    """
    Main Genetic Algorithm
    
    Algorithm:
    1. Initialize population (m chromosomes)
    2. Evaluate fitness
    3. For i = 1 to i_max:
        a. Select elites (p_elitism * m)
        b. Generate offspring via crossover (p_crossover * m)
        c. Generate offspring via mutation (p_mutation * m)
        d. Form new generation
        e. Evaluate fitness
        f. Track best solution
    4. Return best solution
    """
    n_elite, n_crossover, n_mutation = config.get_operator_counts()
    
    print("="*70)
    print("GENETIC ALGORITHM - CONFIGURATION")
    print("="*70)
    print(f"Population size (m): {config.population_size}")
    print(f"Max generations (i_max): {config.max_generations}")
    print(f"Elitism: {config.p_elitism*100:.0f}% → {n_elite} members")
    print(f"Crossover: {config.p_crossover*100:.0f}% → {n_crossover} members")
    print(f"Mutation: {config.p_mutation*100:.0f}% → {n_mutation} members")
    print(f"\nCrossover alpha: {config.crossover_alpha}")
    print(f"Mutation std dev: {config.mutation_std}")
    print(f"Mutation probability: {config.mutation_probability}")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # INITIALIZATION (Generation 0)
    # -------------------------------------------------------------------------
    population, fitness_values, feasibility_flags, objective_values = \
        initialize_population(config, x0, v0, t0, params)
    
    # Track global best
    best_idx = np.argmin(fitness_values)
    best_solution = population[best_idx].copy()
    best_fitness = fitness_values[best_idx]
    best_objective = objective_values[best_idx]
    best_generation = 0
    
    # History for plotting
    history = {
        'best_fitness': [best_fitness],
        'best_objective': [best_objective],
        'avg_fitness': [np.mean(fitness_values)],
        'feasibility_rate': [np.mean(feasibility_flags)]
    }
    
    print(f"\nGeneration 0:")
    print(f"  Best fitness: {best_fitness:.4f} | Best objective: {best_objective:.4f}")
    print(f"  Feasible solutions: {np.sum(feasibility_flags)}/{config.population_size}")
    
    # -------------------------------------------------------------------------
    # MAIN LOOP (Generations 1 to i_max)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STARTING EVOLUTION")
    print("="*70)
    
    for generation in range(1, config.max_generations + 1):
        
        # ---------------------------------------------------------------------
        # STEP 1: Elitism - Preserve best members
        # ---------------------------------------------------------------------
        elites = select_elites(population, fitness_values, n_elite)
        
        # ---------------------------------------------------------------------
        # STEP 2: Crossover - Generate offspring from best parents
        # ---------------------------------------------------------------------
        crossover_offspring = apply_crossover_operator(
            population, fitness_values, n_crossover, config, params, x0, v0, t0
        )
        
        # ---------------------------------------------------------------------
        # STEP 3: Mutation - Generate offspring from worst parents
        # ---------------------------------------------------------------------
        mutation_offspring = apply_mutation_operator(
            population, fitness_values, n_mutation, config, params, x0, v0, t0
        )
        
        # ---------------------------------------------------------------------
        # STEP 4: Form new generation
        # ---------------------------------------------------------------------
        new_population = np.vstack([elites, crossover_offspring, mutation_offspring])
        
        # ---------------------------------------------------------------------
        # STEP 5: Evaluate new generation
        # ---------------------------------------------------------------------
        new_fitness = []
        new_feasibility = []
        new_objectives = []
        
        for chromosome in new_population:
            fitness, is_feasible, f_obj = evaluate_fitness(
                chromosome, x0, v0, t0, params, config
            )
            new_fitness.append(fitness)
            new_feasibility.append(is_feasible)
            new_objectives.append(f_obj)
        
        population = new_population
        fitness_values = np.array(new_fitness)
        feasibility_flags = np.array(new_feasibility)
        objective_values = np.array(new_objectives)
        
        # ---------------------------------------------------------------------
        # STEP 6: Update best solution
        # ---------------------------------------------------------------------
        current_best_idx = np.argmin(fitness_values)
        if fitness_values[current_best_idx] < best_fitness:
            best_solution = population[current_best_idx].copy()
            best_fitness = fitness_values[current_best_idx]
            best_objective = objective_values[current_best_idx]
            best_generation = generation
        
        # Track history
        history['best_fitness'].append(best_fitness)
        history['best_objective'].append(best_objective)
        history['avg_fitness'].append(np.mean(fitness_values))
        history['feasibility_rate'].append(np.mean(feasibility_flags))
        
        # Print progress
        if config.verbose and (generation % config.print_every == 0 or generation == config.max_generations):
            print(f"\nGeneration {generation}/{config.max_generations}:")
            print(f"  Best fitness: {best_fitness:.4f} | Best objective: {best_objective:.4f}")
            print(f"  Current avg fitness: {np.mean(fitness_values):.4f}")
            print(f"  Feasible: {np.sum(feasibility_flags)}/{config.population_size}")
            print(f"  (Best found at generation {best_generation})")
    
    # -------------------------------------------------------------------------
    # FINAL RESULTS
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("GA OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Best solution found at generation: {best_generation}")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Best objective: {best_objective:.4f}")
    
    # Check if best solution is feasible
    _, is_feasible, _ = evaluate_fitness(best_solution, x0, v0, t0, params, config)
    print(f"Best solution feasibility: {'FEASIBLE' if is_feasible else 'INFEASIBLE'}")
    
    return best_solution, best_fitness, best_objective, history


# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================

def plot_ga_convergence(history):
    """
    Plot best objective function value over generations
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    generations = range(len(history['best_objective']))
    
    # Main plot: Best objective over generations
    ax.plot(generations, history['best_objective'], 
            'b-', linewidth=2.5, marker='o', markersize=4, 
            markevery=max(1, len(generations)//20), label='Best Objective')
    
    ax.set_xlabel('Generation (Iteration)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Objective Function Value', fontsize=13, fontweight='bold')
    ax.set_title('GA Convergence: Best Objective Over Generations', 
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    # Add annotation for best value
    best_gen = np.argmin(history['best_objective'])
    best_val = history['best_objective'][best_gen]
    
    ax.annotate(f'Best: {best_val:.4f}\nat Generation {best_gen}',
                xy=(best_gen, best_val),
                xytext=(best_gen + len(generations)*0.15, best_val + (max(history['best_objective']) - min(history['best_objective']))*0.1),
                fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    plt.tight_layout()
    return fig


# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================

# if __name__ == "__main__":
#     # -------------------------------------------------------------------------
#     # PROBLEM SETUP
#     # -------------------------------------------------------------------------
#     params = ProblemParameters()
    
#     # Initial conditions
#     x0 = np.zeros(params.N)
#     v0 = np.array([12.0, 11.0, 13.0, 10.0])
#     t0 = np.array([0.0, 0.5, 1.0, 1.5])
    
#     print("\nProblem Configuration:")
#     print(f"  N = {params.N} vehicles")
#     print(f"  K = {params.K} time steps")
#     print(f"  Decision variables: {params.N * params.K} (accel) + {np.sum(params.get_conflict_matrix())//2} (binary)")
    
#     # -------------------------------------------------------------------------
#     # GA CONFIGURATION - MODIFY THESE TO EXPERIMENT
#     # -------------------------------------------------------------------------
#     config = GAConfig()
    
#     # Example: Try different configurations
#     # config.population_size = 20
#     # config.max_generations = 50
#     # config.p_elitism = 0.10
#     # config.p_crossover = 0.70
#     # config.p_mutation = 0.20
#     # config.crossover_alpha = 0.5
#     # config.mutation_std = 1.0
    
#     # -------------------------------------------------------------------------
#     # RUN GA
#     # -------------------------------------------------------------------------
#     best_solution, best_fitness, best_objective, history = genetic_algorithm(
#         x0, v0, t0, params, config
#     )
    
#     # -------------------------------------------------------------------------
#     # VISUALIZE RESULTS
#     # -------------------------------------------------------------------------
#     print("\nGenerating convergence plots...")
#     fig_convergence = plot_ga_convergence(history)
    
#     print("\nGenerating best solution visualization...")
#     fig_solution, anim = plot_combined_visualization(best_solution, x0, v0, t0, params)
    
#     plt.show()