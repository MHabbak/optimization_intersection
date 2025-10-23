"""
Flexible vehicle configuration for intersection optimization
Supports any number of vehicles from any direction
"""

import numpy as np
from metaheuristic_intersection import ProblemParameters

def create_vehicle_config(vehicle_specs):
    """
    Create flexible vehicle configuration
    
    Args:
        vehicle_specs: List of dicts, each with:
            - 'direction': 'EW', 'WE', 'NS', or 'SN'
            - 'v0': Initial velocity (m/s)
            - 't0': Arrival time (s)
            - 'x0': Initial position (default 0.0)
    
    Returns:
        params, x0, v0, t0
    
    Example:
        vehicles = [
            {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
            {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
            {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
            {'direction': 'SN', 'v0': 10.0, 't0': 1.5},
            {'direction': 'EW', 'v0': 14.0, 't0': 2.0},  # 5th vehicle
            {'direction': 'NS', 'v0': 12.5, 't0': 2.5},  # 6th vehicle
        ]
    """
    # Count vehicles per direction
    direction_counts = {'EW': 0, 'WE': 0, 'NS': 0, 'SN': 0}
    
    for spec in vehicle_specs:
        direction = spec['direction'].upper()
        if direction not in direction_counts:
            raise ValueError(f"Invalid direction: {direction}. Use 'EW', 'WE', 'NS', or 'SN'")
        direction_counts[direction] += 1
    
    N_total = len(vehicle_specs)
    
    # Create parameters with flexible vehicle counts
    params = ProblemParameters(
        N=N_total,
        N_EW=direction_counts['EW'],
        N_WE=direction_counts['WE'],
        N_NS=direction_counts['NS'],
        N_SN=direction_counts['SN']
    )
    
    # Sort vehicles by direction order: EW, WE, NS, SN
    # This ensures vehicle indices match the expected order
    direction_order = {'EW': 0, 'WE': 1, 'NS': 2, 'SN': 3}
    sorted_specs = sorted(vehicle_specs, key=lambda x: direction_order[x['direction'].upper()])
    
    # Extract initial conditions
    x0 = np.array([spec.get('x0', 0.0) for spec in sorted_specs])
    v0 = np.array([spec['v0'] for spec in sorted_specs])
    t0 = np.array([spec['t0'] for spec in sorted_specs])
    
    return params, x0, v0, t0, sorted_specs


def print_vehicle_summary(params, x0, v0, t0, sorted_specs):
    """Print summary of vehicle configuration"""
    print("\n" + "="*70)
    print("VEHICLE CONFIGURATION")
    print("="*70)
    print(f"Total vehicles: {params.N}")
    print(f"  East→West: {params.N_EW}")
    print(f"  West→East: {params.N_WE}")
    print(f"  North→South: {params.N_NS}")
    print(f"  South→North: {params.N_SN}")
    
    print(f"\nDetailed Vehicle List:")
    direction_names = {
        'EW': 'East → West',
        'WE': 'West → East',
        'NS': 'North → South',
        'SN': 'South → North'
    }
    
    for i, spec in enumerate(sorted_specs):
        dir_code = spec['direction'].upper()
        print(f"  V{i}: {direction_names[dir_code]:15s} | "
              f"x0={x0[i]:4.1f}m | v0={v0[i]:5.1f}m/s | t0={t0[i]:4.1f}s")
    
    print("="*70)


# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================

def config_4_way_balanced():
    """Original 4-vehicle configuration (1 per direction)"""
    vehicles = [
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        {'direction': 'SN', 'v0': 10.0, 't0': 1.5},
    ]
    return create_vehicle_config(vehicles)


def config_6_vehicles():
    """6-vehicle configuration"""
    vehicles = [
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'EW', 'v0': 14.0, 't0': 2.0},  # 2nd EW vehicle
        {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        {'direction': 'NS', 'v0': 12.5, 't0': 2.5},  # 2nd NS vehicle
        {'direction': 'SN', 'v0': 10.0, 't0': 1.5},
    ]
    return create_vehicle_config(vehicles)


def config_heavy_eastwest():
    """Heavy traffic from East-West, light North-South"""
    vehicles = [
        # Heavy EW traffic
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'EW', 'v0': 13.5, 't0': 1.5},
        {'direction': 'EW', 'v0': 11.5, 't0': 3.0},
        # Heavy WE traffic
        {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
        {'direction': 'WE', 'v0': 12.5, 't0': 2.0},
        # Light NS/SN
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        {'direction': 'SN', 'v0': 10.0, 't0': 2.5},
    ]
    return create_vehicle_config(vehicles)


def config_10_vehicles():
    """Large-scale 10-vehicle scenario"""
    vehicles = [
        {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
        {'direction': 'EW', 'v0': 13.0, 't0': 2.0},
        {'direction': 'WE', 'v0': 11.0, 't0': 0.5},
        {'direction': 'WE', 'v0': 12.0, 't0': 2.5},
        {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
        {'direction': 'NS', 'v0': 14.0, 't0': 3.0},
        {'direction': 'NS', 'v0': 12.5, 't0': 4.5},
        {'direction': 'SN', 'v0': 10.0, 't0': 1.5},
        {'direction': 'SN', 'v0': 11.5, 't0': 3.5},
        {'direction': 'SN', 'v0': 13.0, 't0': 5.0},
    ]
    return create_vehicle_config(vehicles)


def config_custom(n_ew=1, n_we=1, n_ns=1, n_sn=1, 
                  v_range=(10.0, 15.0), t_spacing=0.5):
    """
    Generate custom configuration with specified counts per direction
    
    Args:
        n_ew, n_we, n_ns, n_sn: Number of vehicles in each direction
        v_range: (min_speed, max_speed) for random velocities
        t_spacing: Time spacing between consecutive vehicles
    """
    vehicles = []
    t_current = 0.0
    
    # Generate vehicles in order: EW, WE, NS, SN
    for direction, count in [('EW', n_ew), ('WE', n_we), ('NS', n_ns), ('SN', n_sn)]:
        for _ in range(count):
            v0 = np.random.uniform(*v_range)
            vehicles.append({
                'direction': direction,
                'v0': v0,
                't0': t_current
            })
            t_current += t_spacing
    
    return create_vehicle_config(vehicles)


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FLEXIBLE VEHICLE CONFIGURATION DEMO")
    print("="*70)
    
    # Example 1: Original 4-vehicle setup
    print("\n[Example 1] Original 4-vehicle balanced configuration:")
    params1, x0_1, v0_1, t0_1, specs1 = config_4_way_balanced()
    print_vehicle_summary(params1, x0_1, v0_1, t0_1, specs1)
    
    # Example 2: 6 vehicles
    print("\n[Example 2] 6-vehicle configuration:")
    params2, x0_2, v0_2, t0_2, specs2 = config_6_vehicles()
    print_vehicle_summary(params2, x0_2, v0_2, t0_2, specs2)
    
    # Example 3: Custom configuration
    print("\n[Example 3] Custom 8-vehicle configuration:")
    params3, x0_3, v0_3, t0_3, specs3 = config_custom(
        n_ew=2, n_we=2, n_ns=2, n_sn=2,
        v_range=(11.0, 14.0),
        t_spacing=0.8
    )
    print_vehicle_summary(params3, x0_3, v0_3, t0_3, specs3)
    
    # Show how to use with SA
    print("\n" + "="*70)
    print("USAGE WITH SIMULATED ANNEALING")
    print("="*70)
    print("""
# Import SA module
from simulated_annealing_m3 import simulated_annealing, plot_convergence
from metaheuristic_intersection import plot_combined_visualization

# Define your vehicles
vehicles = [
    {'direction': 'EW', 'v0': 12.0, 't0': 0.0},
    {'direction': 'NS', 'v0': 13.0, 't0': 1.0},
    # ... add more vehicles
]

# Create configuration
params, x0, v0, t0, specs = create_vehicle_config(vehicles)
print_vehicle_summary(params, x0, v0, t0, specs)

# Run optimization
x_best, f_best, history = simulated_annealing(
    params, x0, v0, t0,
    T_init=1000.0,
    T_final=0.1,
    max_iter=5000
)

# Visualize results
plot_convergence(history)
plot_combined_visualization(x_best, x0, v0, t0, params)
plt.show()
    """)
