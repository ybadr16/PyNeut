# validate_slab.py
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_single_particle
from src.medium import Region, Plane
from src.settings import Settings
from src.random_number_generator import RNGHandler

# --- 1. MOCK READER ---
class MockReader:
    def get_cross_sections(self, element, energy, sampler, N):
        # Pure Absorber: Sigma_t = 1.0
        return 0.0, 1.0, 0.0, 1.0

def run_slab_test():
    print("Running Slab Transmission Validation (Beer-Lambert Law)...")

    thicknesses = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    results = []

    # Use Criticality (Analog) mode so absorption kills the particle
    settings = Settings(mode="criticality")

    # Dummy variables
    A = 1.0
    N = 1.0
    sampler = None
    track = False
    region_bounds = None

    # Small offset to ensure we start SAFELY INSIDE the region
    z_start = 0.0001

    print(f"{'Thickness':<10} | {'Theory':<10} | {'Simulated':<10} | {'Diff %':<10}")
    print("-" * 50)

    for T in thicknesses:
        # 1. Geometry: Single Finite Slab
        # We assume the planes are defined such that z >= 0 and z <= T
        slab = Region(
            surfaces=[
                Plane(0,0,-1, 0),   # Front (z>=0)
                Plane(0,0,1, T),    # Back (z<=T)
                Plane(-1,0,0, 50),  # Sides...
                Plane(1,0,0, 50),
                Plane(0,-1,0, 50),
                Plane(0,1,0, 50)
            ],
            name="Absorbium", priority=1, element="Absorbium"
        )

        # NO VOID REGION.
        # If the particle leaves "Absorbium", it automatically "Escapes".
        mediums = [slab]

        escaped_count = 0
        num_particles = 5000

        for i in range(num_particles):
            rng = RNGHandler(seed=i)

            # Start INSIDE the slab
            state = {
                "x": 0.0, "y": 0.0, "z": z_start,
                "theta": 0.0, "phi": 0.0,
                "has_interacted": False, "energy": 1e6, "weight": 1.0
            }

            reader = MockReader()
            args = (state, reader, mediums, A, N, sampler, region_bounds, track, rng, settings)
            data = simulate_single_particle(args)

            if data["result"] == "escaped":
                 escaped_count += 1

        simulated_T = escaped_count / num_particles

        # Adjust theory for the slight offset
        # Distance to travel = Total Thickness - Start Position
        dist_to_travel = T - z_start
        theoretical_T = np.exp(-1.0 * dist_to_travel)

        diff = abs(simulated_T - theoretical_T) / theoretical_T * 100
        print(f"{T:<10.1f} | {theoretical_T:<10.4f} | {simulated_T:<10.4f} | {diff:<10.2f}%")
        results.append(simulated_T)

    plt.figure(figsize=(8, 6))
    x_smooth = np.linspace(0, 5.5, 100)
    plt.plot(x_smooth, np.exp(-x_smooth), 'r-', label='Theory')
    plt.plot(thicknesses, results, 'bo', label='Simulation')
    plt.yscale('log')
    plt.xlabel('Slab Thickness (mean free paths)')
    plt.ylabel('Transmission Probability (Log Scale)')
    plt.title('Validation: Beer-Lambert Law')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_slab_test()
