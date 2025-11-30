import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_single_particle
from src.medium import Box
from src.settings import Settings
from src.random_number_generator import RNGHandler

class MockReader:
    # Pure Absorber: Sigma_a = 1.0, Sigma_t = 1.0
    def get_cross_sections(self, element, energy, sampler, N):
        return 0.0, 1.0, 0.0, 1.0

def run_benchmark():
    print("Running Point Source Benchmark (Attenuation in Infinite Medium)...")

    # Infinite Box of Absorbium
    # We make it huge so particles effectively never escape, they just fly until absorbed.
    box = Box(-100, 100, -100, 100, -100, 100)
    box.element = "Absorbium"
    box.priority = 1
    mediums = [box]

    settings = Settings(mode="criticality")

    num_particles = 10000

    # --- FIX: Use a simple list to store distances ---
    absorbed_distances = []

    print(f"Simulating {num_particles} particles...")

    for i in range(num_particles):
        rng = RNGHandler(seed=i)

        # ISOTROPIC SOURCE at (0,0,0)
        # Sample random direction on sphere
        phi = rng.uniform(0, 2*np.pi)
        costheta = rng.uniform(-1, 1)
        theta = np.arccos(costheta)

        state = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "theta": theta, "phi": phi,
            "has_interacted": False, "energy": 1e6, "weight": 1.0
        }

        # Run simulation
        args = (state, MockReader(), mediums, 1.0, 1.0, None, None, False, rng, settings)
        data = simulate_single_particle(args)

        if data["result"] == "absorbed":
            # Extract distance from origin (0,0,0) to absorption point
            x, y, z = data["absorbed_coords"][0][0:3]
            distance = np.sqrt(x**2 + y**2 + z**2)

            absorbed_distances.append(distance)

    # --- PLOTTING ---
    if not absorbed_distances:
        print("[ERROR] No particles were absorbed. Check geometry/physics.")
        return

    distances = np.array(absorbed_distances)

    plt.figure(figsize=(8,6))

    # Theoretical PDF for flight distance in infinite medium:
    # P(r) = Sigma_t * exp(-Sigma_t * r)
    # Here Sigma_t = 1.0
    r_space = np.linspace(0, 8, 100)
    pdf_theoretical = 1.0 * np.exp(-1.0 * r_space)

    plt.hist(distances, bins=50, density=True, alpha=0.6, color='skyblue', label='Simulated Flight Distance')
    plt.plot(r_space, pdf_theoretical, 'r-', lw=2, label='Theory (exp(-r))')

    plt.xlabel("Distance traveled before absorption (cm)")
    plt.ylabel("Probability Density")
    plt.title("Benchmark: Flight Distance Distribution (Isotropic Source)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_benchmark()
