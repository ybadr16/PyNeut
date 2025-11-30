import numpy as np
import matplotlib.pyplot as plt
from src.physics import elastic_scattering
from src.random_number_generator import RNGHandler

def run_hydrogen_test():
    print("Running Hydrogen Scattering Validation...")

    A = 1.0 # Hydrogen
    E_in = 1.0e6 # 1 MeV
    num_samples = 10000
    final_energies = []

    for i in range(num_samples):
        rng = RNGHandler(seed=i)
        # Mock sampler that returns 0 temp (target at rest)
        # We need a dummy object for sampler
        class MockSampler:
            def sample_target_velocity(self, *args): return 0.0, 0.0, 0.0

        E_out, _, _ = elastic_scattering(E_in, A, MockSampler(), rng)
        final_energies.append(E_out)

    # Theoretical Average for Hydrogen
    # E_avg = E_in / 2
    avg_sim = np.mean(final_energies)
    print(f"Input Energy: {E_in} eV")
    print(f"Expected Avg: {E_in/2} eV")
    print(f"Simulated Avg: {avg_sim:.2f} eV")

    # Plot Histogram
    plt.figure()
    count, bins, ignored = plt.hist(final_energies, 50, density=True, alpha=0.6, color='g', label='Simulation')

    # Theory Line
    plt.plot([0, E_in], [1/E_in, 1/E_in], 'r-', linewidth=2, label='Theory (Uniform)')

    plt.title("Hydrogen (A=1) Scattering Kernel Validation")
    plt.xlabel("Outgoing Energy (eV)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_hydrogen_test()
