import time
import numpy as np
import os
import h5py
import sys

# --- 1. THE OLD SLOW CLASS (Direct Disk Access) ---
class UncachedReader:
    def __init__(self, base_path):
        self.base_path = base_path

    def get_cross_section(self, element, mt, energy):
        # This simulates your original code: Opening the file EVERY time
        mt_str = f"{mt:03}"
        file_path = os.path.join(self.base_path, f"neutron/{element}.h5")

        reaction_group_path = f"{element}/reactions/reaction_{mt_str}/294K"
        energy_path = f"{element}/energy/294K"

        with h5py.File(file_path, 'r') as f:
            energy_data = f[energy_path][:]
            xs_dataset = f[f"{reaction_group_path}/xs"]
            xs_data = xs_dataset[:]
            threshold_idx = xs_dataset.attrs.get('threshold_idx', 0)

            xs_full = np.zeros_like(energy_data)
            xs_full[threshold_idx:threshold_idx + len(xs_data)] = xs_data

            if energy < energy_data[threshold_idx]:
                return 0.0
            return np.interp(energy, energy_data, xs_full)

# --- 2. THE NEW FAST CLASS (Memory Caching) ---
class CachedReader:
    def __init__(self, base_path):
        self.base_path = base_path
        self._cache = {}

    def _load_to_cache(self, element, mt):
        mt_str = f"{mt:03}"
        file_path = os.path.join(self.base_path, f"neutron/{element}.h5")
        reaction_group_path = f"{element}/reactions/reaction_{mt_str}/294K"
        energy_path = f"{element}/energy/294K"

        with h5py.File(file_path, 'r') as f:
            energy_data = f[energy_path][:]
            xs_dataset = f[f"{reaction_group_path}/xs"]
            xs_data = xs_dataset[:]
            threshold_idx = xs_dataset.attrs.get('threshold_idx', 0)

            xs_full = np.zeros_like(energy_data)
            xs_full[threshold_idx:threshold_idx + len(xs_data)] = xs_data

            self._cache[(element, mt)] = {
                'energy': energy_data,
                'xs': xs_full,
                'threshold': energy_data[threshold_idx] if threshold_idx < len(energy_data) else 0.0
            }

    def get_cross_section(self, element, mt, energy):
        key = (element, mt)
        if key not in self._cache:
            self._load_to_cache(element, mt)

        data = self._cache[key]
        if energy < data['threshold']:
            return 0.0
        return np.interp(energy, data['energy'], data['xs'])

# --- 3. THE RACE ---
def run_benchmark():
    # SETUP
    base_path = "./endfb"
    if not os.path.exists(base_path):
        print(f"Error: Could not find '{base_path}'. Please run this in your project root.")
        return

    # Test parameters
    element = "Pb208"  # Ensure you have Pb208.h5 in your folder
    mt = 2             # Elastic scattering
    N_lookups = 100000   # How many times we ask for a cross-section

    # Generate random energies (1 eV to 1 MeV)
    test_energies = np.random.uniform(1, 1e6, N_lookups)

    print(f"--- BENCHMARK STARTED ---")
    print(f"Target: {element} (MT={mt})")
    print(f"Operations: {N_lookups} lookups")
    print("-" * 30)

    # TEST 1: OLD METHOD
    print("Running Uncached Reader (Original)...")
    old_reader = UncachedReader(base_path)

    start_time = time.perf_counter()
    for E in test_energies:
        old_reader.get_cross_section(element, mt, E)
    end_time = time.perf_counter()

    old_duration = end_time - start_time
    print(f"Uncached Time: {old_duration:.4f} seconds")

    # TEST 2: NEW METHOD
    print("\nRunning Cached Reader (Optimized)...")
    new_reader = CachedReader(base_path)

    # We do one warmup call to load the cache (simulating startup)
    new_reader.get_cross_section(element, mt, 1.0)

    start_time = time.perf_counter()
    for E in test_energies:
        new_reader.get_cross_section(element, mt, E)
    end_time = time.perf_counter()

    new_duration = end_time - start_time
    print(f"Cached Time:   {new_duration:.4f} seconds")

    # RESULTS
    print("-" * 30)
    speedup = old_duration / new_duration
    print(f"SPEEDUP FACTOR: {speedup:.1f}x FASTER")
    print("-" * 30)

if __name__ == "__main__":
    run_benchmark()
