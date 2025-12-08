### src/cross_section_read.py ###
from .physics import calculate_E_cm_prime
import os
import h5py
import numpy as np

class CrossSectionReader:
    def __init__(self, base_path: str):
        """
        Initialize the CrossSectionReader with the base path to the data files.
        :param base_path: Base directory where HDF5 files are located.
        """
        self.base_path = base_path
        # Cache structure: {(element, mt): {'energy': np.array, 'xs': np.array}}
        self._cache = {}

    def _load_data_to_cache(self, element: str, mt: int):
        """
        Internal method to load data from HDF5 into memory cache.
        """
        # Validate inputs
        if not element.isalnum():
            raise ValueError("Invalid element format. Use alphanumeric characters (e.g., U235, Pb208).")
        if not (1 <= mt <= 999):
            raise ValueError("MT number must be between 1 and 999.")

        mt_str = f"{mt:03}"
        file_path = os.path.join(self.base_path, f"neutron/{element}.h5")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HDF5 file for {element} not found at {file_path}.")

        reaction_group_path = f"{element}/reactions/reaction_{mt_str}/294K"
        energy_path = f"{element}/energy/294K"

        try:
            with h5py.File(file_path, 'r') as f:
                # Load energy data
                if energy_path not in f:
                    raise KeyError(f"Energy data path '{energy_path}' not found in HDF5 file.")
                energy_data = f[energy_path][:]

                # Load cross-section data
                if reaction_group_path not in f:
                    raise KeyError(f"Reaction group path '{reaction_group_path}' not found in HDF5 file.")

                xs_dataset = f[f"{reaction_group_path}/xs"]
                xs_data = xs_dataset[:]
                threshold_idx = xs_dataset.attrs.get('threshold_idx', 0)

                # Validate threshold index
                if not (0 <= threshold_idx < len(energy_data)):
                    raise ValueError("Invalid threshold index in the HDF5 file.")

                # Construct the full cross-section array in memory
                xs_full = np.zeros_like(energy_data)
                xs_full[threshold_idx:threshold_idx + len(xs_data)] = xs_data

                # Store in cache
                self._cache[(element, mt)] = {
                    'energy': energy_data,
                    'xs': xs_full,
                    'threshold_energy': energy_data[threshold_idx] if threshold_idx < len(energy_data) else 0.0
                }

        except (OSError, KeyError, ValueError) as e:
            raise RuntimeError(f"Error while reading HDF5 file: {e}") from e

    def get_cross_section(self, element: str, mt: int, energy: float) -> float:
        """
        Get the cross-section for a specific nuclide, reaction, and energy using cached data.
        """
        cache_key = (element, mt)

        # Lazy loading: If data isn't in memory, load it now
        if cache_key not in self._cache:
            self._load_data_to_cache(element, mt)

        data = self._cache[cache_key]

        # Optimization: Quick check for threshold
        if energy < data['threshold_energy']:
            return 0.0

        # Fast in-memory interpolation
        return np.interp(energy, data['energy'], data['xs'])

    def calculate_macroscopic_xs(self, microscopic_xs: float, number_density: float) -> float:
        if microscopic_xs < 0:
            raise ValueError("Microscopic cross section cannot be negative")
        if number_density < 0:
            raise ValueError("Number density cannot be negative")

        microscopic_xs_cm2 = microscopic_xs * 1e-24
        return microscopic_xs_cm2 * number_density

    def get_macroscopic_xs(self, element: str, mt: int, energy: float, number_density: float) -> float:
        microscopic_xs = self.get_cross_section(element, mt, energy)
        return self.calculate_macroscopic_xs(microscopic_xs, number_density)

    def get_cross_sections(self, element, energy, sampler, number_density):
        """
        Get energy-dependent macroscopic cross sections for a given element and energy.
        """
        # Get microscopic cross-sections and convert to macroscopic
        energy_cm = calculate_E_cm_prime(energy, 2.5, sampler)  # 2.5 is A for now

        # Calculate each macroscopic cross section directly
        # Note: We query the cache for MT=2 and MT=102
        Sigma_s = self.get_macroscopic_xs(element, 2, energy_cm, number_density)     # Scattering
        Sigma_a = self.get_macroscopic_xs(element, 102, energy, number_density)      # Radiative capture

        try:
            # Try to get fission, default to 0 if not found (lazy load will fail if file missing)
            # However, for generic handling, we wrap the specific call
            Sigma_f = self.get_macroscopic_xs(element, 18, energy, number_density)   # Fission
        except (RuntimeError, KeyError):
            # If the MT 18 doesn't exist in the file, we assume 0 fission
            Sigma_f = 0.0

        Sigma_t = Sigma_s + Sigma_a + Sigma_f

        return Sigma_s, Sigma_a, Sigma_f, Sigma_t
