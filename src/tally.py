# src/tally.py
import numpy as np

class Tally:
    def __init__(self):
        self.results = {"absorbed": 0.0, "escaped": 0, "killed": 0}
        self.absorbed_coordinates = []
        self.energy_spectrum = []
        self.region_count = 0  # Tracks particles detected in a specific region

    def update(self, result, absorbed_weight=0.0, absorbed_coords=None, final_energy=None, region_detected=False):
        # Update tally counts for discrete events (escaped, killed)
        if result in ["escaped", "killed"]:
            self.results[result] += 1

        # Accumulate weight for absorption
        self.results["absorbed"] += absorbed_weight

        # Track absorbed coordinates
        if absorbed_coords:
            self.absorbed_coordinates.extend(absorbed_coords)

        # Track energy spectrum
        if final_energy is not None:
            self.energy_spectrum.append(final_energy)

        # Update region detection count
        if region_detected:
            self.region_count += 1

    def merge_partial_results(self, partial_results):
            """
            Merge partial results from a worker process.
            """
            res_type = partial_results["result"]
            if res_type in ["escaped", "killed"]:
                self.results[res_type] += 1

            # Merge absorbed weight
            self.results["absorbed"] += partial_results["absorbed_weight"]

            if partial_results["absorbed_coords"]:
                self.absorbed_coordinates.extend(partial_results["absorbed_coords"])

            if partial_results["final_energy"] is not None:
                self.energy_spectrum.append(partial_results["final_energy"])

            if partial_results["region_detected"]:
                self.region_count += 1



    def print_summary(self, num_particles):
            # Print a summary of results
            print(f"--- Simulation Results ---")
            print(f"  Total Particles Started: {num_particles}")
            print(f"  Particles Escaped:       {self.results['escaped']}")
            print(f"  Particles Killed (Roulette): {self.results['killed']}")
            print(f"  Total Weight Absorbed:   {self.results['absorbed']:.4f}")

            # Use the correct attribute name here:
            print(f"  Total Absorption Events Recorded: {len(self.absorbed_coordinates)}")

            print(f"  Detected within detection region (if specified): {self.region_count}")

            if self.energy_spectrum:
                avg_energy = sum(self.energy_spectrum) / len(self.energy_spectrum)
                print(f"  Average final energy:    {avg_energy:.2e} eV")
            else:
                print(f"  No particles left to calculate average final energy.")

    def get_results(self):
        return {
            "results": self.results,
            "absorbed_events": self.absorbed_events,
            "energy_spectrum": self.energy_spectrum,
            "region_count": self.region_count,
        }
