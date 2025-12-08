# main.py
from src.cross_section_read import CrossSectionReader
from src.vt_calc import VelocitySampler
from src.simulation import simulate_single_particle
from src.material import Material
from src.medium import Region, Plane, Cylinder
from src.tally import Tally
from src.random_number_generator import RNGHandler
from src.settings import Settings
from src.mesh import MeshTally

from multiprocessing import Pool
import json
import time
import numpy as np
from collections import deque

def main():
    # 1. SETUP
    base_path = "./endfb"
    reader = CrossSectionReader(base_path)

    # --- OPTIMIZATION: PRE-LOAD CACHE ---
    # We trigger the load here so the worker processes inherit
    # the cached data, preventing every worker from reading the disk.
    print("Pre-loading cross sections into memory...")
    # Add any elements used in your 'mediums' list below
    elements_to_load = ["B10"]

    for elem in elements_to_load:
        try:
            # We request a dummy energy (1.0 eV) to trigger the file read
            # MT=2 (Elastic), MT=102 (Capture) are the essentials
            reader.get_cross_section(elem, 2, 1.0)
            reader.get_cross_section(elem, 102, 1.0)
            # reader.get_cross_section(elem, 18, 1.0) # Uncomment if using Fissionable material
        except Exception as e:
            print(f"Warning: Could not pre-load {elem}: {e}")
    print("Pre-loading complete.")
    # ------------------------------------

    #lead = Material(name="Lead", density=11.35, atomic_mass=208, atomic_weight_ratio=2.5)
    boron = Material(name="Boron", density=2.34, atomic_mass=10, atomic_weight_ratio=10.0)

    # Correct: Use raw number density (atoms/cm^3)
    N = boron.number_density
    A = boron.atomic_weight_ratio
    mass_in_kg = boron.kg_mass
    sampler = VelocitySampler(mass=mass_in_kg)

    mediums = [
        Region(
            surfaces=[
                Cylinder("z", 10, (0, 0, 0)),
                Plane(0, 0, -1, 10),
                Plane(0, 0, 1, 10)
            ],
            name="Cylinder",
            priority=1,
            # MAKE SURE THIS MATCHES YOUR FILENAME (e.g. "Pb-208" or "82208")
            element="B10"
        ),
        Region(
            surfaces=[
                Plane(-1, 0, 0, 20), Plane(1, 0, 0, 20),
                Plane(0, -1, 0, 40), Plane(0, 1, 0, 40),
                Plane(0, 0, -1, 20), Plane(0, 0, 1, 20)
            ],
            name="Void",
            priority=0,
            is_void=True
        )
    ]

    # Use Shielding mode (Implicit Capture) for better heatmaps
    settings = Settings(mode="shielding", particles=100000)

    # --- INITIALIZE MESH TALLY ---
    # Create a 3D grid around the cylinder (Radius 10, Height 20)
    # Bounds: -15 to 15 ensures we see the surroundings too
    # Dims: 40x40x40 voxels
    mesh = MeshTally(
        x_bounds=(-15, 15),
        y_bounds=(-15, 15),
        z_bounds=(-15, 15),
        dims=(40, 40, 40)
    )
    # -----------------------------

    num_particles = settings.num_particles
    rngs = [RNGHandler(seed=12345 + i) for i in range(num_particles)]
    tally = Tally()

    particle_states = [
        {
            "x": -10.5, "y": 0.0, "z": 0.0,
            "theta": rng.uniform(0, np.pi),
            "phi": rng.uniform(0, 2 * np.pi),
            "has_interacted": False,
            "energy": 50,  # 50 eV
            "weight": 1.0
        }
        for rng in rngs
    ]

    all_trajectories = {}
    region_bounds = (14.9, 15.1, -15, 15, -15, 15)
    track = False

    args = [
        (state, reader, mediums, A, N, sampler, region_bounds, track, rng, settings)
        for state, rng in zip(particle_states, rngs)
    ]

    print(f"Starting simulation of {num_particles} particles...")

    sim_start_time = time.perf_counter()
    with Pool() as pool:
        partial_results = pool.map(simulate_single_particle, args)
    sim_end_time = time.perf_counter()

    # Merge results and SCORE MESH
    for idx, result in enumerate(partial_results):
        tally.merge_partial_results(result)

        # --- FILL THE MESH ---
        # Implicit capture returns (x, y, z, weight_lost) in 'absorbed_coords'
        if result["absorbed_coords"]:
            for event in result["absorbed_coords"]:
                x, y, z, w = event
                mesh.score(x, y, z, w)
        # ---------------------

        if result["trajectory"]:
            all_trajectories[idx + 1] = result["trajectory"]

    # Export Data
    mesh.write_vtk("dose_map.vtk") # <--- Generates the visualization file

    with open("all_trajectories.json", "w") as f:
        json.dump(all_trajectories, f, indent=4)

    tally.print_summary(num_particles)

    print(f"Particle simulation time: {sim_end_time - sim_start_time:.2f} seconds")
    print(f"Rate: {num_particles / (sim_end_time - sim_start_time):.0f} particles/s")

if __name__ == "__main__":
    main()
