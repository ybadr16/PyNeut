# src/simulation.py
from .geometry import calculate_nearest_boundary, calculate_void_si_max
from .physics import elastic_scattering, sample_new_direction_cosines
import numpy as np

def simulate_single_particle(args):
    """
    Simulates a single particle and returns partial tally results.
    """
    # Unpack settings
    state, reader, mediums, A, N, sampler, region_bounds, track_coordinates, rng, settings = args

    # Call the simulation kernel
    result, absorbed_coords, fissioned, _, final_energy, region_count, trajectory, total_absorbed_weight = simulate_particle(
        state, reader, mediums, A, N, sampler, region_bounds, track_coordinates=track_coordinates, rng=rng, settings=settings
    )

    # Return results using the EXACT keys expected by Tally and Validation scripts
    return {
        "result": result,
        "absorbed_weight": total_absorbed_weight,
        "absorbed_coords": absorbed_coords,
        "fissioned": fissioned,
        "final_energy": final_energy,
        "final_weight": state["weight"], # <--- ADDED: Needed for Consistency Validation
        "region_detected": region_count > 0,
        "trajectory": trajectory if track_coordinates else None,
    }

def simulate_particle(state, reader, mediums, A, N, sampler, region_bounds=None, track_coordinates=False, rng=None, settings=None):
    """
    Simulate the trajectory of a single particle.
    """
    epsilon = 1e-6
    region_count = 0
    absorbed_coordinates = []
    fission_coordinates = []
    trajectory = [] if track_coordinates else None

    # Track the accumulated weight absorbed (needed for Implicit Capture)
    total_absorbed_weight = 0.0

    # Geometry State Initialization
    x_prev, y_prev, z_prev = state["x"], state["y"], state["z"]
    u = np.sin(state["theta"]) * np.cos(state["phi"])
    v = np.sin(state["theta"]) * np.sin(state["phi"])
    w = np.cos(state["theta"])

    while True:
        # 1. TRACKING
        if track_coordinates:
            trajectory.append((state["x"], state["y"], state["z"]))

        # 2. REGION DETECTION
        if region_bounds:
            x_min, x_max, y_min, y_max, z_min, z_max = region_bounds
            if x_min <= state["x"] <= x_max and y_min <= state["y"] <= y_max and z_min <= state["z"] <= z_max:
                if not state.get("was_in_region", False):
                    region_count += 1
                    state["was_in_region"] = True

        # 3. LOCATE CURRENT MEDIUM (Original robust loop)
        current_medium = None
        max_priority = -float('inf')

        point_check = (state["x"], state["y"], state["z"])
        for medium in mediums:
            if medium.contains(*point_check) and medium.priority > max_priority:
                current_medium = medium
                max_priority = medium.priority

        if current_medium is None:
            return "escaped", absorbed_coordinates, fission_coordinates, None, state["energy"], region_count, trajectory, total_absorbed_weight

        # 4. NEAREST BOUNDARY
        nearest_point, nearest_medium, nearest_distance = calculate_nearest_boundary(state, mediums, u, v, w)

        if nearest_point is None:
            return "escaped", absorbed_coordinates, fission_coordinates, None, state["energy"], region_count, trajectory, total_absorbed_weight

        # 5. VOID HANDLING
        if current_medium.is_void:
            if nearest_point is not None:
                state["x"], state["y"], state["z"] = nearest_point
                state["x"] += epsilon * u
                state["y"] += epsilon * v
                state["z"] += epsilon * w
            continue

        # 6. GET CROSS SECTIONS
        sigma_s, sigma_a, sigma_f, Sigma_t = reader.get_cross_sections(
            current_medium.element, state["energy"], sampler, N
        )

        # 7. SAMPLE DISTANCE
        if Sigma_t <= 0:
            si = float('inf')
        else:
            si = -np.log(1 - rng.random()) / Sigma_t

        # 8. MOVE PARTICLE
        if si > nearest_distance:
            state["x"], state["y"], state["z"] = nearest_point
            state["x"] += epsilon * u
            state["y"] += epsilon * v
            state["z"] += epsilon * w
            continue

        state["x"] += si * u
        state["y"] += si * v
        state["z"] += si * w
        x_prev, y_prev, z_prev = state["x"], state["y"], state["z"]

        if not current_medium.contains(state["x"], state["y"], state["z"]):
            continue

        # --- PHYSICS KERNEL ---
        state["has_interacted"] = True

        # === MODE SELECTION ===
        if settings and settings.use_implicit_capture:
            # === IMPLICIT CAPTURE ===

            # A. Russian Roulette
            if state["weight"] < settings.weight_cutoff:
                if rng.random() < settings.roulette_survival_prob:
                    state["weight"] /= settings.roulette_survival_prob
                else:
                    return "killed", absorbed_coordinates, fission_coordinates, None, state["energy"], region_count, trajectory, total_absorbed_weight

            # B. Weight Reduction
            if Sigma_t > 0:
                p_absorb = (sigma_a + sigma_f) / Sigma_t
                p_scatter = sigma_s / Sigma_t
            else:
                p_absorb = 0.0
                p_scatter = 1.0

            weight_loss = state["weight"] * p_absorb
            total_absorbed_weight += weight_loss

            if weight_loss > 0:
                # Appending (x, y, z, weight_loss)
                absorbed_coordinates.append((state["x"], state["y"], state["z"], weight_loss))

            state["weight"] *= p_scatter

            if state["weight"] <= 0:
                 return "killed", absorbed_coordinates, fission_coordinates, None, state["energy"], region_count, trajectory, total_absorbed_weight

        else:
            # === ANALOG MONTE CARLO (Old Logic) ===
            interaction_prob = rng.random()

            # Calculate probabilities safely
            p_scatter = sigma_s / Sigma_t if Sigma_t > 0 else 0
            p_absorb = sigma_a / Sigma_t if Sigma_t > 0 else 0

            if interaction_prob < p_scatter:
                # Scattering
                pass
            elif interaction_prob < (p_scatter + p_absorb):
                # Absorption
                weight_deposited = state["weight"]
                total_absorbed_weight += weight_deposited
                absorbed_coordinates.append((state["x"], state["y"], state["z"], weight_deposited))

                return "absorbed", absorbed_coordinates, fission_coordinates, None, state["energy"], region_count, trajectory, total_absorbed_weight

            # Fission falls through here (Survival)

        # --- SCATTERING KINEMATICS ---
        E_prime, mu_cm, mu_lab = elastic_scattering(state["energy"], A, sampler, rng)

        # Calculate new direction cosines
        u, v, w, state["phi"] = sample_new_direction_cosines(u, v, w, mu_lab, rng)

        # FIX: Update Global Theta using the new z-direction cosine (w)
        # Using np.arccos(mu_lab) was incorrect (resetting to relative scattering angle)
        state["theta"] = np.arccos(w)
        state["energy"] = E_prime
