# src/physics.py
import numpy as np

# Constants
m_n = 1.674927471e-27  # Neutron mass in kg
eV_to_J = 1.60217663e-19 # Conversion factor

def calculate_mu_lab(mu_cm, E, E_prime, E_cm_prime, A):
    """
    Matches OpenMC Eq :label: angle-com-to-lab
    """
    if E_prime <= 0: return 0.0

    term1 = mu_cm * np.sqrt(E_cm_prime / E_prime)
    term2 = (1 / (A + 1)) * np.sqrt(E / E_prime)
    mu_lab = term1 + term2

    # Clamp to [-1, 1] for numerical stability
    return max(-1.0, min(1.0, mu_lab))

def calculate_E_cm_prime(initial_energy, A, sampler):
    """
    Calculates Center-of-Mass energy.
    Partially matches OpenMC 'Elastic Scattering' section,
    but simplifies vector math to scalars for thermal approximation.
    """
    E = initial_energy

    # Unit Conversion
    E_joules = E * eV_to_J
    v_l = np.sqrt(2 * E_joules / m_n)

    # Sample target velocity (Free Gas approximation)
    # OpenMC uses 400kT threshold (approx 10 eV is fine)
    v_t = sampler.sample_velocity(vn=v_l) if E < 10 else 0.0

    # Calculate CM velocity (Scalar approximation of OpenMC vector Eq :label: velocity-com)
    # v_cm = (v_n + A*v_t) / (A+1)
    v_cm = (v_l + A * v_t) / (A + 1)

    # Neutron velocity in CM frame
    # OpenMC Eq :label: velocity-neutron-com
    v_l_cm = abs(v_l - v_cm)

    # Calculate E_cm_prime
    # E = 0.5 * m * v^2
    E_cm_prime_joules = 0.5 * m_n * v_l_cm**2

    # Convert back to eV
    E_cm_prime = E_cm_prime_joules / eV_to_J

    return E_cm_prime

def calculate_E_prime(E_cm_prime, initial_energy, A, rng):
    """
    Matches OpenMC Eq :label: energy-com-to-lab
    """
    E = initial_energy

    # Sample mu_cm (Isotropic in CM for elastic scattering)
    try:
        mu_cm = 2 * rng.random() - 1
    except:
        mu_cm = 2 * rng.uniform(0, 1) - 1

    # OpenMC Formula
    numerator = E + 2 * mu_cm * (A + 1) * np.sqrt(E * E_cm_prime)
    term2 = numerator / ((A + 1)**2)
    E_prime = E_cm_prime + term2

    return E_prime, mu_cm

def elastic_scattering(initial_energy, A, sampler, rng):
    E_cm_prime = calculate_E_cm_prime(initial_energy, A, sampler)
    E_prime, mu_cm = calculate_E_prime(E_cm_prime, initial_energy, A, rng)

    # Safety clamp
    E_prime = max(1e-5, E_prime)

    mu_lab = calculate_mu_lab(mu_cm, initial_energy, E_prime, E_cm_prime, A)
    return E_prime, mu_cm, mu_lab

def sample_new_direction_cosines(u, v, w, mu_lab, rng):
    """
    Matches OpenMC Eq :label: post-collision-angle
    Includes singularity check for w ~ 1.
    """
    phi = 2 * np.pi * rng.random()
    sin_theta = np.sqrt(max(0.0, 1.0 - mu_lab**2))

    # Check for singularity (particle moving parallel to z-axis)
    # This prevents division by zero in the standard rotation matrix
    if abs(w) >= 0.999999:
        sign = 1.0 if w > 0 else -1.0
        u_new = sin_theta * np.cos(phi)
        v_new = sin_theta * np.sin(phi)
        w_new = sign * mu_lab
    else:
        denom = np.sqrt(max(1e-12, 1.0 - w**2))
        u_new = (mu_lab * u) + (sin_theta / denom) * (u * w * np.cos(phi) - v * np.sin(phi))
        v_new = (mu_lab * v) + (sin_theta / denom) * (v * w * np.cos(phi) + u * np.sin(phi))
        w_new = (mu_lab * w) - (sin_theta * denom * np.cos(phi))

    # Normalize
    norm = np.sqrt(u_new**2 + v_new**2 + w_new**2)
    return u_new/norm, v_new/norm, w_new/norm, phi
