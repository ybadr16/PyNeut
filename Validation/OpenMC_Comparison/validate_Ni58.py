"""
PyNeut vs OpenMC — Ni58 validation case

Cases
-----
  ni58_sphere_14mev : Ni58 sphere R=5 cm, E=14 MeV (MT 28 and the (n,p)/(n,a)
                      partials are 40% of the total)

OpenMC reference is computed live against the local data library.
"""
from _common import validate_case, print_result

_CASES = [
    {'name': 'ni58_sphere_14mev', 'el': 'Ni58', 'rho': 8.908, 'A': 57.438,
     'E': 14e6, 'geo': 'sphere', 'dims': [5.0]},
]


def run(n_particles=10_000):
    results = []
    for case in _CASES:
        print(f"  Running {case['name']} …")
        r = validate_case(case, n_particles, isotope='Ni58')
        print_result(r)
        results.append(r)
    return results


if __name__ == '__main__':
    print('\n=== Ni58 Validation ===')
    run()
