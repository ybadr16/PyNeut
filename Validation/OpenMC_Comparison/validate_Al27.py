"""
PyNeut vs OpenMC — Al27 validation case

Cases
-----
  al27_slab_1mev  : Al27 thin slab half-thickness=0.5 cm, E=1 MeV
  al27_slab_14mev : Al27 slab half-thickness=5 cm, E=14 MeV (MT 5, 22, 28,
                    and the charged-particle channels are 30% of the total)

OpenMC reference is computed live against the local data library.
"""
from _common import validate_case, print_result

_CASES = [
    {'name': 'al27_slab_1mev', 'el': 'Al27', 'rho': 2.70, 'A': 26.74975,
     'E': 1e6, 'geo': 'slab', 'dims': [0.5]},
    {'name': 'al27_slab_14mev', 'el': 'Al27', 'rho': 2.70, 'A': 26.74975,
     'E': 14e6, 'geo': 'slab', 'dims': [5.0]},
]


def run(n_particles=10_000):
    results = []
    for case in _CASES:
        print(f"  Running {case['name']} …")
        r = validate_case(case, n_particles, isotope='Al27')
        print_result(r)
        results.append(r)
    return results


if __name__ == '__main__':
    print('\n=== Al27 Validation ===')
    run()
