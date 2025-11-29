# PYNEUT - Python Neutronics

A Python-based Monte Carlo code for simulating neutron transport through various materials and geometries. **Intended for educational purposes only.**

## Overview

This code implements fundamental neutron physics including:
- Elastic scattering with thermal motion effects
- Energy-dependent cross sections (ENDF/B-VIII data)
- Multi-region geometry support (cylinders, planes, spheres, boxes)
- Parallel particle tracking via multiprocessing
- Optional trajectory recording for visualization

## Requirements

* Python 3.x
* See `requirements.txt` for dependencies

## Installation

```bash
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.cross_section_read import CrossSectionReader
from src.material import Material
from src.medium import Region, Cylinder, Plane
from src.simulation import simulate_single_particle
from src.vt_calc import VelocitySampler
from src.random_number_generator import RNGHandler
from multiprocessing import Pool

# Initialize cross-section reader (requires ENDF/B-VIII HDF5 data in ./endfb/)
reader = CrossSectionReader("./endfb")

# Define material
lead = Material(name="Lead", density=11.35, atomic_mass=208, atomic_weight_ratio=2.5)
sampler = VelocitySampler(lead.kg_mass)

# Define geometry
regions = [
    Region(
        surfaces=[
            Cylinder("z", 10, (0, 0, 0)),
            Plane(0, 0, -1, 10),
            Plane(0, 0, 1, 10)
        ],
        name="Lead Shield",
        priority=1,
        element="Pb208"
    ),
    Region(
        surfaces=[
            Plane(-1, 0, 0, 20), Plane(1, 0, 0, 20),
            Plane(0, -1, 0, 20), Plane(0, 1, 0, 20),
            Plane(0, 0, -1, 20), Plane(0, 0, 1, 20)
        ],
        name="Void",
        priority=0,
        is_void=True
    )
]

# Run simulation
num_particles = 100
rngs = [RNGHandler(seed=12345 + i) for i in range(num_particles)]

particle_states = [
    {
        "x": -15, "y": 0, "z": 0,
        "theta": 0, "phi": 0,
        "energy": 1e6,  # eV
        "has_interacted": False
    }
    for _ in rngs
]

args = [
    (state, reader, regions, lead.atomic_weight_ratio, 
     lead.number_density, sampler, None, False, rng)
    for state, rng in zip(particle_states, rngs)
]

with Pool() as pool:
    results = pool.map(simulate_single_particle, args)

# See main.py for complete example with tallying
```

## ENDF/B-VIII Data

Place nuclear data files in `./endfb/neutron/` directory:
- Format: HDF5 files (e.g., `Pb208.h5`, `U235.h5`)
- Source: [NNDC ENDF/B-VIII](https://www.nndc.bnl.gov/endf/)

## Code Structure

```
src/
├── cross_section_read.py  # ENDF data reader
├── material.py            # Material properties
├── medium.py              # Geometry primitives (Region, Plane, Cylinder, Sphere, Box)
├── physics.py             # Scattering physics
├── vt_calc.py            # Thermal velocity sampling
├── geometry.py            # Boundary tracking
├── simulation.py          # Main transport loop
├── tally.py              # Results accumulation
└── random_number_generator.py
```

## Features

- **Energy-dependent physics**: Uses ENDF/B-VIII cross-section libraries
- **Thermal scattering**: Accounts for target nucleus motion at 294K
- **Flexible geometry**: Boolean combinations of surfaces
- **Multiprocessing**: Parallel particle simulation
- **Trajectory tracking**: Optional particle path recording

## Usage Examples

See `QUICK_START.md` for examples including:


## Documentation

- `README.md` - This file
- `API_REFERENCE.md` - Complete function documentation
- `QUICK_START.md` - Usage examples and tutorials
- `BUGS_AND_IMPROVEMENTS.md` - Known issues and planned improvements
- `CRITICAL_FIXES.md` - Important bug fixes to apply


### Validation Status

⚠️ **This code has not yet been validated against established Monte Carlo codes (MCNP, OpenMC, etc.), but it is in the plan**

## Limitations

- Single isotope per region (no material mixtures)
- Isotropic scattering in center-of-mass frame only
- Fixed temperature (294K) for thermal motion
- No secondary particle generation from fission yet
- Mono-energetic source only

## Physics Models

- **Elastic scattering**: Includes thermal motion effects via Maxwell-Boltzmann sampling
- **Absorption**: Radiative capture (MT=102)
- **Fission**: Basic tracking (no daughter neutrons) - Intended to be implemented
- **Cross sections**: Microscopic data from ENDF/B-VIII, interpolated linearly

## Performance

Typical performance on modern CPU:
- ~0.01-0.1 seconds per particle (geometry dependent)
- Scales linearly with CPU cores via multiprocessing
- Memory: ~100 MB for 1000 particles with trajectory tracking


## Contributing

This is an educational project. Improvements welcome, especially:
- Validation against benchmark problems
- Additional physics models (anisotropic scattering, etc.)
- Performance optimizations
- Better error handling and input validation

## Future Work
- Validation in simple analytical problems
- Validation against other MC codes
- Full fission Implementation
- Material mixtures
- Multi-group energy structure
- Variance reduction techniques
- Spatial mesh tallies
- Temperature-dependent cross sections
- Visualization tools

## References

- ENDF/B-VIII Nuclear Data: https://www.nndc.bnl.gov/endf/
- OpenMC Documentation: https://docs.openmc.org/en/stable/methods/index.html
- Neutron Physics: Duderstadt & Hamilton, "Nuclear Reactor Analysis"
- Computational Methods in Reactor Shielding: James Wood

## License

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)MIT

## Contact

Youssef - Nuclear and Radiation Engineering, Alexandria University

---

**Disclaimer**: This code is for educational purposes only. Not validated for production use, regulatory submissions, or safety-critical applications.