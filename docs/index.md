# Monte Carlo Neutron Transport Simulation

A Python-based Monte Carlo code for simulating neutron transport through various materials and geometries. **Intended for educational purposes only.**

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Quick Links
- [📖 API Reference](https://ybadr16.github.io/PYNEUT/api-reference) - Complete function documentation
- [💻 GitHub Repository](https://github.com/ybadr16/PYNEUT)
- [🐛 Report Issues](https://github.com/ybadr16/PYNEUT/issues)

---

## Features

✅ Energy-dependent cross sections (ENDF/B-VIII data)  
✅ Thermal scattering with Maxwell-Boltzmann sampling  
✅ Multi-region geometry (cylinders, planes, spheres, boxes)  
✅ Parallel processing via multiprocessing  
✅ Optional trajectory tracking  

---

## Installation

```bash
git clone https://github.com/ybadr16/PYNEUT.git
cd PYNEUT
pip install -e .
```

See [requirements.txt](https://github.com/ybadr16/PYNEUT/blob/main/requirements.txt) for dependencies.

---

## Quick Example

```python
from src.cross_section_read import CrossSectionReader
from src.material import Material
from src.medium import Region, Cylinder, Plane
from src.simulation import simulate_single_particle
from src.vt_calc import VelocitySampler
from src.random_number_generator import RNGHandler
from multiprocessing import Pool
import numpy as np

# Initialize cross-section reader (requires ENDF/B-VIII data in ./endfb/)
reader = CrossSectionReader("./endfb")

# Define material
lead = Material(name="Lead", density=11.35, atomic_mass=208, atomic_weight_ratio=2.5)
sampler = VelocitySampler(lead.kg_mass)

# Define geometry - lead cylinder shield
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

# Create particles
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

# Prepare arguments for multiprocessing
args = [
    (state, reader, regions, lead.atomic_weight_ratio, 
     lead.number_density, sampler, None, False, rng)
    for state, rng in zip(particle_states, rngs)
]

# Run simulation
with Pool() as pool:
    results = pool.map(simulate_single_particle, args)

# Process results
from src.tally import Tally
tally = Tally()
for result in results:
    tally.merge_partial_results(result)

tally.print_summary(num_particles)
```

---

## ENDF/B-VIII Nuclear Data

Place nuclear data files in `./endfb/neutron/` directory:
- Format: HDF5 files (e.g., `Pb208.h5`, `U235.h5`)
- Source: [NNDC ENDF/B-VIII](https://www.nndc.bnl.gov/endf/)

---

## Code Structure

```
src/
├── cross_section_read.py  # ENDF data reader
├── material.py            # Material properties
├── medium.py              # Geometry primitives (Region, Plane, Cylinder, Sphere)
├── physics.py             # Scattering physics
├── vt_calc.py            # Thermal velocity sampling
├── geometry.py            # Boundary tracking
├── simulation.py          # Main transport loop
├── tally.py              # Results accumulation
└── random_number_generator.py
```

---

## Physics Models

- **Elastic scattering**: Thermal motion via Maxwell-Boltzmann sampling
- **Absorption**: Radiative capture (MT=102)
- **Fission**: Basic event tracking
- **Cross sections**: ENDF/B-VIII data, linearly interpolated
- **Temperature**: 294K (configurable in code)

---

## Current Limitations

- Single isotope per region (no material mixtures)
- Isotropic scattering in center-of-mass frame
- Fixed temperature (294K)
- No secondary particles from fission
- Mono-energetic source

See [GitHub Issues](https://github.com/ybadr16/PYNEUT/issues) for planned enhancements.

---

## Testing

```bash
# Run geometry tests
pytest tests/test_cylinder.py -v

# Run your simulation
python main.py
```

---

## Performance

Typical performance on modern CPU:
- ~0.01-0.1 seconds per particle (geometry dependent)
- Scales linearly with CPU cores via multiprocessing
- Memory: ~100 MB for 1000 particles with trajectory tracking

---

## Contributing

Contributions welcome! Please:
1. Check [existing issues](https://github.com/ybadr16/PYNEUT/issues)
2. Open a new issue to discuss major changes
3. Submit pull requests



---

## Contact

**Youssef**  
Nuclear and Radiation Engineering  
Alexandria University, Egypt

[GitHub Repository](https://github.com/ybadr16/PYNEUT) • [Report Issues](https://github.com/ybadr16/PYNEUT/issues)

---

<p align="center">
  <em>⚠️ Disclaimer: For educational purposes only. Not validated for production use.</em>
</p>
