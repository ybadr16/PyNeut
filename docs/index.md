<p align="center">
  <img src="assets/pyneut-logo.jpg" alt="PyNeut logo" width="420">
</p>

# PyNeut validation report

PyNeut is a continuous-energy Monte Carlo neutron transport code in Python,
driven by ENDF/B-VIII.0 data. This page explains how PyNeut is checked, gives
every result, and describes the physics behind the numbers and its limits.
Installation and usage are in the [README](https://github.com/ybadr16/PyNeut#readme),
the [Quick Start](QUICK_START.md) and the [API reference](api-reference.md).

**Contents:** [Method](#method) ·
[Fixed source](#fixed-source-transport-vs-openmc) ·
[Mixtures](#material-mixtures) ·
[Criticality](#criticality-k-eigenvalue) ·
[Pin cell](#beavrs-pin-cell-k) ·
[Vector engine](#vector-engine) ·
[Tallies](#tallies-vs-openmc) ·
[Analytic and data checks](#analytic-and-data-layer-checks) ·
[Physics](#physics-models) ·
[Limitations](#limitations)

<p align="center">
  <img src="images/validation_summary.png"
       alt="Every PyNeut validation case plotted against OpenMC as a z-score; all sixteen fall inside the two-sigma agreement band"
       width="88%">
</p>

---

## Method

The OpenMC (0.15) reference results are generated **live**, from the same
ENDF/B HDF5 files PyNeut reads, with identical geometry and source. The
scripts are in `Validation/OpenMC_Comparison/`. The comparison is built to
avoid three common traps:

- **Graded by statistical significance, not percent difference.** Both codes
  report a standard error: batch means for PyNeut, OpenMC's own batch
  statistics for OpenMC. Agreement is graded by
  `z = |OpenMC − PyNeut| / √(σ²_OMC + σ²_PyN)`:
  **OK ≤ 2σ, WARNING ≤ 4σ, FAIL > 4σ**. A large z flags a real bias even when
  the percent difference looks tiny.
- **The same estimator in both codes.** The mean escape energy is taken from
  the same energy histogram in both codes, so the binning bias cancels. Without
  this, two cases that looked like sub-0.3 % passes turned out to be artifacts
  of the reference binning.
- **Spectrum shape is graded separately from the integrals.** The leakage
  spectrum is compared with a coarse-group χ²/dof (GOOD ≤ 2, FAIR ≤ 5,
  POOR > 5), reported independently of the integral status.

**Matched masses.** Every case takes the target mass from the data file's
atomic-weight ratio (AWR), and `_common.py` refuses a case whose declared AWR
disagrees with the file. Six case files once supplied the molar mass in g/mol
instead. That gave the two codes masses and number densities that differed by
0.5–1.0 %, and caused a spurious 2.1σ warning on `c12_cyl_2mev`.

The committed results were produced against a data snapshot fingerprinted in
`validation_manifest.json` (50 files from OpenMC's official ENDF/B-VIII.0 HDF5
library, measured 2026-10-25). The pin-cell and k-eigenvalue references are
from an earlier snapshot (48 files, 2026-08). A cross-code
comparison holds only for one data snapshot. The pin cell, for example, moves
by about 330 pcm across an evaluation refresh.

---

## Fixed-source transport vs OpenMC

N = 10 000 histories per case, from a monoenergetic isotropic point source at
the origin. Dimensions are in cm, and slabs are 100 × 100 cm in the other two
directions. Source: `Validation/OpenMC_Comparison/validation_results.csv`.

| Case | E | Regime | Leak z | E z | Integral | Spectrum (χ²/dof) |
|------|--:|--------|------:|----:|:--------:|:-----------------:|
| Pb208 sphere R=10 | 2 MeV | fast, elastic | 0.9 | 0.3 | OK | GOOD (1.14) |
| Pb208 sphere R=10 | 5 MeV | fast | 1.1 | 0.1 | OK | GOOD (0.63) |
| Pb208 sphere R=10 | 14 MeV | fast, (n,2n) | 0.4 | 0.6 | OK | GOOD (0.49) |
| Fe56 slab, 4 thick | 14 MeV | fast, (n,xn) | 0.3 | 0.6 | OK | GOOD (0.43) |
| Fe56 sphere R=10 | 1 MeV | fast | 0.8 | 0.4 | OK | GOOD (1.0) |
| Fe56 sphere R=15 | 25 keV | intermediate | 0.1 | 0.6 | OK | GOOD (0.67) |
| Be9 sphere R=10 | 14 MeV | fast, (n,2n) | 0.9 | 0.1 | OK | GOOD (0.38) |
| Al27 slab, 1 thick | 1 MeV | fast | 0.0 | 0.2 | OK | GOOD (0.64) |
| Al27 slab, 10 thick | 14 MeV | fast, MT 5/22/28 | 0.3 | 0.6 | OK | GOOD (2.0) |
| C12 cylinder R=10, height 20 | 2 MeV | fast | 0.0 | 0.4 | OK | GOOD (0.77) |
| O16 sphere R=20 | 10 MeV | fast | 0.4 | 1.6 | OK | GOOD (0.71) |
| O16 sphere R=15, liquid density | 14 MeV | fast, MT 22/28, (n,α) | 0.7 | 0.1 | OK | GOOD (1.8) |
| Ni58 sphere R=5 | 14 MeV | fast, MT 28, (n,p) | 0.9 | 0.3 | OK | GOOD (0.69) |
| B10 sphere R=1 | 1 keV | epithermal | 0.8 | 1.3 | OK | GOOD (0.68) |
| Cd112 sphere R=2 | 0.0253 eV | thermal | 0.7 | 0.3 | OK | GOOD (0.06) |
| C12 sphere R=5 | 0.0253 eV | thermal | 1.2 | 0.4 | OK | GOOD (0.26) |

**16 OK, 0 warnings, 0 failures; all 16 spectrum shapes GOOD.** The table was regenerated on 2026-10-25 against OpenMC's official
ENDF/B-VIII.0 HDF5 library, with per-neutron leakage spectra and with every
reaction counted in the total cross section. The three 14 MeV rows for Al27,
O16 and Ni58 were added to test that fix (see
[Reactions above 5 MeV](#reactions-above-5-mev)), and the Fe56 slab is no
longer a warning. The spectrum column uses the grading fixed for B10 below.
The B10 row moved within its error bars in both codes
relative to the earlier data snapshot, because that library's B10 file is
not byte-identical to the earlier one.

<p align="center">
  <img src="images/xs_verification.png" alt="U235 cross sections read by PyNeut overlaid on OpenMC" width="49%">
  <img src="images/leakage_spectra.png" alt="Leakage spectra from PyNeut and OpenMC" width="49%">
</p>

**Fast neutrons (the intended domain).** Leakage and mean escape energy agree
within about 2σ everywhere. The spectrum shape is GOOD for the elastic,
discrete-inelastic and resonance cases.

**Thermal.** Below 10 eV elastic scattering uses free-gas kinematics: the
target's 294 K motion is sampled, so neutrons can gain energy and settle into
the medium's Maxwellian. `c12_thermal` and `cd112_thermal` agree within 0.4σ
on mean escape energy, with GOOD shape. These two used to be the suite's
failures (c12 was 86σ off) with a stationary-target kernel. After that fix a
~3 % hot bias remained. It came from three compounding errors:

1. The flight cross section was Doppler-broadened again, although the library
   data is already broadened.
2. The free-gas branch probability used the raw neutron speed instead of the
   dimensionless ratio βv.
3. The target mass was taken as the AWR instead of AWR·mₙ.

All three are fixed. OpenMC also runs free-gas here (no S(α,β)).

**(n,2n) and the POOR spectrum shapes.** The two (n,2n) spheres agree on
leakage and mean escape energy, but their spectrum shapes were POOR
(χ²/dof 29 and 46). Fe56 at 14 MeV shows the same effect at lower strength
(5.5). The single-collision emission
itself has been checked against OpenMC point by point at the incident-energy
grid nodes (`validate_secondary.py`, outgoing-energy χ²/dof ≈ 1 for Be9, Fe56
and Pb208 at 14 MeV). Three things are applied:

- the CM→lab boost for reactions given in the CM frame,
- unit-base outgoing-energy interpolation between incident grids,
- the correlated emission angle (Law 61 / Kalbach–Mann).

The Maxwellian-evaporation fallback is used **0 %** of the time. (An earlier
"18 % / χ² ≈ 86" emission bias was a diagnostic artifact: OpenMC's incident
block *i* was compared against PyNeut's block *i+1*.)

**The POOR (n,2n) shapes were the tally, not the physics.** Until October
2026 these three spectra were POOR (χ²/dof 29, 46 and 5.5), with PyNeut's
spectrum built by histogramming each *history's* weight-averaged escape energy.
When (n,xn) makes one history leak several neutrons, say at 0.5 and 3 MeV,
that puts all of its weight in one bin near 1.75 MeV, while OpenMC scores each
neutron at its own energy. Total leakage and mean escape energy stay exactly
right; only the shape narrows. To test it, the same PyNeut histories (the
suite's own seeds) were binned both ways and graded against the same live
OpenMC spectrum:

| Case | χ²/dof, per-history binning | χ²/dof, per-neutron | Histories leaking > 1 unit of weight |
|------|--:|--:|--:|
| Pb208 (n,2n), 14 MeV | 28.57 | **0.52** | 5 038 |
| Be9 (n,2n), 14 MeV | 46.28 | **0.38** | 4 863 |
| Fe56 slab, 14 MeV | 5.47 | **1.12** | 1 490 |
| Pb208, 2 MeV (control) | 0.90 | 0.90 | 0 |
| Fe56, 1 MeV (control) | 1.02 | 1.02 | 0 |

The per-history column reproduces the previously committed values exactly,
so these are the same histories. The single-neutron controls are identical
both ways, as they must be. The suite now scores leakage per neutron
(`pyneut.tallies`). The earlier explanation, that the emitted neutrons are
sampled independently, is withdrawn: the emission was right all along.

The Fe56 slab's integral warning had a separate cause, the missing reactions
described below.

**B10 at 1 keV was POOR (χ²/dof 15) because of the grading, not the
physics.** OpenMC runs analog, so its leakage spectrum below about 0.6 keV,
which takes two or more scatters in a near-black absorber to reach, was empty:
0 ± 0 in eight of the twenty groups. PyNeut runs with implicit capture and
resolves that tail at 1e-8 to 4e-5 per source neutron. Each empty group was
scored as an exact zero, and those eight groups gave 235 of the χ² of 240. The
tail PyNeut reports is worth 0.86 expected counts in OpenMC's 10 000
histories, so OpenMC sees none of it 42 % of the time. In the eight groups
where OpenMC has counts, χ²/dof was 0.66.

The grader now follows Pearson's rule. Groups are merged until each holds at
least five expected analog counts, and OpenMC's variance is never taken below
the Poisson variance of that expectation (`compare_spectra` in
`_common.py`). B10 is then GOOD (0.68). Every other case moved by at most
0.33 in χ²/dof and kept its grade, apart from Al27 at 14 MeV going from 2.1
(FAIR) to 2.0 (GOOD). A ±20 % tilt in a well-populated spectrum still grades
POOR (`tests/test_spectrum_grading.py`).

Each case's spectrum can be inspected directly. `run_all.py --plots` writes
one panel per case to `images/cases/`:

<p align="center">
  <img src="images/cases/spectrum_pb208_n2n.png"
       alt="Pb208 14 MeV leakage spectrum from both codes with their ratio below; the two agree across the whole range, chi-squared per degree of freedom 0.52"
       width="62%">
</p>

### Reactions above 5 MeV

Until October 2026 PyNeut built the total cross section from elastic,
MT 16/17/51–91, MT 102–107 and fission only. Every other reaction was
missing: MT 5 (n,anything), MT 22 (n,nα), MT 28 (n,np) and the
charged-particle partials MT 600–849. Below 5 MeV this changed nothing (every
nuclide matched OpenMC's total to better than 0.01 %), but at 14 MeV the total
was 2.8 % low for Fe56, 7.9 % for O16, 18 % for Al27 and 22 % for Ni58 and
Fe54. Flights were too long and the missing channels were never sampled.

The reader now classifies each reaction from its products, as OpenMC does.
Redundant reactions are skipped. A reaction with a neutron among its
products is transported with its ENDF secondary distribution, emitting as
many neutrons as its tabulated yield gives, and one without is absorption.
The total is then the sum of the non-redundant reactions, and it matches
OpenMC's to 1e-8 at every energy for every nuclide tested
(`tests/test_reactions.py`).

The three cases below were added for this. Each was run on the previous code
and on the fixed code against the same live OpenMC result (χ²/dof as graded
before the low-count fix below):

| Case | Before: leak z / E z / χ²/dof | After: leak z / E z / χ²/dof |
|------|--:|--:|
| Fe56 slab, 4 thick, 14 MeV | 2.5 / 1.4 / 1.1 (WARN) | 0.3 / 0.6 / 0.44 |
| Al27 slab, 10 thick, 14 MeV | 2.6 / **23** / **33** (FAIL) | 0.3 / 0.6 / 2.1 |
| O16 sphere R=15, 14 MeV | 1.9 / **12** / **12** (FAIL) | 0.7 / 0.1 / 1.9 |
| Ni58 sphere R=5, 14 MeV | 1.1 / **33** / **69** (FAIL) | 0.9 / 0.3 / 0.77 |

The leakage itself was nearly right before, because the missing reactions
mostly emit a neutron too. The mean escape energy and the spectrum show the
error: the neutrons that should have lost energy in (n,np) or (n,nα) left
with too much. The iron tally model in [Tallies vs OpenMC](#tallies-vs-openmc)
now agrees as well.

Below 5 MeV the newly counted reactions are under 1e-6 of the U235 and U238
totals, so the k_eff results are unchanged.

A reaction whose sampled neutron count is zero (MT 5's yield is fractional,
about 0.57 for Fe56 at 14 MeV) ends the history. OpenMC instead scales the
neutron's weight by the yield. Both are unbiased. PyNeut does not score that
loss as absorption, matching OpenMC's absorption score, and counts it in the
`removal_events` and `removed_weight` counters instead.

---

## Material mixtures

`validate_mixture.py` tests per-collision isotope selection on a B4C-like
**B10 + C12** sphere (a strong absorber plus a moderator), with identical
number densities in both codes, at N = 40 000:

| Metric | OpenMC | PyNeut | z |
|--------|-------:|-------:|--:|
| leakage | 0.67970 ± 0.00275 | 0.67931 ± 0.00171 | 0.1 |
| mean escape energy (eV) | 766 009 ± 1 333 | 766 323 ± 1 087 | 0.2 |

The spectrum shape is GOOD (χ²/dof = 1.5). Both codes moved by the same
amount from the earlier data snapshot.

---

## Criticality (k-eigenvalue)

`validate_keff.py` compares PyNeut's fission-source power iteration (with the
collision estimator) against OpenMC's eigenvalue mode. The model is a bare
U235 metal sphere at 18.74 g/cm³, with 2 000 neutrons per generation, 120
generations and 20 inactive:

| Sphere R (cm) | PyNeut k_eff | OpenMC k_eff | z | Status |
|--------------:|-------------:|-------------:|--:|:------:|
| 6.0 | 0.73854 ± 0.00205 | 0.73915 ± 0.00119 | 0.3 | OK |
| 8.7 | 1.02582 ± 0.00222 | 1.02332 ± 0.00139 | 1.0 | OK |
| 11.0 | 1.22578 ± 0.00248 | 1.22636 ± 0.00165 | 0.2 | OK |

<p align="center">
  <img src="images/keff_vs_radius.png" alt="k_eff versus bare U235 sphere radius for PyNeut and OpenMC" width="49%">
  <img src="images/keff_convergence.png" alt="Fission-source convergence of k_eff over successive generations" width="49%">
</p>

The radius sweep crosses k = 1 at **R = 8.42 ± 0.01 cm**, from a weighted
linear fit to the radii between 8.0 and 8.7 cm. That
is the critical radius of *pure* U235 metal. Godiva's 8.7407 cm belongs to the
93.7 %-enriched alloy, which also contains U238 and U234. PyNeut stays within
z ≤ 1.0 of OpenMC over the range, even though it uses a Watt χ instead of
OpenMC's tabulated ENDF χ.

**ICSBEP fast-metal benchmarks** (`validate_keff.py --benchmarks`). Each one
runs through both codes and is compared with the published k_eff ≡ 1.0000:

| Benchmark | Fissile | PyNeut k_eff | OpenMC k_eff | z (OMC) | z (bench) |
|-----------|:-------:|-------------:|-------------:|:-------:|:---------:|
| Godiva (HEU-MET-FAST-001) | U235 | 1.0008 ± 0.0017 | 1.0004 ± 0.0011 | 0.2 | 0.4 |
| Jezebel-23 (U233-MET-FAST-001) | U233 | 1.0008 ± 0.0018 | 1.0007 ± 0.0013 | 0.1 | 0.4 |
| Jezebel (PU-MET-FAST-001) | Pu239 | 1.0000 ± 0.0022 | 0.9991 ± 0.0012 | 0.4 | 0.0 |

<p align="center">
  <img src="images/icsbep_benchmarks.png"
       alt="PyNeut and OpenMC k_eff for the Godiva, Jezebel-23 and Jezebel benchmarks against the published unity value"
       width="72%">
</p>

Across three fissile isotopes, PyNeut is within **0.4σ of the evaluated
k_eff** and **0.4σ of OpenMC**. OpenMC itself lands within 0.5σ of every
evaluated value, which confirms the benchmark specifications.

---

## BEAVRS pin cell (k∞)

`validate_pincell.py` models a BEAVRS hot-zero-power pin cell: UO2 fuel, a
helium gap, Zircaloy-4 clad and borated water, inside an all-reflective cube,
which makes it an infinite lattice. Both codes run **matched physics**:
free-gas hydrogen at 294 K with no S(α,β). In a near-critical reflective
lattice, the single-run error bar understates the true scatter, so each code
is repeated over five independent seeds:

| | k∞ (mean of 5 seeds) |
|--|--:|
| PyNeut | 1.29056 ± 0.00074 |
| OpenMC | 1.28932 ± 0.00034 |
| difference | **+124 ± 81 pcm, z = 1.53** |

Source: `pincell_reference.json`. An earlier ~1500 pcm under-prediction was
closed by the free-gas fixes. A later +580 pcm residual turned out to be a
mismatch between the data snapshots of the two codes' reference runs. Because
PyNeut has no S(α,β), this validates the transport. It is not a prediction of
a real lattice's k∞.

<p align="center">
  <img src="images/geometry_pincell.png"
       alt="Material map of the BEAVRS pin cell: UO2 fuel, helium gap, Zircaloy-4 clad and borated water"
       width="60%">
</p>

The material map above uses the same containment test the tracker uses, so it
shows exactly what the transport sees, priorities included. You can render
one for any model with `tools/plot_geometry.py`.

---

## Vector engine

`pyneut/vector_transport.py` advances the whole particle bank with NumPy, one
event (a free flight followed by a crossing or collision) per pass. It
implements the same physics as the per-history kernel in `simulation.py`.
Because RNG draws are consumed in a different order, the two engines are
compared statistically. `run_all_vector.py` grades all 16 fixed-source cases
at N = 100 000 against the same OpenMC references
(`validation_results_vector.csv`):

| Case | Leak z | E z | Status | Vector hist/s | Scalar hist/s | Speed-up |
|------|------:|----:|:------:|--------------:|--------------:|---------:|
| pb208_elastic | 0.8 | 0.3 | OK | 362 k | 8.4 k | 43× |
| pb208_5mev | 0.9 | 1.0 | OK | 192 k | 7.0 k | 27× |
| pb208_n2n | 0.5 | 0.7 | OK | 21.6 k | 2.8 k | 7.6× |
| fe56_slab_14mev | 0.8 | 0.3 | OK | 37.8 k | 6.7 k | 5.6× |
| fe56_sphere_1mev | 0.8 | 2.0 | OK | 197 k | 7.5 k | 26× |
| fe56_sphere_25kev | 0.4 | 1.2 | OK | 1.30 M | 46.1 k | 28× |
| be9_n2n_14mev | 1.3 | 0.4 | OK | 16.6 k | 1.8 k | 9.0× |
| al27_slab_1mev | 0.0 | 0.3 | OK | 1.05 M | 26.1 k | 40× |
| al27_slab_14mev | 0.2 | 0.8 | OK | 82.2 k | 6.2 k | 13× |
| c12_cyl_2mev | 0.0 | 1.3 | OK | 133 k | 3.8 k | 35× |
| o16_sphere_10mev | 0.2 | 0.2 | OK | 2.41 M | 53.9 k | 45× |
| o16_sphere_14mev | 0.9 | 0.1 | OK | 208 k | 14.8 k | 14× |
| ni58_sphere_14mev | 0.4 | 1.0 | OK | 63.4 k | 7.4 k | 8.6× |
| b10_epithermal_1kev | 1.6 | 0.6 | OK | 443 k | 8.9 k | 50× |
| cd112_thermal | 0.6 | 2.8 | WARN | 797 k | 15.9 k | 50× |
| c12_thermal | 1.2 | 1.8 | OK | 275 k | 4.7 k | 58× |

**15 OK, 1 warning, 0 failures**, on a single core. The (n,xn) cases speed
up least, because their secondary-neutron sampling still runs neutron by
neutron through the scalar ENDF readers. The Cd112 warning is on the mean
escape energy (2.8σ) and has been there since the engine was added; the
scalar suite has the same case at 0.3σ. `run_transport_parallel` and `run_keff_parallel`
add one process per core on top of this.

---

## Tallies vs OpenMC

The tallies of the input format (`pyneut/tallies.py`) were compared bin by
bin with OpenMC tallies on matched models. PyNeut went through `pyneut run`
and OpenMC through its Python API, with identical atom densities, and both
results are per source neutron:

| Model | Tally | Bins | max \|z\| | χ²/dof | PyNeut / OpenMC total |
|-------|-------|--:|--:|--:|--:|
| Pb208 sphere, 2 MeV | flux by energy (track-length) | 2 | 0.76 | 0.29 | 15.231 / 15.229 cm |
| | flux by energy (collision) | 2 | 1.00 | 0.81 | 15.252 / 15.174 cm |
| | absorption | 1 | 0.31 | 0.10 | 3.198e-4 / 3.193e-4 |
| Void cavity + two Pb208 shells, 14 MeV | flux by cell × energy | 36 | 2.6 | 1.89 | 38.72 / 38.46 cm |
| | absorption by cell | 2 | 1.4 | 1.02 | 1.203e-3 / 1.189e-3 |
| | leakage spectrum | 12 | 2.7 | 1.44 | 1.6043 / 1.5969 |
| Void cavity + Fe56 shell + Pb208 shell, 14 MeV | flux by cell × energy | 36 | 3.0 | 1.46 | 32.40 / 32.55 cm |
| | absorption by cell | 2 | 0.45 | 0.11 | 6.009e-2 / 6.014e-2 |
| | leakage spectrum | 12 | 2.1 | 1.39 | 1.2239 / 1.2283 |
| C12 block, 2 MeV | 4 × 4 × 4 mesh flux (collision) | 64 | 3.3 | 1.11 | 21.60 / 21.79 cm |
| U235 core + Pb208 reflector, eigenvalue | flux by cell × energy | 6 | 1.7 | 1.53 | 20.06 / 20.03 cm |
| | absorption by cell | 2 | 1.6 | 1.27 | 0.4229 / 0.4202 |

Flux is track length per source neutron (cm), as OpenMC reports it. The
cavity case covers a void cell (seen only by the track-length estimator),
several cells and (n,2n). The same model with iron in place of the inner lead
shell did not agree until every reaction was counted in the total (see
[Reactions above 5 MeV](#reactions-above-5-mev)). PyNeut had too much flux
above 12 MeV in the iron (+7σ) and 2–3 % too much absorption. It now
agrees.

---

## Analytic and data-layer checks

**Analytic benchmarks** (`Validation/analytic_benchmarks.py`): 13 / 13 pass,
within 3σ of the exact result.

- **Uncollided attenuation** through a slab follows Beer–Lambert
  `T = exp(−Σ_t·d)` within 1.6σ over 0.5–3 mean free paths. This checks the
  data reader, free-flight sampling and geometry.
- **Elastic kinematics:** the post-collision energy ratio is uniform on
  `[α, 1]` (100 % of samples), and ⟨E′/E⟩ and the mean lethargy gain ξ match
  the textbook formulas within 0.3σ for A = 12, 56 and 207.

**Cross-section reader** (`validate_xs.py`). Elastic, capture and (n,2n) were
compared against `openmc.data.IncidentNeutron.from_hdf5` for every isotope.
All agree to 0.000 %: each reaction is read and interpolated exactly as
OpenMC reads it. The *total* is checked separately, in
`tests/test_reactions.py`, against OpenMC's for Fe56, Al27, O16, Ni58, B10 and
U235 from thermal to 19 MeV (relative tolerance 1e-8).

**Secondary-sampling diagnostics.** The kernel counts how each (n,xn)
secondary energy is sampled: Law 61 correlated, Law 4, or the Maxwellian
fallback. The suite reports the fallback rate per case, and it is 0 % in every
case.

**Unit tests:** `pytest` runs 270 tests, and all of them pass against the ENDF/B-VIII.0 library (six compare totals with `openmc.data` and also need OpenMC installed). They cover geometry against
analytic and trimesh ground truth, material and number-density arithmetic,
the cross-section reader, scattering kinematics, inelastic physics, batch
statistics, tally uncertainties, criticality, mixtures, the vector engine and
the input format. Every file in `examples/` is loaded, checked and run, and
the input-file path is checked to give the same answer as the hand-built API
at the same seed. The 47 tests that need the ENDF library skip without it.
The input-format tests use a synthetic data library instead.

---

## Physics models

| Process | Treatment |
|---------|-----------|
| Elastic | **Above 10 eV:** stationary-target two-body kinematics with the tabulated ENDF angular distribution. **Below 10 eV:** free-gas kinematics. The target's 294 K speed and direction are sampled together, so neutrons can gain energy and settle into the medium's Maxwellian. |
| Discrete inelastic | MT 51–90, two-body kinematics, isotropic in the CM frame |
| Continuum and (n,xn) | Every other reaction with a neutron product: (n,2n) = MT 16, (n,3n) = MT 17, continuum = MT 91, and MT 5 (n,anything), 22 (n,nα), 28 (n,np) and the rest. Correlated energy and angle (ENDF Law 61 / Kalbach–Mann), with Law 4 and Maxwellian-evaporation fallbacks, unit-base interpolation and a CM→lab boost. The number of neutrons out is sampled from the reaction's tabulated yield, which can be fractional (MT 5) and can be zero, which removes the neutron. Emitted neutrons are sampled independently and banked. |
| Absorption | Every non-redundant reaction with no neutron product: MT 102–117 and the (n,p)/(n,α)… partials MT 600–849 |
| Total cross section | The sum of the non-redundant reactions, as OpenMC builds it. It matches OpenMC's total to 1e-8 at every energy for every nuclide in the library. |
| Unresolved resonances | A cross-section band is sampled per collision from the URR probability tables (OpenMC HDF5 `urr` group). It drives both the flight distance and the reaction selection. |
| Fission | MT 18. In analog `criticality` mode, each fission emits ν̄ (tabulated total yield) neutrons with Watt-χ energies into the next generation's fission bank. In `shielding` mode fission is scored as absorption. |
| Cross sections | ENDF/B-VIII.0 at 294 K, lin–lin interpolation on a per-nuclide union grid, identical to OpenMC |
| Variance reduction | Implicit capture with weight-cutoff Russian roulette (`shielding` mode) |
| k_eff estimators | Collision estimator w·Σᵢ νᵢΣ_f,ᵢ/Σ_t (reported as primary) and generation multiplication, both with batch-means errors |

---

## Limitations

These are stated so the results are not over-trusted. Where possible, the
validation above puts a number on them.

- **Free-gas thermal scattering only (no S(α,β)).** The free-gas model is
  validated, but the binding of atoms in real moderators (water, graphite) is
  not modelled. Thermal results in bound media are therefore approximate, and
  the pin cell is a matched-physics test, not a prediction.
- **One temperature.** Only the 294 K data is loaded, with no on-the-fly
  Doppler broadening.
- **Approximate fission source.** k-eigenvalue uses a Watt χ, not the
  tabulated ENDF fission spectrum. ν̄ is the total (prompt plus delayed), so
  there are no delayed-neutron kinetics and no β_eff.
- **Neutrons only.** Charged-particle reactions remove the neutron, but the
  products and any photons are not transported.
- **Hand-built sources.** You fill per-particle dicts (scalar engine) or arrays
  (vector engine). There is no built-in spectrum or volumetric source sampler,
  apart from `criticality.uniform_sphere_source`.
- **Geometry.** Planes, spheres, and axis-aligned cylinders and boxes, with no
  rotations or general quadrics. Crossings are handled by a 10⁻⁶ cm nudge
  past the surface.
- **No natural-abundance database.** Mixtures list isotopes explicitly. A few
  element aliases exist (`Fe` → Fe56, `Pb` → Pb208, `Be` → Be9, `Al` → Al27,
  `C`/`Graphite` → C0). They map an element to a single file and do **not**
  expand it into its natural isotopes.

---

<p align="center">
  <em>For educational and research-prototyping use only. Not validated for production, regulatory or safety-critical work.</em>
</p>
