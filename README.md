# ExoPlanetLab

Bayesian orbit fitting of Gaia epoch astrometry, alone or together with radial
velocities.

ExoPlanetLab fits Keplerian orbits (Thiele-Innes constants) to the along-scan
measurements of Gaia DR4, comparing the orbit with the epoch residuals through
the projection of the single-star fit that produced them. One likelihood can
be sampled with five samplers. It was written for the Gaia DR4 epoch
astrometry and validated on the DR4 pre-release sources.

**Version 1.0.0.** A paper describing the code is in preparation.

## What it does

- Orbit fits to epoch astrometry, with or without radial velocities from
  several instruments; the single-star parameters are recovered for every
  posterior draw.
- Gaussian or Student-t noise per measurement, with the single-star terms
  refitted robustly.
- A Keplerian periodogram with bootstrap false-alarm probabilities, robust to
  outliers, for the signal count and the starting values.
- Five samplers: affine-invariant MCMC with parallel tempering (`emcee`),
  nested sampling (`dynesty`, `nessai`), and reversible-jump MCMC (`rjmcmc`,
  `rjflow`).
- The number of companions from fixed-N evidences; acceleration (7- and
  9-parameter) and scan-angle harmonic models, alone or inside an orbit fit.
- Upper limits on undetected companions against period
  (`python3 Tools/upper_limits.py <source_id>`).
- Injection of synthetic companions into the observing geometry of a real
  source, for tests of detection and calibration.

## Installation

Python 3.11 or later. ExoPlanetLab is run from a clone of the repository; its
modules are not installed as a package.

    git clone https://github.com/hobbsSe/ExoPlanetLab.git
    cd ExoPlanetLab
    pip install -e ".[gaia]"            # the dependencies; add ,nested for dynesty / nessai

The dependencies are listed in `pyproject.toml`: numpy, scipy, astropy,
matplotlib, corner, pandas, emcee and eryn; `gaiasupdate` and astroquery to
read Gaia data; dynesty, nessai and torch for the nested and flow samplers.

## Data

The Gaia DR4 pre-release epoch astrometry is not distributed with the code.
Download it from <https://www.cosmos.esa.int/web/gaia/dr4-prerelease> and
either place the file at

    data-release-4-tutorials/extracted_files/GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml

or give its location with the setting `dr4_prerelease_file`. Radial
velocities are read from `RVdata/<source_id>/rv_data.csv` (or `rv_file`); the
files for the three pre-release sources with published orbits are included,
with their references in the file headers.

## Running

A run is described by settings, given in a configuration file, on the command
line, or both:

    python3 epl.py examples/gaia4.toml
    python3 epl.py examples/gaia4.toml --set solver=dynesty --tag Gaia-4-dynesty
    python3 epl.py --source-id 4318465066420528000 --rv-mode none
    python3 epl.py examples/injection.toml --dry-run      # show the settings, do not run
    python3 epl.py --list-settings

```toml
# examples/gaia4.toml
source_mode = "gaia_dr4_prerelease"
source_id   = 1457486023639239296
rv_mode     = "external"
solver      = "emcee"
astro_noise = "student"
run_tag     = "Gaia-4"
```

Setting names are checked, so a misspelt one is an error. Each setting is also
an environment variable with the prefix `EXOPLANETLAB_` (for example
`EXOPLANETLAB_SOLVER=dynesty python3 ExoPlanetLab.py`); the command line takes
precedence over the file, and the file over the environment.

Several runs side by side, one core each:

    python3 Tools/run_batch.py jobs.txt --max-parallel 6

where each line of `jobs.txt` is a name followed by `NAME=VALUE` settings. A
tempered evidence run keeps up to 2 GB of chains in memory (`pt_store_max_gb`).

Each run writes a folder in `Results/` with the posterior samples
(`posterior_samples.npz`), a log, the plots, and `run_info.json` (seed,
settings, package versions) from which it can be repeated.

Model comparison: fit with `evidence = true` and `n_fit` = 1, 2, ... under tags
ending in `-N1`, `-N2`, ..., then

    python3 Tools/evidence_ladder.py <tag prefix>

## Reference runs

The reference batch fits a simulated two-planet system with every sampler,
and the three DR4 pre-release sources with published orbits, with and without
their radial velocities:

    python3 Tools/run_batch.py Tools/jobs_reference.txt

**Simulated system.** Two planets in a 2:1 period ratio (P = 1450 and
2900 d, signal-to-noise ratios 53 and 19), 140 along-scan epochs over
9.9 years, the outer planet observed for 1.2 orbits.

- Without radial velocities all five samplers agree to within 0.1σ in every
  parameter, and the residuals are 1.04–1.05 times the noise. The outer
  planet's periastron time and orbit lie along a nearly degenerate ridge, so
  its median sits 2–3σ from the truth in every sampler.
- 50 simulated radial velocities remove the ridge: every offset falls to
  1.1σ or less, and the width of the periastron time halves.

Run times on one core of a laptop (Apple M1 Max), all eight runs side by side:

| Sampler | Astrometry | + RV |
|---|---|---|
| `emcee` | 6 min | 5 min |
| `nessai` | 8 min | 8 min |
| `dynesty` | 12 min | 32 min |
| `rjmcmc` | 31 min | – |
| `rjflow` | 25 min | – |

**DR4 pre-release sources.** Fitted with their radial velocities:

- Gaia BH3: every parameter within 1.4σ of the published solutions.
- Gaia-4: the period and orientation differ by 2–3σ from the published
  solution, which rests on DR3 data.
- HD 114762: a nearly face-on orbit. The astrometry bounds the inclination
  from above, i < 11° (95%), so the companion mass is above 71 M_Jup: most
  likely an M dwarf, not a planet. With K free (the default) the inclination
  comes from the astrometric orbit alone, and the mass from K and i,
  133 (+124/−46) M_Jup, follows the orientation prior. It is lower and much
  wider than the published 225 ± 14 M_Jup. With `rv_coupled = 1`, K is tied
  to the astrometric orbit through the parallax, as for a dark companion,
  and the radial velocities fix the inclination as well: i ≈ 3.5° and
  M₂ ≈ 240 M_Jup, close to the published values.

## Injection tests

180 single companions were injected into the real observing geometry of five
quiet DR4 pre-release sources and fitted as real data, with signal-to-noise
ratios S from 5 to 100 (`Tools/make_injection_jobs.py` writes the jobs,
`Tools/jobs_injection.txt` holds them, `Tools/injection_summary.py` collects
the results):

- All 180 runs converged.
- Every signal with S ≥ 10 was detected (ln B > 5), and 83% of those with
  S = 5–10.
- The posteriors are unbiased. The credible intervals have the right
  coverage for every parameter except the period: 90% of the true periods
  lie inside their 95% intervals.
- None of 30 data sets without a companion gave a false detection.
- In Student-t noise like that of the DR4 data, the outlier-robust
  periodogram screen finds many more weak signals than least squares.

On the same data the samplers also agree with Octofitter and kepmodel, and
for Gaia-4 with kima.

## Tests

    python3 -m pytest tests

## Documentation

The user guide, `ExoPlanetLabDoc/ExoPlanetLab.pdf`, describes the model,
the priors, the samplers and every setting, with example results, exercises,
troubleshooting and the limitations of the code.

## Citing

See `CITATION.cff`. Please also cite the samplers you use (emcee, eryn,
dynesty, nessai) and, for Gaia data, follow the Gaia acknowledgement
instructions.

## Licence

GPL-3.0-or-later; see `LICENSE.txt`.
