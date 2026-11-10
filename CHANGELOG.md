# Changelog

Notable changes to ExoPlanetLab.

## 1.0.0 (2026-11-10)

First public release.

### Added
- `epl.py`: run from a configuration file (TOML) and/or the command line;
  setting names are checked. Example configurations in `examples/`.
- Settings `rv_file`, `m_star_msun`, `parallax_mas`, `m_star_sigma`,
  `parallax_sigma` (previously edited in `ExoPlanetLab.py`).
- `pyproject.toml` (dependencies), `README.md`, `CITATION.cff`. The README
  describes the reference runs (`Tools/jobs_reference.txt`) and their
  results: the simulated two-planet system with every sampler, with run
  times, and the three DR4 pre-release sources with published orbits;
  and the injection tests (detection, calibration, false detections).
- Upper limits on undetected companions against period, without sampling
  (`limits.py`, `Tools/upper_limits.py`); `Tools/scaling_benchmark.py`.
- Ranked starting candidates: for one orbit from the screen's detections and
  the distinct peaks of its first periodogram; for two or more orbits from a
  beam search over periodogram peaks (`start_search`). `seed_rank` starts from
  the k-th candidate; `stop_after_start` prints them and stops.
- Acceleration (7- and 9-parameter) and scan-angle harmonic models, alone and
  inside an orbit fit (`astro_accel`, `astro_harm`); linear RV trend
  (`rv_trend`).
- Evidence ladder for the number of companions (`evidence`, `n_fit`), with a
  bridge-sampling estimate and a stability test; evidence on the normalised
  prior from the nested samplers.
- Robust (Student-t score) signal screen (`screen_robust`).
- Relative (imaging) astrometry of a resolved companion (`relast_mode`,
  `relast_file`), with a fitted imaging jitter (`relast_jitter`, on by default
  for data from a file) and the companion's mass for a given flux ratio
  (`relast_flux_ratio`); `make_injection_jobs.py --relast`.
- RV amplitude tied to the astrometric orbit for a dark companion
  (`rv_coupled`); `Tools/coupled_k.py` gives the same from an existing fit by
  importance sampling. With K free the inclination comes from the
  astrometric orbit alone. For a nearly face-on orbit that bounds it only
  from above, so the mass from K and i follows the orientation prior. The
  coupling lets the radial velocities fix the inclination as well. For
  HD 114762 the mass is 133 (+124/−46) M_Jup with K free, against a published
  225 ± 14; with `rv_coupled` the fit gives i ≈ 3.5° and about 240 M_Jup.
- Diagnostics of the DR4 residuals (`Tools/noise_diagnostics.py`); experimental
  error factors per gate (`astro_gate_f`).
- Checks used in the paper: coverage of the upper limits on injected companions
  (`Tools/limits_coverage.py`); chi2/dof of the orbit fits and within transits
  (`Tools/excess_noise_chi2.py`); chi2 of published orbits held fixed, with and
  without the projection (`Tools/published_orbit_chi2.py`); the effect of the
  pseudo-colour prior on the projection (`Tools/projection_prior_check.py`).
- `Tools/coupled_k.py --name` reads runs with other names (e.g. `EVRV-{tag}-N1`).
- Reversible-jump solvers: the autocorrelation time of the cold chain at the
  most probable number of planets is measured and printed with the effective
  number of samples and the sampling error of a reported median. Three
  estimates are given: per walker, from the scatter of the walker means, and
  for all walkers together. The walkers are not independent, and only the
  third gives the error of a reported value: on the two-planet reference case
  it is about 400 steps against 115 per walker (measured on a chain of 40,000
  steps), which means about 1000 effective samples of the standard 400,000
  and an error of 0.04 sigma on a median; the chain of 40,000 steps gives
  0.02 sigma. `T0` is folded to one orbital cycle before the measurement.
  `EXOPLANETLAB_RJ_SAVE_CHAIN=<file.npy>` saves the chain used.
- Run seed and `run_info.json` for every run; unit tests and a CI workflow.
- `Tools/rerun.py`: repeat a run from the `run_info.json` in its `Results/`
  folder, with the recorded settings and seed, in a fresh process without the
  settings of the calling shell. `--set KEY=VALUE` changes a setting
  (`KEY=` removes it), `--tag` names the new folder, `--dry-run` prints the
  command. A run that used one core gets one BLAS thread again. It notes when
  the git commit or a package version differs from the recorded one, or when
  the original run did not finish.
- `sim_n`: simulate only the first one or two planets of the simulated system.
- `priors.py`: the priors of the orbital elements as a module, with tests.

### Changed
- `run_info.json` is written when a run starts and again when it ends, with a
  `finished` field, so a run that stopped can also be repeated.
- Settings that cannot work together now stop the run before its `Results/`
  folder is created: a simulated source with external RVs but no `rv_file`, a
  missing RV file, an unknown or unsupported `orient_prior`, nested-sampler
  evidence with relative astrometry, and an invalid `orbit_separation`. The
  message for `orient_prior = "isotropic"` names the solvers that support it
  (`emcee` and `nessai`, and `dynesty` with `evidence`).
- Source modes: `gaia_dr4` reads Gaia DR4 epoch astrometry for a source from a
  DR4 epoch-astrometry file (`dr4_file`, required) in the format of the
  pre-release export; `gaia_dr4_prerelease` is the same with the pre-release
  export. The two modes share everything else (projection, Student-t default,
  injections, Hipparcos, orbit-corrected single-star terms).
- Removed the `gaia` mode. Gaia DR3 published no epoch astrometry, so it could
  only simulate data from a DR3 orbital solution; its astrometric error term
  (sigma_astro) went with it. An unknown `source_mode` now stops at once.
- `campbell_elements` and `orbit.omega_from_ti` return a consistent
  (omega, Omega) pair with Omega in [0, 180) deg.
- Default store of the tempered sampler capped at 2 GB per run
  (`pt_store_max_gb`, was 8).
- Reversible-jump solvers, default weights of the in-model moves: the
  differential-evolution move, accepted 0.04-0.05 of the time, has weight
  0.10 (was 0.45 in `rjmcmc`, 0.30 in `rjflow`); the KDE and flow moves have
  0.65. `rjmcmc` takes 30 min on the reference case, from 26.

### Fixed
- Reversible-jump solvers (`rjmcmc`, `rjflow`): the cold-chain in-model
  acceptance on the two-planet reference case was 0.02. It is now 0.36
  (`rjmcmc`) and 0.39 (`rjflow`). The causes, in order of effect:
  - The KDE and flow moves drew one planet from its own distribution while
    the other planets stayed fixed. The planets are correlated in the joint
    posterior (coefficients up to 0.96), so the region allowed for one planet
    given the other is 3% of the volume of its own distribution, and 2% of
    the proposals were accepted. Each planet is now drawn given the others
    (`_SourceConditioner`; `EXOPLANETLAB_RJ_CONDITIONAL=0` restores the old
    moves). KDE move 0.02 to 0.46, flow move 0.01 to 0.53.
  - The flow was trained for 800 epochs on 2000 samples and learned the
    individual samples. Training now stops when the loss on a held-out fifth
    stops improving (`EXOPLANETLAB_RJ_FLOW_PATIENCE`, 40 epochs), and the
    linear correlations are removed before training. Training time per flow
    went from 70 s to about 2 s.
  - The Gaussian move capped the condition number of its covariance in raw
    units, used one covariance for all planets and one step factor for all
    temperatures. It now has a covariance per planet and a step factor per
    temperature, and draws through the correlation matrix (0.00 to 0.21).
  - The KDE bandwidth factor was 0.25; it is now Scott's rule
    (`EXOPLANETLAB_RJ_KDE_BW`).
  - The flow's standardisation added an absolute 1e-8 to every spread.
- Reversible-jump solvers: the parameter tables, the corner plots, the saved
  samples and the best fit were taken from the emcee warm-up at the known
  number of planets, because the chain of the reversible-jump stage used to
  stand still. They are now taken from that chain when its cold-chain
  in-model acceptance is at least 0.05, and the output says which was used
  (`EXOPLANETLAB_RJ_REPORT` = `chain` or `warmup` forces one). On the
  reference case the two agree to a small fraction of the intervals.
- Reversible-jump solvers: warm-start samples far below the best
  ln-probability (walkers in a secondary maximum) are not passed on to the
  proposals.
- Reversible-jump solvers: the best-fit period was converted from its
  logarithm twice, because the best-fit vector was a view of the sample array
  that is converted in place. This affected the plots that use the best fit
  of the reversible-jump stage, extra planets beyond the warm start, and runs
  without the warm start; the reported tables were not affected.
- `solver_rjmcmc.py`, `solver_rjflow.py`: print the cold-chain acceptance of
  each in-model move and the step factor of each temperature.
