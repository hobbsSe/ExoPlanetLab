# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.
#
# Octofitter.jl fit of DR4 pre-release epoch astrometry, for the cross-code
# comparison with ExoPlanetLab. Astrometry only.
#
# The model follows Octofitter's own tutorial "Gaia DR4 (Pre-release)"
# (v9, GaiaDR4AstromObs): rows with used_by_agis_al, the nine CCDs of each
# transit reduced to one point (median, bootstrap error floored at
# median(ipd_error_al)/sqrt(n)), a per-transit astrometric jitter, the
# single-star terms (offsets, proper motion, parallax) fitted with the orbit,
# a dark companion (flux 0) on a Campbell-element orbit with isotropic
# orientation, sampled with Pigeons parallel tempering. Only the priors on the
# masses, the semi-major axis and the starting point change between sources.
#
# Input:  Results/crosscheck/<name>_epoch_astrometry.csv
#         (python3 crosscheck/export_octofitter_csv.py <source_id> ...)
# Output: Results/crosscheck/octofitter_<name>_chain.csv  (posterior draws
#         plus derived P [d] and photocentre a0 [mas]) and a short summary.
#
# Usage:  julia crosscheck/octofitter_fit.jl gaia4 [n_rounds] [hip]
#         names: gaia4, gaia_bh3, hd114762; n_rounds default 10 (2^n scans)
#         hip: add the Hipparcos IAD (HipparcosIADObs, Octofitter's defaults:
#         free position / proper-motion offsets from the Hipparcos catalogue,
#         parallax shared, jitter; renormalized and corrected records) --
#         the cross-check of ExoPlanetLab's hipparcos.py. Output
#         then octofitter_<name>_hip_chain.csv.

using Octofitter
using Distributions
using CSV, DataFrames
using Statistics
using Random
using Pigeons

const ROOT = dirname(@__DIR__)
const OUT = joinpath(ROOT, "Results", "crosscheck")
const REF_EPOCH_MJD = 57936.375          # J2017.5, the DR4 reference epoch

# Per-source settings: Gaia DR3 source_id, host-mass prior (as ExoPlanetLab's
# RV files), companion-mass and semi-major-axis priors, and a starting point.
const SOURCES = Dict(
    "gaia4" => (sid=1457486023639239296, mA=(0.644, 0.024),
                mb=LogUniform(0.3mjup, 100mjup), amax=10.0,
                init=(plx=13.6, mb=11.8mjup, a=1.17, e=0.3, i=2.1)),
    "gaia_bh3" => (sid=4318465066420528000, mA=(0.76, 0.05),
                   mb=LogUniform(1.0, 100.0), amax=50.0,
                   init=(plx=1.66, mb=33.0, a=16.5, e=0.73, i=1.93)),
    "hd114762" => (sid=3937211745905473024, mA=(1.046, 0.040),
                   mb=LogUniform(1mjup, 1.0), amax=5.0,
                   init=(plx=25.3, mb=0.2, a=0.40, e=0.34, i=0.15)),
)

name = ARGS[1]
n_rounds = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
use_hip = length(ARGS) >= 3 && ARGS[3] == "hip"
S = SOURCES[name]
const HIP_IDS = Dict("hd114762" => 64426)
use_hip && !haskey(HIP_IDS, name) && error("no Hipparcos entry for $name")
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"      # Octofitter's Hipparcos IAD download

# ---- data: CCD rows -> one row per transit (as the tutorial) --------------
load_dr4_ccd(path) = subset(CSV.read(path, DataFrame, comment="#"), :used_by_agis_al)

function reduce_transits(ccd::DataFrame; n_boot::Int=256)
    out = Dict{Int64,DataFrame}()
    for src in groupby(ccd, :source_id)
        sid = src.source_id[1]
        rng = Xoshiro(sid)
        rows = NamedTuple[]
        for tr in groupby(sort(DataFrame(src), :transit_id), :transit_id)
            pos = collect(Float64, tr.centroid_pos_al)
            n = length(pos)
            med = median(pos)
            floor_err = median(tr.ipd_error_al) / sqrt(n)
            if n == 1
                err = tr.ipd_error_al[1]
            else
                meds = [median(rand(rng, pos, n)) for _ in 1:n_boot]
                err = max(std(meds), floor_err)
            end
            push!(rows, (; transit_id=tr.transit_id[1], epoch=mean(tr.epoch),
                           scan_pos_angle=tr.scan_pos_angle[1],
                           parallax_factor_al=tr.parallax_factor_al[1],
                           centroid_pos_al=med, centroid_pos_error_al=err, n_ccd=n))
        end
        out[sid] = DataFrame(rows)
    end
    return out
end

csv = joinpath(OUT, "$(name)_epoch_astrometry.csv")
tdata = reduce_transits(load_dr4_ccd(csv))[S.sid]
println("$name: $(nrow(tdata)) transits over ",
        round((maximum(tdata.epoch) - minimum(tdata.epoch)) / 365.25, digits=2), " yr")

# ---- model ------------------------------------------------------------------
gaia_obs = GaiaDR4AstromObs(
    tdata,
    target = Photocentre,
    ref    = Barycentre,
    name   = "GaiaDR4",
    variables = @variables begin
        astrometric_jitter ~ LogUniform(0.00001, 10)
        ra_offset_mas  ~ Normal(0, 1000)
        dec_offset_mas ~ Normal(0, 1000)
        pmra  ~ Uniform(-1000, 1000)
        pmdec ~ Uniform(-1000, 1000)
        ref_epoch = $REF_EPOCH_MJD
    end
)

orbit_ref_epoch = mean(gaia_obs.table.epoch)
mA_mu, mA_sig = S.mA
mb_prior = S.mb
amax = S.amax

A = Body(name="A", variables=@variables begin
    mass ~ truncated(Normal(mA_mu, mA_sig), lower=0.1)
    flux = 1.0
end)

b = Body(name="b", about=A, variables=@variables begin
    mass ~ mb_prior
    flux = 0.0
    a ~ Uniform(0, amax)
    e ~ Uniform(0, 0.99)
    ω ~ UniformCircular()
    i ~ Sine()
    Ω ~ UniformCircular()
    θ ~ UniformCircular()
    epoch = $orbit_ref_epoch
end)

obs_list = Any[gaia_obs]
if use_hip
    push!(obs_list, HipparcosIADObs(hip_id=HIP_IDS[name], target=A, blends=(),
                                    ref=Barycentre, renormalize=true))
end
sys = System(name=name, bodies=[A, b], observations=obs_list,
             variables=@variables begin
                 plx ~ Uniform(0, 100)
             end)

model = Octofitter.LogDensityModel(sys; verbosity=2)

sol = Octofitter._query_gaia_dr3(gaia_id=S.sid)
initialize!(model, (;
    plx = S.init.plx,
    bodies = (A = (; mass = mA_mu),
              b = (; mass = S.init.mb, a = S.init.a, e = S.init.e, i = S.init.i)),
    observations = (GaiaDR4 = (astrometric_jitter = 0.1, ra_offset_mas = 0.0,
                               dec_offset_mas = 0.0, pmra = sol.pmra, pmdec = sol.pmdec),),
))

t0 = time()
chain, pt = octofit_pigeons(model, n_chains=16, n_rounds=n_rounds,
                            n_chains_variational=0, variational=nothing)
elapsed_min = (time() - t0) / 60

# ---- output -------------------------------------------------------------------
df = DataFrame(chain)
a    = vec(chain["b_a"]);  mb = vec(chain["b_mass"]);  mA = vec(chain["A_mass"])
plx  = vec(chain["plx"])
Mtot = mA .+ mb
df.P_day  = sqrt.(a .^ 3 ./ Mtot) .* 365.25
df.a0_mas = a .* mb ./ Mtot .* plx          # photocentre = primary (flux_b = 0)
out = joinpath(OUT, "octofitter_$(name)$(use_hip ? "_hip" : "")_chain.csv")
CSV.write(out, df)

q(v) = round.(quantile(v, (0.16, 0.5, 0.84)), sigdigits=6)
println("\n$name, Octofitter $(pkgversion(Octofitter)), n_rounds=$n_rounds, ",
        "$(round(elapsed_min, digits=1)) min sampling")
println("P [d]      : ", q(df.P_day))
println("e          : ", q(vec(chain["b_e"])))
println("i [deg]    : ", q(rad2deg.(vec(chain["b_i"]))))
println("a0 [mas]   : ", q(df.a0_mas))
println("plx [mas]  : ", q(plx))
println("jitter/tr. : ", q(vec(chain["GaiaDR4_astrometric_jitter"])))
println("pmra, pmdec (Gaia): ", q(vec(chain["GaiaDR4_pmra"])), "  ", q(vec(chain["GaiaDR4_pmdec"])))
if use_hip
    for k in ("iad_Δra", "iad_Δdec", "iad_pmra", "iad_pmdec", "hip_iad_jitter")
        key = "Hipparcos IAD_" * k
        haskey(chain, key) || (key = "Hipparcos_IAD_" * k)
        haskey(chain, key) && println(rpad("Hip " * k, 18), ": ", q(vec(chain[key])))
    end
end
println("wrote $out")
