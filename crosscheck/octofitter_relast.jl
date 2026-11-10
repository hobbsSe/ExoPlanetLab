# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David Hobbs  <david.hobbs@fysik.lu.se>.
#
# Octofitter.jl fit of simulated Gaia DR4 epoch astrometry with relative
# astrometry of the companion, for the cross-check of ExoPlanetLab's
# EXOPLANETLAB_RELAST_MODE.
#
# The model is that of octofitter_fit.jl (GaiaDR4AstromObs of the photocentre,
# single-star terms fitted with the orbit, a per-point astrometric jitter, a
# dark companion on a Campbell-element orbit with isotropic orientation,
# Pigeons parallel tempering) with three changes: the CCD rows are fitted as
# they are, not reduced to transits, so that the points and errors are those
# ExoPlanetLab fits; RelAstromObs of the companion relative to its host is
# added; and the host mass has a wide uniform prior, so that the mass fraction
# M_b / M_tot = 1 / s is set by the data, as ExoPlanetLab's free scale s is.
#
# Input:  Results/crosscheck/relast_<case>_epoch_astrometry.csv,
#         relast_<case>_rel_astrometry.csv, relast_<case>_truth.json
#         (python3 crosscheck/export_relast_injection.py)
# Output: Results/crosscheck/octofitter_relast_<case>_chain.csv (posterior
#         draws plus derived P [d], a0 and a_rel [mas], s) and a short summary.
#
# Usage:  julia crosscheck/octofitter_relast.jl short [n_rounds]

using Octofitter
using Distributions
using CSV, DataFrames
using Statistics
using Pigeons

const ROOT = dirname(@__DIR__)
const OUT = joinpath(ROOT, "Results", "crosscheck")
const REF_EPOCH_MJD = 57936.375          # J2017.5, the DR4 reference epoch

case = ARGS[1]
n_rounds = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
base = joinpath(OUT, "relast_$(case)")

# the injected values: starting point only
truth_txt = read(base * "_truth.json", String)
tr(k) = parse(Float64, match(Regex("\"$k\": ([-+0-9.eE]+)"), truth_txt).captures[1])

# ---- data -----------------------------------------------------------------
gdata = CSV.read(base * "_epoch_astrometry.csv", DataFrame, comment="#")
println("$case: $(nrow(gdata)) along-scan points over ",
        round((maximum(gdata.epoch) - minimum(gdata.epoch)) / 365.25, digits=2), " yr")

# ExoPlanetLab's file: time(d) from TIME_REF_JD, dra*, ddec, errors [mas], rho
rel_lines = readlines(base * "_rel_astrometry.csv")
tref = parse(Float64, match(r"TIME_REF_JD\s*=\s*([0-9.]+)",
                            first(filter(l -> occursin("TIME_REF_JD", l), rel_lines))).captures[1])
rel = [parse.(Float64, strip.(split(l, ","))) for l in rel_lines
       if !startswith(strip(l), "#") && !isempty(strip(l))]
rdata = DataFrame(epoch = [r[1] + tref - 2400000.5 for r in rel],
                  ra = [r[2] for r in rel], dec = [r[3] for r in rel],
                  σ_ra = [r[4] for r in rel], σ_dec = [r[5] for r in rel],
                  cor = [r[6] for r in rel])
println("$(nrow(rdata)) relative positions, MJD ", round(minimum(rdata.epoch), digits=1),
        " to ", round(maximum(rdata.epoch), digits=1))

# ---- model ------------------------------------------------------------------
gaia_obs = GaiaDR4AstromObs(
    gdata,
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

A = Body(name="A", variables=@variables begin
    mass ~ Uniform(0.1, 5.0)
    flux = 1.0
end)

b = Body(name="b", about=A, variables=@variables begin
    mass ~ LogUniform(0.001, 1.0)
    flux = 0.0
    a ~ Uniform(0, 20)
    e ~ Uniform(0, 0.99)
    ω ~ UniformCircular()
    i ~ Sine()
    Ω ~ UniformCircular()
    θ ~ UniformCircular()
    epoch = $orbit_ref_epoch
end)

rel_obs = RelAstromObs(rdata; target=b, ref=A, name="RelAst")

sys = System(name="relast_$(case)", bodies=[A, b], observations=Any[gaia_obs, rel_obs],
             variables=@variables begin
                 plx ~ Uniform(0, 100)
             end)

model = Octofitter.LogDensityModel(sys; verbosity=2)

initialize!(model, (;
    plx = tr("plx"),
    bodies = (A = (; mass = tr("M1")),
              b = (; mass = tr("M2"), a = tr("a_rel_mas") / tr("plx"), e = tr("e"),
                     i = deg2rad(tr("inc_deg")))),
    observations = (GaiaDR4 = (astrometric_jitter = 0.01, ra_offset_mas = 0.0,
                               dec_offset_mas = 0.0, pmra = tr("pmra"), pmdec = tr("pmdec")),),
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
df.P_day     = sqrt.(a .^ 3 ./ Mtot) .* 365.25
df.a_rel_mas = a .* plx
df.a0_mas    = a .* mb ./ Mtot .* plx       # photocentre = primary (flux_b = 0)
df.s         = Mtot ./ mb
out = joinpath(OUT, "octofitter_relast_$(case)_chain.csv")
CSV.write(out, df)

q(v) = round.(quantile(v, (0.16, 0.5, 0.84)), sigdigits=6)
println("\n$case, Octofitter $(pkgversion(Octofitter)), n_rounds=$n_rounds, ",
        "$(round(elapsed_min, digits=1)) min sampling")
println("P [d]      : ", q(df.P_day), "   true ", tr("P"))
println("e          : ", q(vec(chain["b_e"])), "   true ", tr("e"))
println("i [deg]    : ", q(rad2deg.(vec(chain["b_i"]))), "   true ", tr("inc_deg"))
println("a0 [mas]   : ", q(df.a0_mas), "   true ", tr("a0_mas"))
println("a_rel [mas]: ", q(df.a_rel_mas), "   true ", tr("a_rel_mas"))
println("s          : ", q(df.s), "   true ", tr("s"))
println("plx [mas]  : ", q(plx), "   true ", tr("plx"))
println("M_A, M_b   : ", q(mA), "  ", q(mb))
println("jitter     : ", q(vec(chain["GaiaDR4_astrometric_jitter"])))
println("wrote $out")
