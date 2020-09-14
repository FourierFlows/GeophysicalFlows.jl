# # Decaying Surface QG turbulence
#
# A simulation of decaying surface quasi-geostrophic turbulence.
# The dynamics include an initial stochastic excitation and small-scale
# hyper-viscous dissipation.

using FourierFlows, Plots, Statistics, Printf, Random

using FourierFlows: parsevalsum
using FFTW: irfft, rfft
using Statistics: mean
using Random: seed!

import GeophysicalFlows.SurfaceQG
import GeophysicalFlows.SurfaceQG: kinetic_energy, buoyancy_variance
import GeophysicalFlows.SurfaceQG: buoyancy_dissipation, buoyancy_work
import GeophysicalFlows.SurfaceQG: buoyancy_drag
import GeophysicalFlows.SurfaceQG: buoyancy_advection, kinetic_energy_advection


# ## Choosing a device: CPU or GPU

dev = CPU()    # Device (CPU/GPU)ENV["GRDIR"]=""
Pkg.build("GR")
nothing # hide


# ## Numerical parameters and time-stepping parameters

     nx = 512            # 2D resolution = nx^2
stepper = "FilteredRK4"  # timestepper
     dt = 0.005          # timestep
 nsteps = 400           # total number of time-steps
 nsubs  = 40         # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

Lx = 2π        # domain size
 ϵ = 1e7           # amplification factor to multiply initial buoyancy field
 nν = 4
 ν = 0.00025*(2e-5)*(2*π/nx)^(2*nν-2)
 ν = 0
nothing # hide

# If we want to add a forcing realization computed every timestep we can Define
# a function. For this decaying case it is not included
#function calcFb!(Fh, sol, t, clock, vars, params, grid)
#  ξ = ArrayType(dev)(exp.(2π*im*rand(eltype(grid), size(sol)))/sqrt(clock.dt))
#  @. Fh = ξ*sqrt.(forcing_spectrum)
#  Fh[abs.(grid.Krsq).==0] .= 0
#  nothing
#end
#nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. Not providing
# a viscosity coefficient ν leads to the module's default value: ν=0. In this
# example numerical instability due to accumulation of enstrophy in high wavenumbers
# is taken care with the `FilteredTimestepper` we picked.
prob = SurfaceQG.Problem(dev; nx=nx, Lx=Lx, dt=dt, stepper=stepper,
                            ν=ν, nν=nν, stochastic=true)
nothing # hide

# Let's define some shortcuts.
sol, cl, vs, pr, gr = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = gr.x, gr.y
nothing # hide


# ## Setting initial conditions
#
# We initialize the buoyancy equation with stochastic excitation that is delta-correlated
# in time, and homogeneously and isotropically correlated in space. The forcing
# has a spectrum with power in a ring in wavenumber space of radius kᵖ and
# width δkᵖ, and it injects energy per unit area and per unit time equalto ϵ.

kᵖ = 14.0    # peak wavenumber for an initial spectrum that is a ring in wavenumber space

gr  = TwoDGrid(nx, Lx)

k = [ gr.kr[i] for i=1:gr.nkr, j=1:gr.nl] # a 2D grid with the zonal wavenumber

#initial_spectrum = @. sqrt(gr.Krsq)^7 / ((sqrt(gr.Krsq) + kᵖ)^12.0)
#init_bh = sqrt.(initial_spectrum).*exp.(2π*im*rand(eltype(initial_spectrum), size(initial_spectrum)))
#init_b = irfft(init_bh, gr.nx)
#init_b *= ϵ   # Renormalize buyancy field to have variance defined by ϵ
@. init_b = 0.01*exp(-(gr.x^2 + (4*gr.y)^2))*(y/Lx^2)


seed!(1234) # reset of the random number generator for reproducibility
nothing # hide

# Our initial condition is simply fluid at rest.
SurfaceQG.set_b!(prob, init_b)


# ## Diagnostics

# Create Diagnostic -- `energy` and `enstrophy` are functions imported at the top.
bb = Diagnostic(buoyancy_variance, prob; nsteps=nsteps)
Dᵇ = Diagnostic(buoyancy_dissipation, prob; nsteps=nsteps) # dissipation by hyperviscosity
Rᵇ = Diagnostic(buoyancy_drag, prob; nsteps=nsteps)
Aᵇ = Diagnostic(buoyancy_advection, prob; nsteps=nsteps)
KE = Diagnostic(kinetic_energy, prob; nsteps=nsteps)
Aᵏ = Diagnostic(kinetic_energy_advection, prob; nsteps=nsteps)

diags = [bb, D, R, KE] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hidenothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
# Define base filename so saved data can be distinguished from other runs
base_filename = string("SurfaceQG_n_", nx, "_visc_", round(ν, sigdigits=1), "_order_", 2*nν, "_ki_", init_wavenumber, "_eps_", ϵ)
# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
datapath = "./Output/Data/"
plotpath = "./Output/Figures/"
plotname = "snapshots"

dataname = joinpath(datapath, base_filename)
plotname = joinpath(plotpath, base_filename)
jld2_filename = joinpath(datapath, string(base_filename, ".jld2"))
nothing # hide

# Do some basic file management,
if isfile(jld2_filename); rm(jld2_filename); end
if !isdir(plotpath); mkdir(plotpath); end
if !isdir(datapath); mkdir(datapath); end
nothing # hide

# and then create Output.
get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*gr.l.*sqrt.(gr.invKrsq).*sol, gr.nx)
out = Output(prob, dataname, (:sol, get_sol), (:u, get_u))
nothing # hide


# ## Visualizing the simulation

# We define a function that plots the buoyancy and velocity fields, their
# corresponding zonal mean structure and timeseries of energy and buoyancy variance.

function plot_output(prob)
    bˢ = prob.vars.b
    b̅ˢ = mean(bˢ, dims=1)'
    ū = mean(prob.vars.u, dims=1)'
    Energy =  0.5*(prob.vars.u.^2 + prob.vars.v.^2)

  pbˢ = heatmap(x, y, bˢ,
       aspectratio = 1,
            legend = false,
                 c = :balance,
              clim = (-8, 8),
             xlims = (-gr.Lx/2, gr.Lx/2),
             ylims = (-gr.Ly/2, gr.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "buoyancy bˢ",
        framestyle = :box)

  pbˢ_zoom = heatmap(x, y, bˢ,
       aspectratio = 1,
            legend = false,
                 c = :balance,
              clim = (-8, 8),
             xlims = (-gr.Lx/8, gr.Lx/8),
             ylims = (-gr.Ly/8, gr.Ly/8),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "buoyancy bˢ",
        framestyle = :box)

  pb̄ˢ = plot(b̅ˢ, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-3, 3),
            xlabel = "zonal mean bˢ",
            ylabel = "y")
  plot!(pb̄ˢ, 0*y, y, linestyle=:dash, linecolor=:black)

  pū = plot(ū, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-0.5, 0.5),
            xlabel = "zonal mean u",
            ylabel = "y")
  plot!(pū, 0*y, y, linestyle=:dash, linecolor=:black)

  pKE = plot(1,
             label = "kinetic energy",
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 4.1),
             ylims = (0, 5e-4),
            xlabel = "μt")

  pb² = plot(1,
             label = "buoyancy variance",
         linecolor = :red,
            legend = :bottomright,
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 4.1),
             ylims = (0, 2.5),
            xlabel = "μt")

  l = @layout grid(2, 3)
  p = plot(pbˢ, pb̄ˢ, pKE, pbˢ_zoom, pū, pb², layout=l, size = (1000, 600), dpi=150)

  return p
end
nothing # hide


# ## Time-stepping the `Problem` forward

startwalltime = time()

p = plot_output(prob)

anim = @animate for j=0:Int(nsteps/nsubs)

  cfl = cl.dt*maximum([maximum(vs.u)/gr.dx, maximum(vs.v)/gr.dy])

  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, walltime: %.2f min",
        cl.step, cl.t, cfl, (time()-startwalltime)/60)

        println(log)

  log = @sprintf("bb diagnostics - Energy: %.2e, Diss: %.2e, Drag: %.2e",
            KE.data[KE.i], D.data[D.i], R.data[R.i])

      println(log)

  #if j%(1000/nsubs)==0; println(log) end

  p[1][1][:z] = Array(vs.b)
  p[1][:title] = "buoyancy, t="*@sprintf("%.2f", cl.t)
  p[4][1][:z] = Array(vs.b.*vs.b)
  p[2][1][:x] = mean(vs.b, dims=1)'
  p[5][1][:x] = mean(vs.u, dims=1)'
  push!(p[3][1], KE.t[KE.i], KE.data[KE.i])
  push!(p[6][1], bb.t[bb.i], bb.data[bb.i])

  #bh = rfft(vs.b)      # Fourier transform of energy density
  #kr, temp_spectrum= FourierFlows.radialspectrum(bh, gr, refinement=1)
  #if  @isdefined(buoyancy_spectrum) # compute radial specturm of `bh`
#      buoyancy_spectrum = [buoyancy_spectrum temp_spectrum]
#  else
#      buoyancy_spectrum = zeros(Complex{Float64}, size(kr,1), 1)
#      buoyancy_spectrum = temp_spectrum
 # end

  stepforward!(prob, diags, nsubs)
  SurfaceQG.updatevars!(prob)

end

mp4(anim, string(plotname, ".mp4"), fps=10)


#mean_buoyancy_spectrum = mean(buoyancy_spectrum, dims=2)

l = @layout grid(2, 2)

Eᵏ = 0.5*(vs.u.^2+vs.v.^2) # Update diagnosed Kinetic Energy
L = Lx

ps1 = heatmap(x, y, vs.b,
          aspectratio = 1,
          legend = false,
               c = :balance,
            clim = (-maximum(abs.(vs.b)), maximum(abs.(vs.b))),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "bˢ(x, y, t="*@sprintf("%.2f", cl.t)*")",
      framestyle = :box)

bh = rfft(vs.b)      # Fourier transform of energy density
kr, buoyancy_spectrum_r = FourierFlows.radialspectrum(bh, gr, refinement=1) # compute radial specturm of `bh`
#nothing # hide

ps2 = plot(kr, abs.(buoyancy_spectrum_r),
    linewidth = 2,
        alpha = 0.7,
       xlabel = "kᵣ", ylabel = "∫ |b̂|² kᵣ dk_θ",
        xlims = (1e0, 1e3),
        ylims = (1e0, 1e5),
       xscale = :log10, yscale = :log10,
        title = "Radial buoyancy variance spectrum",
       legend = false)

ps3 = ps1
ps4 = ps2


plot_spectra = plot(ps1, ps2, ps3, ps4, layout=l, size = (1200, 1200))

png(joinpath(plotpath, string("Spectra_", base_filename)))



i₀ = 1
db²dt_numerical = (bb[(i₀+1):bb.i] - bb[i₀:bb.i-1])/cl.dt #numerical first-order approximation of energy tendency
ii = (i₀):bb.i-1
ii2 = (i₀+1):bb.i

t = bb.t[ii]
db²dt_computed = - D[ii]

residual = db²dt_computed - db²dt_numerical

l = @layout grid(2, 3)

L = Lx

pb1 = heatmap(x, y, vs.u,
     aspectratio = 1,
          legend = false,
               c = :balance,
            clim = (-maximum(abs.(vs.b)), maximum(abs.(vs.b))),
           xlims = (-L/8, L/8),
           ylims = (-L/8, L/8),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "bˢ(x, y, t="*@sprintf("%.2f", cl.t)*")",
      framestyle = :box)

pb2 = plot(t, [db²dt_computed[ii], db²dt_numerical],
           label = ["computed Work minus Dissipation" "numerical db²/dt"],
       linestyle = [:solid :dashdotdot],
       linewidth = 2,
           alpha = 0.8,
           ylims = (-2, 1),
          xlabel = "μt",
          ylabel = "dE/dt")

pb3 = plot(t, residual,
         label = "residual db²/dt = computed - numerical",
     linewidth = 2,
         alpha = 0.7,
         ylims = (-1, 2),
        xlabel = "t")

pb4 = heatmap(x, y, vs.b,
     aspectratio = 1,
          legend = false,
               c = :balance,
            clim = (-maximum(abs.(vs.b)), maximum(abs.(vs.b))),
           xlims = (-L/8, L/8),
           ylims = (-L/8, L/8),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "bˢ(x, y, t="*@sprintf("%.2f", cl.t)*")",
      framestyle = :box)

pb5 = plot(t, [R[ii] D[ii]],
         label = ["Drag" "Dissipation"],
     linestyle = [:solid :dash],
     linewidth = 2,
         alpha = 0.8,
        xlabel = "t",
        ylabel = "Sources and sinks of buoyancy variance")

pb6 = plot(t, [KE[ii] bb[ii]],
           label = ["Kinetic Energy" "Buoyancy Variance"],
       linestyle = [:solid :dash],
       linewidth = 2,
           alpha = 0.8,
          xlabel = "t",
          ylabel = "Evolution of energy and buoyancy variance")

plot_budgets = plot(pb1, pb2, pb3, pb4, pb5, pb6, layout=l, size = (2400, 1200))

png(plotname)

# Last we save the output.
saveoutput(out)

#using LinearAlgebra: mul!, ldiv!

#  ub = vs.u
#  vb = vs.v
#  ub *= vs.b
#  vb *= vs.b

#  ubh = vs.uh
#  vbh = vs.vh
#  N = vs.vh

#  mul!(ubh, gr.rfftplan, ub)
#  mul!(vbh, gr.rfftplan, vb)

#  N = - im * gr.kr .* ubh - im * gr.l .* vbh

#using FourierFlows: abs2

#diag_bb = 2*gr.Krsq.^nν .* abs2.(bh)
#diag_bb[1, 1] = 0

#diag_bb_diss = ν / (gr.Lx * gr.Ly) * parsevalsum(diag_bb, gr)

#  advective_residual = 1 / (2 * gr.Lx * gr.Ly) * parsevalsum(ubh+vbh, gr)
#divergence = 1 / (2 * gr.Lx * gr.Ly) * parsevalsum(- im * gr.l .*ubh - im * gr.kr .*vbh, gr)
