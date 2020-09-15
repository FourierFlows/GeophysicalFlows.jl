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
import GeophysicalFlows.SurfaceQG: kinetic_energy, buoyancy_variance, buoyancy_dissipation


# ## Choosing a device: CPU or GPU

dev = CPU()    # Device (CPU/GPU)ENV["GRDIR"]=""
Pkg.build("GR")
nothing # hide


# ## Numerical parameters and time-stepping parameters

     nx = 512            # 2D resolution = nx^2
stepper = "FilteredRK4"  # timestepper
     dt = 0.005          # timestep
     tf = 25             # length of time for simulation
 nsteps = tf/dt           # total number of time-steps
 nsubs  = round(Int, nsteps/tf)         # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

  L = 2π        # domain size
 nν = 4
  ν = 1e-19
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. Not providing
# a viscosity coefficient ν leads to the module's default value: ν=0. In this
# example numerical instability due to accumulation of enstrophy in high wavenumbers
# is taken care with the `FilteredTimestepper` we picked.
prob = SurfaceQG.Problem(dev; nx=nx, Lx=L, dt=dt, stepper=stepper,
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

gr  = TwoDGrid(nx, L)

init_b = exp.(-((repeat(gr.x', nx)').^2 + (4*repeat(gr.y', nx)).^2))

seed!(1234) # reset of the random number generator for reproducibility
nothing # hide

# Our initial condition is simply fluid at rest.
SurfaceQG.set_b!(prob, init_b)


# ## Diagnostics

# Create Diagnostic -- `energy` and `enstrophy` are functions imported at the top.
bb = Diagnostic(buoyancy_variance, prob; nsteps=nsteps)
KE = Diagnostic(kinetic_energy, prob; nsteps=nsteps)
Dᵇ = Diagnostic(buoyancy_dissipation, prob; nsteps=nsteps)
diags = [bb, KE, Dᵇ] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hidenothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
# Define base filename so saved data can be distinguished from other runs
base_filename = string("SurfaceQG_decaying_n_", nx, "_visc_", round(ν, sigdigits=1), "_order_", 2*nν)
# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
datapath = "./"
plotpath = "./"

dataname = joinpath(datapath, base_filename)
plotname = joinpath(plotpath, base_filename)
nothing # hide

# Do some basic file management,
if !isdir(plotpath); mkdir(plotpath); end
if !isdir(datapath); mkdir(datapath); end
nothing # hide

# and then create Output.
get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*gr.l.*sqrt.(gr.invKrsq).*sol, gr.nx)
out = Output(prob, dataname, (:sol, get_sol), (:u, get_u))
nothing # hide


# ## Visualizing the simulation

# We define a function that plots the buoyancy field and the time evolution of
# kinetic energy and buoyancy variance.

function plot_output(prob)
    bˢ = prob.vars.b
    Energy =  0.5*(prob.vars.u.^2 + prob.vars.v.^2)

  pbˢ = heatmap(x, y, bˢ,
       aspectratio = 1,
                 c = :RdBu,
              clim = (0, 1),
             xlims = (-gr.Lx/2, gr.Lx/2),
             ylims = (-gr.Ly/2, gr.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "buoyancy bˢ",
        framestyle = :box)


  pKE = plot(1,
             label = "kinetic energy ½(u²+v²)",
         linewidth = 2,
             alpha = 0.7,
             xlims = (0, tf),
             ylims = (0, 5e-3),
            xlabel = "μt")

  pb² = plot(1,
             label = "buoyancy variance (bˢ)²",
         linecolor = :red,
            legend = :bottomright,
         linewidth = 2,
             alpha = 0.7,
             xlims = (0, tf),
             ylims = (0, 5e-2),
            xlabel = "t")

  l = @layout grid(1, 3)
  p = plot(pbˢ, pKE, pb², layout=l, size = (1500, 500), dpi=150)

  return p
end
nothing # hide


# ## Time-stepping the `Problem` forward and create animation by updating plot

startwalltime = time()

p = plot_output(prob)

anim = @animate for j=0:Int(nsteps/nsubs)

  cfl = cl.dt*maximum([maximum(vs.u)/gr.dx, maximum(vs.v)/gr.dy])

  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, walltime: %.2f min",
        cl.step, cl.t, cfl, (time()-startwalltime)/60)

        println(log)

  log = @sprintf("buoyancy variance diagnostics - bb: %.2e, Diss: %.2e",
            bb.data[bb.i], Dᵇ.data[Dᵇ.i])

      println(log)

  p[1][1][:z] = Array(vs.b)
  p[1][:title] = "buoyancy, t="*@sprintf("%.2f", cl.t)
  push!(p[2][1], KE.t[KE.i], KE.data[KE.i])
  push!(p[3][1], bb.t[bb.i], bb.data[bb.i])

  stepforward!(prob, diags, nsubs)
  SurfaceQG.updatevars!(prob)

end

mp4(anim, string(plotname, ".mp4"), fps=2)


# ## Create a plot showing the final buoyancy and velocity fields

l = @layout grid(1, 3)

pb1 = heatmap(x, y, vs.u',
     aspectratio = 1,
               c = :RdBu,
            clim = (-maximum(abs.(vs.u)), maximum(abs.(vs.u))),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "uˢ(x, y, t="*@sprintf("%.2f", cl.t)*")",
      framestyle = :box)

      pb2 = heatmap(x, y, vs.v',
           aspectratio = 1,
                     c = :RdBu,
                  clim = (-maximum(abs.(vs.v)), maximum(abs.(vs.v))),
                 xlims = (-L/2, L/2),
                 ylims = (-L/2, L/2),
                xticks = -3:3,
                yticks = -3:3,
                xlabel = "x",
                ylabel = "y",
                 title = "vˢ(x, y, t="*@sprintf("%.2f", cl.t)*")",
            framestyle = :box)

pb3 = heatmap(x, y, vs.b',
     aspectratio = 1,
               c = :RdBu,
            clim = (0, 1),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "bˢ(x, y, t="*@sprintf("%.2f", cl.t)*")",
      framestyle = :box)


plot_end = plot(pb1, pb2, pb3, layout=l, size = (1500, 500))

png(plotname)

# Last we save the output.
saveoutput(out)
