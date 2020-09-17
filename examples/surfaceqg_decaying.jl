# # Decaying Surface QG turbulence
#
#md # This example can be run online via [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/surfaceqg_decaying.ipynb).
#md # Also, it can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/surfaceqg_decaying.ipynb).
#
# A simulation of decaying surface quasi-geostrophic turbulence.
# We reproduce here the initial value problem for an elliptical 
# vortex as done by Held et al. 1995, _J. Fluid Mech_.

using FourierFlows, Plots, Statistics, Printf, Random

using FFTW: irfft
using Statistics: mean
using Random: seed!

import GeophysicalFlows.SurfaceQG
import GeophysicalFlows.SurfaceQG: kinetic_energy, buoyancy_variance, buoyancy_dissipation


# ## Choosing a device: CPU or GPU

dev = CPU()    # Device (CPU/GPU)
nothing # hide


# ## Numerical parameters and time-stepping parameters

     nx = 256               # 2D resolution = nx^2
stepper = "FilteredETDRK4"  # timestepper
     dt = 0.02              # timestep
     tf = 80                # length of time for simulation
 nsteps = Int(tf / dt)      # total number of time-steps
 nsubs  = round(Int, nsteps/200)         # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

  L = 2π        # domain size
 nν = 4
  ν = 1e-19
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. In this
# example numerical instability due to accumulation of buoyancy variance at high
# wavenumbers is taken care with the `FilteredTimestepper` we picked.
prob = SurfaceQG.Problem(dev; nx=nx, Lx=L, dt=dt, stepper=stepper, ν=ν, nν=nν)
nothing # hide

# Let's define some shortcuts.
sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y
nothing # hide


# ## Setting initial conditions
#
# We initialize the buoyancy equation with an elliptical vortex.
b₀ = @. exp(-(x'^2 + 4 * y^2))
SurfaceQG.set_b!(prob, b₀)
nothing # hide

# Let's plot the initial condition.
heatmap(x, y, prob.vars.b,
     aspectratio = 1,
               c = :deep,
            clim = (0, 1),
           xlims = (-grid.Lx/2, grid.Lx/2),
           ylims = (-grid.Ly/2, grid.Ly/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "buoyancy bˢ",
      framestyle = :box)


# ## Diagnostics

# Create Diagnostic -- `energy` and `enstrophy` are functions imported at the top.
B  = Diagnostic(buoyancy_variance, prob; nsteps=nsteps)
KE = Diagnostic(kinetic_energy, prob; nsteps=nsteps)
Dᵇ = Diagnostic(buoyancy_dissipation, prob; nsteps=nsteps)
diags = [B, KE, Dᵇ] # A list of Diagnostics types passed to "stepforward!" are updated every timestep.
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
get_u(prob) = irfft(im * grid.l .* sqrt.(grid.invKrsq) .* sol, grid.nx)
out = Output(prob, dataname, (:sol, get_sol), (:u, get_u))
nothing # hide


# ## Visualizing the simulation

# We define a function that plots the buoyancy field and the time evolution of
# kinetic energy and buoyancy variance.

function plot_output(prob)
  bˢ = prob.vars.b

  pbˢ = heatmap(x, y, bˢ,
       aspectratio = 1,
                 c = :deep,
              clim = (0, 1),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "buoyancy bˢ",
        framestyle = :box)


  pKE = plot(1,
             label = "kinetic energy ½(u²+v²)",
         linewidth = 2,
            legend = :bottomright,
             alpha = 0.7,
             xlims = (0, tf),
             ylims = (0, 1e-2),
            xlabel = "t")

  pb² = plot(1,
             label = "buoyancy variance (bˢ)²",
         linecolor = :red,
            legend = :bottomright,
         linewidth = 2,
             alpha = 0.7,
             xlims = (0, tf),
             ylims = (0, 2e-2),
            xlabel = "t")

  l = @layout Plots.grid(1, 3, heights=[0.9 ,0.7, 0.7], widths=[0.4, 0.3, 0.3])
  p = plot(pbˢ, pKE, pb², layout=l, size = (1500, 500), dpi=150)

  return p
end
nothing # hide


# ## Time-stepping the `Problem` forward and create animation by updating the plot.

startwalltime = time()

p = plot_output(prob)

anim = @animate for j=0:Int(nsteps/nsubs)

  cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

  if j%(1000/nsubs)==0
    log1 = @sprintf("step: %04d, t: %.1f, cfl: %.3f, walltime: %.2f min",
          clock.step, clock.t, cfl, (time()-startwalltime)/60)
    log2 = @sprintf("buoyancy variance diagnostics - B: %.2e, Diss: %.2e",
              B.data[B.i], Dᵇ.data[Dᵇ.i])
    println(log1)
    println(log2)
  end

  p[1][1][:z] = Array(vars.b)
  p[1][:title] = "buoyancy, t="*@sprintf("%.2f", clock.t)
  push!(p[2][1], KE.t[KE.i], KE.data[KE.i])
  push!(p[3][1], B.t[B.i], B.data[B.i])

  stepforward!(prob, diags, nsubs)
  SurfaceQG.updatevars!(prob)

end

mp4(anim, "sqg_ellipticalvortex.mp4", fps=14)

# Let's see how all flow fields look like at the end of the simulation.

pb1 = heatmap(x, y, vars.u,
     aspectratio = 1,
               c = :RdBu,
            clim = (-maximum(abs.(vars.u)), maximum(abs.(vars.u))),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "uˢ(x, y, t="*@sprintf("%.2f", clock.t)*")",
      framestyle = :box)

pb2 = heatmap(x, y, vars.v,
     aspectratio = 1,
               c = :RdBu,
            clim = (-maximum(abs.(vars.v)), maximum(abs.(vars.v))),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "vˢ(x, y, t="*@sprintf("%.2f", clock.t)*")",
      framestyle = :box)

pb3 = heatmap(x, y, vars.b,
     aspectratio = 1,
               c = :deep,
            clim = (0, 1),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "bˢ(x, y, t="*@sprintf("%.2f", clock.t)*")",
      framestyle = :box)

layout = @layout Plots.grid(1, 3)

plot_final = plot(pb1, pb2, pb3, layout=layout, size = (1500, 500))

# Last we can save the output by calling `saveoutput(out)`.

