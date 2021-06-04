# # Decaying Surface QG turbulence
#
#md # This example can be run online via [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/surfaceqg_decaying.ipynb).
#md # Also, it can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/surfaceqg_decaying.ipynb).
#
# A simulation of decaying surface quasi-geostrophic turbulence.
# We reproduce here the initial value problem for an elliptical 
# vortex as done by Held et al. 1995, _J. Fluid Mech_.
# 
# An example of decaying barotropic quasi-geostrophic turbulence over topography.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add GeophysicalFlows, Plots, Printf, Random, Statistics"
# ```

# ## Let's begin
# Let's load `GeophysicalFlows.jl` and some other needed packages.
#
using GeophysicalFlows, Plots, Printf, Random

using Statistics: mean
using Random: seed!


# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
nothing # hide


# ## Numerical parameters and time-stepping parameters

      n = 256                       # 2D resolution = n²
stepper = "FilteredETDRK4"          # timestepper
     dt = 0.03                      # timestep
     tf = 60                        # length of time for simulation
 nsteps = Int(tf / dt)              # total number of time-steps
 nsubs  = round(Int, nsteps/100)    # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

 L = 2π        # domain size
 ν = 1e-19     # hyper-viscosity coefficient
nν = 4         # hyper-viscosity order
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. In this
# example numerical instability due to accumulation of buoyancy variance at high
# wavenumbers is taken care with the `FilteredTimestepper` we picked.
prob = SurfaceQG.Problem(dev; nx=n, Lx=L, dt=dt, stepper=stepper, ν=ν, nν=nν)
nothing # hide

# Let's define some shortcuts.
sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y
#md nothing # hide


# ## Setting initial conditions
#
# We initialize the buoyancy equation with an elliptical vortex.
X, Y = gridpoints(grid)
b₀ = @. exp(-(X^2 + 4*Y^2))

SurfaceQG.set_b!(prob, b₀)
nothing # hide

# Let's plot the initial condition. Note that when plotting, we decorate the variable to be 
# plotted with `Array()` to make sure it is brought back on the CPU when `vars` live on the GPU.
heatmap(x, y, Array(vars.b'),
     aspectratio = 1,
               c = :deep,
            clim = (0, 1),
           xlims = (-grid.Lx/2, grid.Lx/2),
           ylims = (-grid.Ly/2, grid.Ly/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "buoyancy bₛ",
      framestyle = :box)


# ## Diagnostics

# Create Diagnostics; `buoyancy_variance`, `kinetic_energy` and `buoyancy_dissipation` 
# functions were imported at the top.
B  = Diagnostic(SurfaceQG.buoyancy_variance, prob; nsteps=nsteps)
KE = Diagnostic(SurfaceQG.kinetic_energy, prob; nsteps=nsteps)
Dᵇ = Diagnostic(SurfaceQG.buoyancy_dissipation, prob; nsteps=nsteps)
diags = [B, KE, Dᵇ] # A list of Diagnostics types passed to `stepforward!`. Diagnostics are updated every timestep.
nothing # hidenothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
# Define base filename so saved data can be distinguished from other runs
base_filename = string("SurfaceQG_decaying_n_", n)
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

# We define a function that plots the buoyancy field and the time evolution of kinetic energy 
# and buoyancy variance.

function plot_output(prob)
  b = prob.vars.b

  pb = heatmap(x, y, Array(b'),
       aspectratio = 1,
                 c = :deep,
              clim = (0, 1),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "buoyancy bₛ",
        framestyle = :box)

  pKE = plot(1,
             label = "kinetic energy ∫½(uₛ²+vₛ²)dxdy/L²",
         linewidth = 2,
            legend = :bottomright,
             alpha = 0.7,
             xlims = (0, tf),
             ylims = (0, 1e-2),
            xlabel = "t")

  pb² = plot(1,
             label = "buoyancy variance ∫bₛ²dxdy/L²",
         linecolor = :red,
            legend = :bottomright,
         linewidth = 2,
             alpha = 0.7,
             xlims = (0, tf),
             ylims = (0, 2e-2),
            xlabel = "t")

  layout = @layout [a{0.5w} Plots.grid(2, 1)]
  p = plot(pb, pKE, pb², layout=layout, size = (900, 500))

  return p
end
nothing # hide


# ## Time-stepping the `Problem` forward and create animation by updating the plot.

startwalltime = time()

p = plot_output(prob)

anim = @animate for j = 0:round(Int, nsteps/nsubs)
  if j % (500 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log1 = @sprintf("step: %04d, t: %.1f, cfl: %.3f, walltime: %.2f min",
          clock.step, clock.t, cfl, (time()-startwalltime)/60)

    log2 = @sprintf("buoyancy variance: %.2e, buoyancy variance dissipation: %.2e",
              B.data[B.i], Dᵇ.data[Dᵇ.i])

    println(log1)

    println(log2)
  end

  p[1][1][:z] = Array(vars.b)
  p[1][:title] = "buoyancy, t=" * @sprintf("%.2f", clock.t)
  push!(p[2][1], KE.t[KE.i], KE.data[KE.i])
  push!(p[3][1], B.t[B.i], B.data[B.i])

  stepforward!(prob, diags, nsubs)
  SurfaceQG.updatevars!(prob)
end

mp4(anim, "sqg_ellipticalvortex.mp4", fps=14)

# Let's see how all flow fields look like at the end of the simulation.

pu = heatmap(x, y, Array(vars.u'),
     aspectratio = 1,
               c = :balance,
            clim = (-maximum(abs.(vars.u)), maximum(abs.(vars.u))),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "uₛ(x, y, t=" * @sprintf("%.2f", clock.t) * ")",
      framestyle = :box)

pv = heatmap(x, y, Array(vars.v'),
     aspectratio = 1,
               c = :balance,
            clim = (-maximum(abs.(vars.v)), maximum(abs.(vars.v))),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "vₛ(x, y, t=" * @sprintf("%.2f", clock.t) * ")",
      framestyle = :box)

pb = heatmap(x, y, Array(vars.b'),
     aspectratio = 1,
               c = :deep,
            clim = (0, 1),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "bₛ(x, y, t=" * @sprintf("%.2f", clock.t) * ")",
      framestyle = :box)

layout = @layout [a{0.5h}; b{0.5w} c{0.5w}]

plot_final = plot(pb, pu, pv, layout=layout, size = (800, 800))

# ## Save

# Last we can save the output by calling
# ```julia
# saveoutput(out)`
# ```
