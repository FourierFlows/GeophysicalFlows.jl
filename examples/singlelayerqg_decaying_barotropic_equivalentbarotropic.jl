# # SingleLayerQG decaying 2D turbulence with and without finite Rossby radius of deformation
#
#md # This example can be run online via [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/twodnavierstokes_decaying.ipynb).
#md # Also, it can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/twodnavierstokes_decaying.ipynb).
#
# A simulation of decaying two-dimensional turbulence.

using FourierFlows, Printf, Random, Plots
 
using Random: seed!
using FFTW: rfft, irfft

import GeophysicalFlows.SingleLayerQG
import GeophysicalFlows: peakedisotropicspectrum


# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
nothing # hide


# ## Numerical, domain, and simulation parameters
#
# First, we pick some numerical and physical parameters for our model.

n, L  = 128, 2π             # grid resolution and domain length
deformation_radius = 0.35
nothing # hide

## Then we pick the time-stepper parameters
    dt = 1e-2  # timestep
nsteps = 4000  # total number of steps
 nsubs = 20    # number of steps between each plot
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. The
# `stepper` keyword defines the time-stepper to be used.
prob_bqg = SingleLayerQG.Problem(dev; nx=n, Lx=L, dt=dt, stepper="FilteredRK4")
prob_eqbqg = SingleLayerQG.Problem(dev; nx=n, Lx=L, deformation_radius = deformation_radius, dt=dt, stepper="FilteredRK4")
nothing # hide

# Next we define some shortcuts for convenience.
x, y = prob_bqg.grid.x, prob_bqg.grid.y
nothing # hide


# ## Setting initial conditions

# Our initial condition closely tries to reproduce the initial condition used
# in the paper by McWilliams (_JFM_, 1984)
seed!(1234)
k₀, E₀ = 6, 0.5
q₀ = peakedisotropicspectrum(prob_bqg.grid, k₀, E₀, mask=prob_bqg.timestepper.filter)
q₀h = rfft(q₀)
ψ₀h = @. 0*q₀h

SingleLayerQG.streamfunctionfrompv!(ψ₀h, q₀h, prob_bqg.params, prob_bqg.grid)

SingleLayerQG.set_q!(prob_bqg, irfft(-prob_bqg.grid.Krsq .* ψ₀h, prob_bqg.grid.nx))
SingleLayerQG.set_q!(prob_eqbqg, irfft(-(prob_eqbqg.grid.Krsq .+ 1/prob_eqbqg.params.deformation_radius^2) .* ψ₀h, prob_bqg.grid.nx))
nothing # hide

relativevorticity(prob) = irfft(-prob.grid.Krsq .* prob.vars.ψh, prob.grid.nx)  

# Let's plot the initial vorticity field:
heatmap(x, y, relativevorticity(prob_bqg)',
         aspectratio = 1,
              c = :balance,
           clim = (-40, 40),
          xlims = (-L/2, L/2),
          ylims = (-L/2, L/2),
         xticks = -3:3,
         yticks = -3:3,
         xlabel = "x",
         ylabel = "y",
          title = "initial vorticity",
     framestyle = :box)


# ## Visualizing the simulation

# We initialize a plot with the vorticity field.

p_bqg = heatmap(x, y, relativevorticity(prob_bqg)',
         aspectratio = 1,
                   c = :balance,
                clim = (-40, 40),
               xlims = (-L/2, L/2),
               ylims = (-L/2, L/2),
              xticks = -3:3,
              yticks = -3:3,
              xlabel = "x",
              ylabel = "y",
               title = "barotropic\n ∇²ψ, t=" * @sprintf("%.2f", prob_bqg.clock.t),
          framestyle = :box)

p_eqbqg = heatmap(x, y, relativevorticity(prob_eqbqg)',
         aspectratio = 1,
                   c = :balance,
                clim = (-40, 40),
               xlims = (-L/2, L/2),
               ylims = (-L/2, L/2),
              xticks = -3:3,
              yticks = -3:3,
              xlabel = "x",
              ylabel = "y",
               title = "equivalent barotropic; deformation radius: " * @sprintf("%.2f", prob_eqbqg.params.deformation_radius) * "\n ∇²ψ, t=" * @sprintf("%.2f", prob_eqbqg.clock.t),
          framestyle = :box)

l = @layout Plots.grid(1, 2)
p = plot(p_bqg, p_eqbqg, layout = l, size = (900, 400))


# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

startwalltime = time()

cfl(prob) = prob.clock.dt * maximum([maximum(prob.vars.u) / prob.grid.dx, maximum(prob.vars.v) / prob.grid.dy])

anim = @animate for j = 0:Int(nsteps/nsubs)
  if j % (1000 / nsubs) == 0
    log_bqg = @sprintf("barotropic; step: %04d, t: %d, cfl: %.2f, walltime: %.2f min",
        prob_bqg.clock.step, prob_bqg.clock.t, cfl(prob_bqg), (time()-startwalltime)/60)
    log_eqbqg = @sprintf("equivalent barotropic; step: %04d, t: %d, cfl: %.2f, walltime: %.2f min",
        prob_eqbqg.clock.step, prob_eqbqg.clock.t, cfl(prob_eqbqg), (time()-startwalltime)/60)

    println(log_bqg)
    println(log_eqbqg)
  end  

  p[1][1][:z] = relativevorticity(prob_bqg)
  p[1][:title] = "barotropic\n ∇²ψ, t=" * @sprintf("%.2f", prob_bqg.clock.t)
  p[2][1][:z] = relativevorticity(prob_eqbqg)
  p[2][:title] = "equivalent barotropic; deformation radius: " * @sprintf("%.2f", prob_eqbqg.params.deformation_radius) * "\n ∇²ψ, t=" * @sprintf("%.2f", prob_eqbqg.clock.t)
  
  stepforward!(prob_bqg, nsubs)
  SingleLayerQG.updatevars!(prob_bqg)
  stepforward!(prob_eqbqg, nsubs)
  SingleLayerQG.updatevars!(prob_eqbqg)
end

gif(anim, "singlelayerqg_barotropic_equivalentbarotropic.gif", fps=18)
