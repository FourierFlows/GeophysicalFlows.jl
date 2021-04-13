# # SingleLayerQG decaying 2D turbulence with and without finite Rossby radius of deformation
#
#md # This example can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/singlelayerqg_decaying_barotropic_equivalentbarotropic.ipynb).
#
# We use here the `SingleLayerQG` module to simulate decaying two-dimensional turbulence and
# investigate how does a finite Rossby radius of deformation affects its evolution.

using FourierFlows, Printf, Random, Plots
 
using Random: seed!
using FFTW: rfft, irfft

import GeophysicalFlows.SingleLayerQG
import GeophysicalFlows: peakedisotropicspectrum


# ## Choosing a device: CPU or GPU

dev = GPU()     # Device (CPU/GPU)
nothing # hide


# ## Numerical, domain, and simulation parameters
#
# First, we pick some numerical and physical parameters for our model.

n, L  = 128, 2π             # grid resolution and domain length
deformation_radius = 0.35   # the deformation radius
nothing # hide

## Then we pick the time-stepper parameters
    dt = 1e-2  # timestep
nsteps = 4000  # total number of steps
 nsubs = 20    # number of steps between each plot
nothing # hide


# ## Problem setup
# We initialize two problems by providing a set of keyword arguments to the `Problem` constructor.
# The two problems are otherwise the same, except one has an infinite deformation radius, `prob_bqg`,
# and the other has finite deformation radius, `prob_eqbqg`.
prob_bqg = SingleLayerQG.Problem(dev; nx=n, Lx=L, dt=dt, stepper="FilteredRK4")
prob_eqbqg = SingleLayerQG.Problem(dev; nx=n, Lx=L, deformation_radius = deformation_radius, dt=dt, stepper="FilteredRK4")
nothing # hide


# ## Setting initial conditions

# For initial condition we construct a relative vorticity with energy most energy around total 
# wavenumber ``k_0``.
seed!(1234)
k₀, E₀ = 6, 0.5
∇²ψ₀ = peakedisotropicspectrum(prob_bqg.grid, k₀, E₀, mask=prob_bqg.timestepper.filter)
nothing # hide

# `SingleLayerQG` allows us to set up the initial ``q`` for each problem via `set_q!()` function.
# To initialize both `prob_bqg` and `prob_eqbqg` with the same flow, we first use function 
# `SingleLayerQG.streamfunctionfrompv!` to get the streamfunction that corresponds to the 
# relative vorticity we computed above. This works in the purely barotropic problem, `prob_bqg`
# since in that case the QGPV is simply the relative vorticity.
∇²ψ₀h = rfft(∇²ψ₀)
ψ₀h = @. 0 * ∇²ψ₀h
SingleLayerQG.streamfunctionfrompv!(ψ₀h, ∇²ψ₀h, prob_bqg.params, prob_bqg.grid)
nothing # hide

# and then use the streamfunction to compute the corresponding ``q_0`` for each problem,
q₀_bqg   = irfft(-prob_bqg.grid.Krsq .* ψ₀h, prob_bqg.grid.nx)
q₀_eqbqg = irfft(-(prob_eqbqg.grid.Krsq .+ 1/prob_eqbqg.params.deformation_radius^2) .* ψ₀h, prob_bqg.grid.nx)
nothing # hide

# Now we can initialize our problems with the same flow.
SingleLayerQG.set_q!(prob_bqg, q₀_bqg)
SingleLayerQG.set_q!(prob_eqbqg, q₀_eqbqg)
nothing # hide


# Let's plot the initial vorticity field for each problem. A function that returns relative 
# vorticity from each problem's state variable will prove useful. Note that when plotting, we 
# decorate the variable to be plotted with `Array()` to make sure it is brought back on the 
# CPU when the variable lives on the GPU.
relativevorticity(prob) = irfft(-prob.grid.Krsq .* prob.vars.ψh, prob.grid.nx)

x, y = prob_bqg.grid.x, prob_bqg.grid.y

p_bqg = heatmap(x, y, Array(relativevorticity(prob_bqg)'),
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

p_eqbqg = heatmap(x, y, Array(relativevorticity(prob_eqbqg)'),
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
p = plot(p_bqg, p_eqbqg, layout = l, size = (800, 380))


# ## Time-stepping the `Problem` forward

# Now we time-step both problems forward and animate the relative vorticity in each case.

startwalltime = time()

cfl(prob) = prob.clock.dt * maximum([maximum(prob.vars.u) / prob.grid.dx, maximum(prob.vars.v) / prob.grid.dy])

anim = @animate for j = 0:Int(nsteps/nsubs)
  if j % (1000 / nsubs) == 0
    log_bqg = @sprintf("barotropic; step: %04d, t: %d, cfl: %.2f, walltime: %.2f min",
        prob_bqg.clock.step, prob_bqg.clock.t, cfl(prob_bqg), (time()-startwalltime)/60)
    println(log_bqg)

    log_eqbqg = @sprintf("equivalent barotropic; step: %04d, t: %d, cfl: %.2f, walltime: %.2f min",
        prob_eqbqg.clock.step, prob_eqbqg.clock.t, cfl(prob_eqbqg), (time()-startwalltime)/60)
    println(log_eqbqg)
  end  

  p[1][1][:z] = Array(relativevorticity(prob_bqg))
  p[1][:title] = "barotropic\n ∇²ψ, t=" * @sprintf("%.2f", prob_bqg.clock.t)
  p[2][1][:z] = Array(relativevorticity(prob_eqbqg))
  p[2][:title] = "equivalent barotropic; deformation radius: " * @sprintf("%.2f", prob_eqbqg.params.deformation_radius) * "\n ∇²ψ, t=" * @sprintf("%.2f", prob_eqbqg.clock.t)
  
  stepforward!(prob_bqg, nsubs)
  SingleLayerQG.updatevars!(prob_bqg)
  
  stepforward!(prob_eqbqg, nsubs)
  SingleLayerQG.updatevars!(prob_eqbqg)
end

mp4(anim, "singlelayerqg_barotropic_equivalentbarotropic.mp4", fps=18)
