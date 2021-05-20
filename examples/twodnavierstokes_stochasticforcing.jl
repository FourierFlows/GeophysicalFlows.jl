# # 2D forced-dissipative turbulence
#
#md # This example can be run online via [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/twodnavierstokes_stochasticforcing.ipynb).
#md # Also, it can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/twodnavierstokes_stochasticforcing.ipynb).
#
# A simulation of forced-dissipative two-dimensional turbulence. We solve the
# two-dimensional vorticity equation with stochastic excitation and dissipation in
# the form of linear drag and hyperviscosity. 

using FourierFlows, Printf, Plots

using FourierFlows: parsevalsum
using Random: seed!
using FFTW: irfft

import GeophysicalFlows.TwoDNavierStokes
import GeophysicalFlows.TwoDNavierStokes: energy, enstrophy


# ## Choosing a device: CPU or GPU

dev = CPU()    # Device (CPU/GPU)
nothing # hide


# ## Numerical, domain, and simulation parameters
#
# First, we pick some numerical and physical parameters for our model.

 n, L  = 256, 2π             # grid resolution and domain length
 ν, nν = 2e-7, 2             # hyperviscosity coefficient and hyperviscosity order
 μ, nμ = 1e-1, 0             # linear drag coefficient
    dt = 0.005               # timestep
nsteps = 4000                # total number of steps
 nsubs = 20                  # number of steps between each plot
nothing # hide


# ## Forcing

# We force the vorticity equation with stochastic excitation that is delta-correlated
# in time and while spatially homogeneously and isotropically correlated. The forcing
# has a spectrum with power in a ring in wavenumber space of radius ``k_f`` (`forcing_wavenumber`) and width 
# ``\delta k_f`` (`forcing_bandwidth`), and it injects energy per unit area and per unit time 
# equal to ``\varepsilon``. That is, the forcing covariance spectrum is proportional to 
# ``\exp{[-(|\bm{k}| - k_f)^2 / (2 \delta k_f^2)]}``.

forcing_wavenumber = 14.0 * 2π/L   # the central forcing wavenumber for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 1.5  * 2π/L   # the width of the forcing spectrum
ε = 0.1                            # energy input rate by the forcing

grid = TwoDGrid(dev, n, L)

K = @. sqrt(grid.Krsq)
forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
@. forcing_spectrum = ifelse(K < 2  * 2π/L, 0, forcing_spectrum)      # no power at low wavenumbers
@. forcing_spectrum = ifelse(K > 20 * 2π/L, 0, forcing_spectrum)      # no power at high wavenumbers
ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε / ε0             # normalize forcing to inject energy at rate ε

seed!(1234)
nothing # hide

# Next we construct function `calcF!` that computes a forcing realization every timestep.
# First we make sure that if `dev=GPU()`, then `CUDA.rand()` function is called for random
# numbers uniformy distributed between 0 and 1.
random_uniform = dev==CPU() ? rand : CUDA.rand

function calcF!(Fh, sol, t, clock, vars, params, grid::AbstractGrid{T}) where T
  @. Fh = sqrt(forcing_spectrum) * exp(2π * im * random_uniform(T)) / sqrt(clock.dt)

  @CUDA.allowscalar Fh[1, 1] = 0 # make sure forcing has zero domain-average

  return nothing
end
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. The
# `stepper` keyword defines the time-stepper to be used.
prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt, stepper="ETDRK4",
                                calcF=calcF!, stochastic=true)
nothing # hide

# Define some shortcuts for convenience.
sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

x, y = grid.x, grid.y
nothing # hide


# First let's see how a forcing realization looks like. Function `calcF!()` computes 
# the forcing in Fourier space and saves it into variable `vars.Fh`, so we first need to
# go back to physical space.
calcF!(vars.Fh, sol, 0.0, clock, vars, params, grid)

heatmap(x, y, irfft(vars.Fh, grid.nx)',
     aspectratio = 1,
               c = :balance,
            clim = (-200, 200),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "a forcing realization",
      framestyle = :box)


# ## Setting initial conditions

# Our initial condition is a fluid at rest.
TwoDNavierStokes.set_ζ!(prob, zeros(grid.nx, grid.ny))


# ## Diagnostics

# Create Diagnostics; the diagnostics are aimed to probe the energy budget.
E  = Diagnostic(energy,                prob, nsteps=nsteps) # energy
Z  = Diagnostic(enstrophy,             prob, nsteps=nsteps) # enstrophy
diags = [E, Z] # a list of Diagnostics passed to `stepforward!` will  be updated every timestep.
nothing # hide


# ## Visualizing the simulation

# We initialize a plot with the vorticity field and the time-series of
# energy and enstrophy diagnostics. To plot energy and enstrophy on the same
# axes we scale enstrophy with $k_f^2$.

p1 = heatmap(x, y, vars.ζ',
         aspectratio = 1,
                   c = :balance,
                clim = (-40, 40),
               xlims = (-L/2, L/2),
               ylims = (-L/2, L/2),
              xticks = -3:3,
              yticks = -3:3,
              xlabel = "x",
              ylabel = "y",
               title = "vorticity, t=" * @sprintf("%.2f", clock.t),
          framestyle = :box)

p2 = plot(2, # this means "a plot with two series"
               label = ["energy E(t)" "enstrophy Z(t) / k_f²"],
              legend = :right,
           linewidth = 2,
               alpha = 0.7,
              xlabel = "μ t",
               xlims = (0, 1.1 * μ * nsteps * dt),
               ylims = (0, 0.55))

l = @layout Plots.grid(1, 2)
p = plot(p1, p2, layout = l, size = (900, 420))


# ## Time-stepping the `Problem` forward

# Finally, we time-step the `Problem` forward in time.

startwalltime = time()

anim = @animate for j = 0:round(Int, nsteps / nsubs)
  if j % (1000/nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
    
    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Z: %.4f, walltime: %.2f min",
          clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)
    println(log)
  end  

  p[1][1][:z] = vars.ζ
  p[1][:title] = "vorticity, μt = " * @sprintf("%.2f", μ * clock.t)
  push!(p[2][1], μ * E.t[E.i], E.data[E.i])
  push!(p[2][2], μ * Z.t[Z.i], Z.data[Z.i] / forcing_wavenumber^2)

  stepforward!(prob, diags, nsubs)
  TwoDNavierStokes.updatevars!(prob)  
end

mp4(anim, "twodturb_forced.mp4", fps=18)
