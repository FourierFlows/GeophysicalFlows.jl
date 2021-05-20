# # 2D forced-dissipative turbulence budgets
#
#md # This example can be run online via [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/twodnavierstokes_stochasticforcing_budgets.ipynb).
#md # Also, it can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/twodnavierstokes_stochasticforcing_budgets.ipynb).
#
# A simulation of forced-dissipative two-dimensional turbulence. We solve the
# two-dimensional vorticity equation with stochastic excitation and dissipation in
# the form of linear drag and hyperviscosity. As a demonstration, we compute how
# each of the forcing and dissipation terms contribute to the energy and the 
# enstrophy budgets.

using FourierFlows, Printf, Plots

using FourierFlows: parsevalsum
using Random: seed!
using FFTW: irfft

import GeophysicalFlows.TwoDNavierStokes
import GeophysicalFlows.TwoDNavierStokes: energy, energy_dissipation_hyperviscosity, energy_dissipation_hypoviscosity, energy_work
import GeophysicalFlows.TwoDNavierStokes: enstrophy, enstrophy_dissipation_hyperviscosity, enstrophy_dissipation_hypoviscosity, enstrophy_work


# ## Choosing a device: CPU or GPU

dev = CPU()    # Device (CPU/GPU)
nothing # hide


# ## Numerical, domain, and simulation parameters
#
# First, we pick some numerical and physical parameters for our model.

 n, L  = 256, 2π              # grid resolution and domain length
 ν, nν = 2e-7, 2              # hyperviscosity coefficient and hyperviscosity order
 μ, nμ = 1e-1, 0              # linear drag coefficient
dt, tf = 0.005, 0.2 / μ       # timestep and final time
    nt = round(Int, tf / dt)  # total timesteps
    ns = 4                    # how many intermediate times we want to plot
nothing # hide


# ## Forcing

# We force the vorticity equation with stochastic excitation that is delta-correlated in time 
# and while spatially homogeneously and isotropically correlated. The forcing has a spectrum 
# with power in a ring in wavenumber space of radius ``k_f`` (`forcing_wavenumber`) and width 
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
@. forcing_spectrum *= ε/ε0             # normalize forcing to inject energy at rate ε

seed!(1234)
nothing # hide

# Next we construct function `calcF!` that computes a forcing realization every timestep
function calcF!(Fh, sol, t, clock, vars, params, grid)
  ξ = ArrayType(dev)(exp.(2π * im * rand(eltype(grid), size(sol))) / sqrt(clock.dt))
  ξ[1, 1] = 0
  @. Fh = ξ * sqrt(forcing_spectrum)
  
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

# Create Diagnostics; the diagnostics are aimed to probe the energy and enstrophy budgets.
E  = Diagnostic(energy,                               prob, nsteps=nt) # energy
Rᵋ = Diagnostic(energy_dissipation_hypoviscosity,     prob, nsteps=nt) # energy dissipation by drag μ
Dᵋ = Diagnostic(energy_dissipation_hyperviscosity,    prob, nsteps=nt) # energy dissipation by drag μ
Wᵋ = Diagnostic(energy_work,                          prob, nsteps=nt) # energy work input by forcing
Z  = Diagnostic(enstrophy,                            prob, nsteps=nt) # enstrophy
Rᶻ = Diagnostic(enstrophy_dissipation_hypoviscosity,  prob, nsteps=nt) # enstrophy dissipation by drag μ
Dᶻ = Diagnostic(enstrophy_dissipation_hyperviscosity, prob, nsteps=nt) # enstrophy dissipation by drag μ
Wᶻ = Diagnostic(enstrophy_work,                       prob, nsteps=nt) # enstrophy work input by forcing
diags = [E, Dᵋ, Wᵋ, Rᵋ, Z, Dᶻ, Wᶻ, Rᶻ] # a list of Diagnostics passed to `stepforward!` will  be updated every timestep.
nothing # hide


# ## Visualizing the simulation

# We define a function that plots the vorticity field and the evolution of
# the diagnostics: energy, enstrophy, and all terms involved in the energy and 
# enstrophy budgets. Last, we also check (by plotting) whether the energy and enstrophy
# budgets are accurately computed, e.g., ``\mathrm{d}E/\mathrm{d}t = W^\varepsilon - 
# R^\varepsilon - D^\varepsilon``.

function computetendencies_and_makeplot(prob, diags)
  sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  TwoDNavierStokes.updatevars!(prob)

  E, Dᵋ, Wᵋ, Rᵋ, Z, Dᶻ, Wᶻ, Rᶻ = diags

  clocktime = round(μ * clock.t, digits=2)

  dEdt_numerical = (E[2:E.i] - E[1:E.i-1]) / clock.dt # numerical first-order approximation of energy tendency
  dZdt_numerical = (Z[2:Z.i] - Z[1:Z.i-1]) / clock.dt # numerical first-order approximation of enstrophy tendency

  dEdt_computed = Wᵋ[2:E.i] + Dᵋ[1:E.i-1] + Rᵋ[1:E.i-1]
  dZdt_computed = Wᶻ[2:Z.i] + Dᶻ[1:Z.i-1] + Rᶻ[1:Z.i-1]

  residual_E = dEdt_computed - dEdt_numerical
  residual_Z = dZdt_computed - dZdt_numerical

  εᶻ = parsevalsum(forcing_spectrum / 2, grid) / (grid.Lx * grid.Ly)

  pζ = heatmap(x, y, vars.ζ',
            aspectratio = 1,
            legend = false,
                 c = :viridis,
              clim = (-25, 25),
             xlims = (-L/2, L/2),
             ylims = (-L/2, L/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "μt",
            ylabel = "y",
             title = "∇²ψ(x, y, μt=" * @sprintf("%.2f", μ * clock.t) * ")",
        framestyle = :box)

  pζ = plot(pζ, size = (400, 400))

  t = E.t[2:E.i]

  p1E = plot(μ * t, [Wᵋ[2:E.i] ε.+0*t Dᵋ[1:E.i-1] Rᵋ[1:E.i-1]],
             label = ["energy work, Wᵋ" "ensemble mean energy work, <Wᵋ>" "dissipation, Dᵋ" "drag, Rᵋ = - 2μE"],
         linestyle = [:solid :dash :solid :solid],
         linewidth = 2,
             alpha = 0.8,
            xlabel = "μt",
            ylabel = "energy sources and sinks")

  p2E = plot(μ * t, [dEdt_computed, dEdt_numerical],
           label = ["computed Wᵋ-Dᵋ" "numerical dE/dt"],
       linestyle = [:solid :dashdotdot],
       linewidth = 2,
           alpha = 0.8,
          xlabel = "μt",
          ylabel = "dE/dt")

  p3E = plot(μ * t, residual_E,
           label = "residual dE/dt = computed - numerical",
       linewidth = 2,
           alpha = 0.7,
          xlabel = "μt")

  t = Z.t[2:E.i]

  p1Z = plot(μ * t, [Wᶻ[2:Z.i] εᶻ.+0*t Dᶻ[1:Z.i-1] Rᶻ[1:Z.i-1]],
           label = ["enstrophy work, Wᶻ" "mean enstrophy work, <Wᶻ>" "enstrophy dissipation, Dᶻ" "enstrophy drag, Rᶻ = - 2μZ"],
       linestyle = [:solid :dash :solid :solid],
       linewidth = 2,
           alpha = 0.8,
          xlabel = "μt",
          ylabel = "enstrophy sources and sinks")


  p2Z = plot(μ * t, [dZdt_computed, dZdt_numerical],
         label = ["computed Wᶻ-Dᶻ" "numerical dZ/dt"],
     linestyle = [:solid :dashdotdot],
     linewidth = 2,
         alpha = 0.8,
        xlabel = "μt",
        ylabel = "dZ/dt")

  p3Z = plot(μ * t, residual_Z,
         label = "residual dZ/dt = computed - numerical",
     linewidth = 2,
         alpha = 0.7,
        xlabel = "μt")
        
  layout = @layout Plots.grid(3, 2)
  
  pbudgets = plot(p1E, p1Z, p2E, p2Z, p3E, p3Z, layout=layout, size = (900, 1200))

  return pζ, pbudgets
end
nothing # hide




# ## Time-stepping the `Problem` forward

# Finally, we time-step the `Problem` forward in time.

startwalltime = time()
for i = 1:ns
  stepforward!(prob, diags, round(Int, nt/ns))
  
  TwoDNavierStokes.updatevars!(prob)
  
  cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, walltime: %.2f min", clock.step, clock.t,
        cfl, (time()-startwalltime)/60)

  println(log)
end


# ## Plot
# And now let's see what we got. First we plot the final snapshot 
# of the vorticity field.

pζ, pbudgets = computetendencies_and_makeplot(prob, diags)

pζ

# And finaly the energy and enstrophy budgets.

pbudgets
