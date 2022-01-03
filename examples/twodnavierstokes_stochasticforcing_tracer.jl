# # 2D forced-dissipative turbulence
#
#md # This example can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/literated/twodnavierstokes_stochasticforcing.ipynb).
#
# A simulation of forced-dissipative two-dimensional turbulence. We solve the
# two-dimensional vorticity equation with stochastic excitation and dissipation in
# the form of linear drag and hyperviscosity. 
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add GeophysicalFlows, Random, Printf, Plots"
# ```

# ## Let's begin
# Let's load `GeophysicalFlows.jl` and some other needed packages.
#
using GeophysicalFlows, Random, Printf, Plots
parsevalsum = FourierFlows.parsevalsum

# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
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
 ntracers = 3   # number of tracers
nothing # hide


# ## Forcing

# We force the vorticity equation with stochastic excitation that is delta-correlated in time 
# and while spatially homogeneously and isotropically correlated. The forcing has a spectrum 
# with power in a ring in wavenumber space of radius ``k_f`` (`forcing_wavenumber`) and width 
# ``δ_f`` (`forcing_bandwidth`), and it injects energy per unit area and per unit time 
# equal to ``\varepsilon``. That is, the forcing covariance spectrum is proportional to 
# ``\exp{[-(|\bm{k}| - k_f)^2 / (2 δ_f^2)]}``.

forcing_wavenumber = 14.0 * 2π/L  # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 1.5  * 2π/L  # the width of the forcing spectrum, `δ_f`
ε = 0.1                           # energy input rate by the forcing

grid = TwoDGrid(dev, n, L)

K = @. sqrt(grid.Krsq)             # a 2D array with the total wavenumber

forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
@CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average

ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε/ε0        # normalize forcing to inject energy at rate ε
nothing # hide


# We reset of the random number generator for reproducibility
if dev==CPU(); Random.seed!(1234); else; CUDA.seed!(1234); end
nothing # hide


# Next we construct function `calcF!` that computes a forcing realization every timestep.
# First we make sure that if `dev=GPU()`, then `CUDA.rand()` function is called for random
# numbers uniformly distributed between 0 and 1.
random_uniform = dev==CPU() ? rand : CUDA.rand

function calcF!(Fh, sol, t, clock, vars, params, grid) 
  Fh .= sqrt.(forcing_spectrum) .* exp.(2π * im * random_uniform(eltype(grid), size(sol[:,:,1]))) ./ sqrt(clock.dt)
  
  return nothing
end
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. The
# `stepper` keyword defines the time-stepper to be used.
prob = TwoDNavierStokesTracer.Problem(ntracers, dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt, stepper="ETDRK4",
                                calcF=calcF!, stochastic=true)
nothing # hide

# Define some shortcuts for convenience.
sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

x, y = grid.x, grid.y
nothing # hide


# First let's see how a forcing realization looks like. Function `calcF!()` computes 
# the forcing in Fourier space and saves it into variable `vars.Fh`, so we first need to
# go back to physical space.

# Note that when plotting, we decorate the variable to be plotted with `Array()` to make sure 
# it is brought back on the CPU when the variable lives on the GPU.
calcF!(vars.Fh, sol, 0.0, clock, vars, params, grid)

heatmap(x, y, Array(irfft(vars.Fh, grid.nx)'),
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

ζ₀ = ArrayType(dev)(zeros(grid.nx, grid.ny))

# Our initial condition for tracers (``c``).
# For the first tracer we use a gaussian centered at ``(x, y) = (L_x/5, 0)``.
gaussian(x, y, σ) = exp(-(x^2 + y^2) / (2σ^2))

amplitude, spread = 0.5, 0.15
for j in 1:ntracers
  if j>1
      global c₀ = cat(c₀, 0.0.*ζ₀, dims=3)
  else
      global c₀ = 0.0.*ζ₀
  end
end
c₀[:,:,1] = [amplitude * gaussian(x[i] - 0.1 * grid.Lx, y[j], spread) for i=1:grid.nx, j=1:grid.ny]
# For the second tracer we use a wider gaussian centered at a different location
c₀[:,:,2] = [amplitude * gaussian(x[i] - 0.2 * grid.Lx, y[j] + 0.1 * grid.Ly, 4*spread) for i=1:grid.nx, j=1:grid.ny]
# For the third tracer we use a straight band in the x-direction that is an eighth as wide as the domain
width = 1/8
[c₀[i,j,3] = amplitude for i=1:grid.nx, j=Int( (1/2) * grid.ny ):Int( (1/2 + width) * grid.ny )]

TwoDNavierStokesTracer.set_ζ_and_tracers!(prob, ζ₀, c₀)
nothing # hide


# ## Diagnostics

# Create Diagnostics; the diagnostics are aimed to probe the energy budget.
E  = Diagnostic(TwoDNavierStokesTracer.energy,                prob, nsteps=nsteps) # energy
Z  = Diagnostic(TwoDNavierStokesTracer.enstrophy,             prob, nsteps=nsteps) # enstrophy
diags = [E, Z] # a list of Diagnostics passed to `stepforward!` will  be updated every timestep.
nothing # hide


# ## Visualizing the simulation

# We initialize a plot with the vorticity field and the time-series of
# energy and enstrophy diagnostics. To plot energy and enstrophy on the same
# axes we scale enstrophy with ``k_f^2``.

p1 = heatmap(x, y, Array(vars.ζ[:,:,1]'),
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

ptimeϕ₁ = heatmap(x, y, Array(vars.ζ[:,:,2]'),
            aspectratio = 1,
                      c = :balance,
                   clim = (-0.5, 0.5),
                  xlims = (-L/2, L/2),
                  ylims = (-L/2, L/2),
                 xticks = -3:3,
                 yticks = -3:3,
                 xlabel = "x",
                 ylabel = "y",
                  title = "ϕ₁(x, y, t=" * @sprintf("%.2f", clock.t) * ")",
             framestyle = :box)
      
  ptimeϕ₂ = heatmap(x, y, Array(vars.ζ[:,:,3]'),
             aspectratio = 1,
                       c = :balance,
                    clim = (-0.5, 0.5),
                   xlims = (-L/2, L/2),
                   ylims = (-L/2, L/2),
                  xticks = -3:3,
                  yticks = -3:3,
                  xlabel = "x",
                  ylabel = "y",
                   title = "ϕ₂(x, y, t=" * @sprintf("%.2f", clock.t) * ")",
              framestyle = :box)
      
  ptimeϕ₃ = heatmap(x, y, Array(vars.ζ[:,:,4]'),
              aspectratio = 1,
                        c = :balance,
                     clim = (-0.5, 0.5),
                    xlims = (-L/2, L/2),
                    ylims = (-L/2, L/2),
                   xticks = -3:3,
                   yticks = -3:3,
                   xlabel = "x",
                   ylabel = "y",
                    title = "ϕ₃(x, y, t=" * @sprintf("%.2f", clock.t) * ")",
               framestyle = :box)

p = plot(p1, p2, ptimeϕ₁, ptimeϕ₂, ptimeϕ₃, layout=5, size = (1500, 800))
#l = @layout Plots.grid(1, 2)
#p = plot(p1, p2, layout = l, size = (900, 420))


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

  p[1][1][:z] = Array(vars.ζ[:,:,1])
  p[1][:title] = "vorticity, μt = " * @sprintf("%.2f", μ * clock.t)
  push!(p[2][1], μ * E.t[E.i], E.data[E.i])
  push!(p[2][2], μ * Z.t[Z.i], Z.data[Z.i] / forcing_wavenumber^2)
  p[3][1][:z] = Array(vars.ζ[:,:,2])
  p[4][1][:z] = Array(vars.ζ[:,:,3])
  p[5][1][:z] = Array(vars.ζ[:,:,4])

  stepforward!(prob, diags, nsubs)
  TwoDNavierStokesTracer.updatevars!(prob)  
end

mp4(anim, "twodturb_forced_tracer.mp4", fps=18)
