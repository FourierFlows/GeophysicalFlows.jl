# # Decaying barotropic QG turbulence over topography
#
#md # This example can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/singlelayerqg_decaying_topography.ipynb).
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


# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
nothing # hide


# ## Numerical parameters and time-stepping parameters

      n = 128            # 2D resolution = n²
stepper = "FilteredRK4"  # timestepper
     dt = 0.05           # timestep
 nsteps = 2000           # total number of time-steps
 nsubs  = 10             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

L = 2π        # domain size
nothing # hide

# Define the topographic potential vorticity, ``\eta = f_0 h(x, y)/H``. The topography here is 
# an elliptical mount at ``(x, y) = (1, 1)``, and an elliptical depression at ``(x, y) = (-1, -1)``.
σx, σy = 0.4, 0.8
topographicPV(x, y) = 3exp(-(x-1)^2/(2σx^2) -(y-1)^2/(2σy^2)) - 2exp(-(x+1)^2/(2σx^2) -(y+1)^2/(2σy^2))
nothing # hide

# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. Not providing a viscosity 
# coefficient `ν` leads to the module's default value: `ν=0`. In this example numerical instability 
# due to accumulation of enstrophy in high wavenumbers is taken care with the `FilteredTimestepper` 
# we picked. Thus, we choose not to do any dealiasing by providing `aliased_fraction=0`.
#
# The topophic PV is prescribed via keyword argument `eta`.
prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, eta=topographicPV,
                                  dt=dt, stepper=stepper, aliased_fraction=0)
nothing # hide

# and define some shortcuts
sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y
nothing # hide

# and let's plot the topographic PV. Note that when plotting, we decorate the variable to be 
# plotted with `Array()` to make sure it is brought back on the CPU when the variable lives 
# on the GPU.
contourf(grid.x, grid.y, Array(params.eta'),
          aspectratio = 1,
            linewidth = 0,
               levels = 10,
                    c = :balance,
                 clim = (-3, 3),
                xlims = (-grid.Lx/2, grid.Lx/2),
                ylims = (-grid.Ly/2, grid.Ly/2),
               xticks = -3:3,
               yticks = -3:3,
               xlabel = "x",
               ylabel = "y",
                title = "topographic PV η=f₀h/H")


# ## Setting initial conditions

# Our initial condition consist of a flow that has power only at wavenumbers with
# ``6 < \frac{L}{2\pi} \sqrt{k_x^2 + k_y^2} < 12`` and initial energy ``E_0``.
# `ArrayType()` function returns the array type appropriate for the device, i.e., `Array` for
# `dev = CPU()` and `CuArray` for `dev = GPU()`.

E₀ = 0.04 # energy of initial condition

K = @. sqrt(grid.Krsq)                             # a 2D array with the total wavenumber

Random.seed!(1234)
qih = ArrayType(dev)(randn(Complex{eltype(grid)}, size(sol)))
@. qih = ifelse(K < 6  * 2π/L, 0, qih)
@. qih = ifelse(K > 12 * 2π/L, 0, qih)
qih *= sqrt(E₀ / SingleLayerQG.energy(qih, vars, params, grid))  # normalize qi to have energy E₀
qi = irfft(qih, grid.nx)

SingleLayerQG.set_q!(prob, qi)
nothing # hide

# Let's plot the initial vorticity field:

p1 = heatmap(x, y, Array(vars.q'),
         aspectratio = 1,
              c = :balance,
           clim = (-8, 8),
          xlims = (-grid.Lx/2, grid.Lx/2),
          ylims = (-grid.Ly/2, grid.Ly/2),
         xticks = -3:3,
         yticks = -3:3,
         xlabel = "x",
         ylabel = "y",
          title = "initial vorticity ∂v/∂x-∂u/∂y",
     framestyle = :box)

p2 = contourf(x, y, Array(vars.ψ'),
        aspectratio = 1,
             c = :viridis,
        levels = range(-0.25, stop=0.25, length=11), 
          clim = (-0.25, 0.25),
         xlims = (-grid.Lx/2, grid.Lx/2),
         ylims = (-grid.Ly/2, grid.Ly/2),
        xticks = -3:3,
        yticks = -3:3,
        xlabel = "x",
        ylabel = "y",
         title = "initial streamfunction ψ",
    framestyle = :box)

layout = @layout Plots.grid(1, 2)
p = plot(p1, p2, layout=layout, size = (800, 360))


# ## Diagnostics

# Create Diagnostics -- `energy` and `enstrophy` functions are imported at the top.
E = Diagnostic(SingleLayerQG.energy, prob; nsteps=nsteps)
Z = Diagnostic(SingleLayerQG.enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide


# ## Output

# We choose folder for outputing `.jld2` files.
filepath = "."
filename = joinpath(filepath, "decayingbetaturb.jld2")
nothing # hide

# Do some basic file management,
if isfile(filename); rm(filename); end
nothing # hide

# and then create Output.
get_sol(prob) = sol # extracts the Fourier-transformed solution
out = Output(prob, filename, (:sol, get_sol))
nothing # hide


# ## Visualizing the simulation

# We define a function that plots the vorticity and streamfunction and 
# their corresponding zonal mean structure.

function plot_output(prob)
  q = prob.vars.q
  ψ = prob.vars.ψ
  η = prob.params.eta

  pq = heatmap(x, y, Array(q'),
       aspectratio = 1,
            legend = false,
                 c = :balance,
              clim = (-6, 6),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "vorticity ∂v/∂x-∂u/∂y",
        framestyle = :box)
  
  contour!(pq, x, y, Array(η'),
          levels=0.5:0.5:3,
          lw=2, c=:black, ls=:solid, alpha=0.7)
  
  contour!(pq, x, y, Array(η'),
          levels=-2:0.5:-0.5,
          lw=2, c=:black, ls=:dash, alpha=0.7)
  
  pψ = contourf(x, y, Array(ψ'),
       aspectratio = 1,
            legend = false,
                 c = :viridis,
            levels = range(-0.75, stop=0.75, length=31),
              clim = (-0.75, 0.75),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "streamfunction ψ",
        framestyle = :box)

  l = @layout Plots.grid(1, 2)
  p = plot(pq, pψ, layout = l, size = (800, 360))
  
  return p
end
nothing # hide


# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

startwalltime = time()

p = plot_output(prob)

anim = @animate for j = 0:round(Int, nsteps/nsubs)

  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
      clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)

    println(log)
  end  

  p[1][1][:z] = Array(vars.q)
  p[1][:title] = "vorticity, t="*@sprintf("%.2f", clock.t)
  p[2][1][:z] = Array(vars.ψ)

  stepforward!(prob, diags, nsubs)
  SingleLayerQG.updatevars!(prob)
end

mp4(anim, "singlelayerqg_decaying_topography.mp4", fps=12)
