# # Decaying barotropic QG beta-plane turbulence
#
#md # This example can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/singlelayerqg_betadecay.ipynb).
# 
# An example of decaying barotropic quasi-geostrophic turbulence on a beta plane.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add GeophysicalFlows, Plots, Printf, Statistics, Random"
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

      n = 128            # 2D resolution: n² grid points
stepper = "FilteredRK4"  # timestepper
     dt = 0.04           # timestep
 nsteps = 2000           # total number of time-steps
 nsubs  = 20             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

L = 2π        # domain size
β = 10.0      # planetary PV gradient
μ = 0.0       # bottom drag
nothing # hide

# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. Not providing a viscosity 
# coefficient `ν` leads to the module's default value: `ν=0`. In this example numerical instability 
# due to accumulation of enstrophy at high wavenumbers is taken care with the `FilteredTimestepper` 
# we picked. Thus, we choose not to do any dealiasing by providing `aliased_fraction=0`.

prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, β=β, μ=μ,
                                  dt=dt, stepper=stepper, aliased_fraction=0)
nothing # hide

# and define some shortcuts
sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y
nothing # hide


# ## Setting initial conditions

# Our initial condition consist of a flow that has power only at wavenumbers with
# ``6 < \frac{L}{2\pi} \sqrt{k_x^2 + k_y^2} < 10`` and initial energy ``E_0``.
# `ArrayType()` function returns the array type appropriate for the device, i.e., `Array` for
# `dev = CPU()` and `CuArray` for `dev = GPU()`.

E₀ = 0.08 # energy of initial condition

K = @. sqrt(grid.Krsq)                          # a 2D array with the total wavenumber

Random.seed!(1234)
q₀h = ArrayType(dev)(randn(Complex{eltype(grid)}, size(sol)))
@. q₀h = ifelse(K < 6  * 2π/L, 0, q₀h)
@. q₀h = ifelse(K > 10 * 2π/L, 0, q₀h)
@. q₀h[1, :] = 0    # remove any power from zonal wavenumber k=0
q₀h *= sqrt(E₀ / SingleLayerQG.energy(q₀h, vars, params, grid)) # normalize q₀ to have energy E₀
q₀ = irfft(q₀h, grid.nx)

SingleLayerQG.set_q!(prob, q₀)
nothing # hide

# Let's plot the initial vorticity and streamfunction. Note that when plotting, we decorate 
# the variable to be plotted with `Array()` to make sure it is brought back on the CPU when 
# `vars` live on the GPU.

p1 = heatmap(x, y, Array(vars.q'),
         aspectratio = 1,
              c = :balance,
           clim = (-12, 12),
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
        levels = range(-0.7, stop=0.7, length=20), 
          clim = (-0.35, 0.35),
         xlims = (-grid.Lx/2, grid.Lx/2),
         ylims = (-grid.Ly/2, grid.Ly/2),
        xticks = -3:3,
        yticks = -3:3,
        xlabel = "x",
        ylabel = "y",
         title = "initial streamfunction ψ",
    framestyle = :box)

layout = @layout Plots.grid(1, 2)
p = plot(p1, p2, layout = layout, size = (800, 360))


# ## Diagnostics

# Create Diagnostics -- `energy` and `enstrophy` functions are imported at the top.
E = Diagnostic(SingleLayerQG.energy, prob; nsteps=nsteps)
Z = Diagnostic(SingleLayerQG.enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_decayingbetaturb"
plotname = "snapshots"
filename = joinpath(filepath, "decayingbetaturb.jld2")
nothing # hide

# Do some basic file management,
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

# and then create Output.
get_sol(prob) = sol # extracts the Fourier-transformed solution
out = Output(prob, filename, (:sol, get_sol))
nothing # hide


# ## Visualizing the simulation

# We define a function that plots the vorticity and streamfunction and 
# their corresponding zonal mean structure.

function plot_output(prob)
  q = Array(prob.vars.q)
  ψ = Array(prob.vars.ψ)
  q̄ = Array(mean(q, dims=1)')
  ū = Array(mean(prob.vars.u, dims=1)')

  pq = heatmap(x, y, q',
       aspectratio = 1,
            legend = false,
                 c = :balance,
              clim = (-12, 12),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "vorticity ∂v/∂x-∂u/∂y",
        framestyle = :box)

  pψ = contourf(x, y, ψ',
       aspectratio = 1,
            legend = false,
                 c = :viridis,
            levels = range(-0.7, stop=0.7, length=20), 
              clim = (-0.35, 0.35),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "streamfunction ψ",
        framestyle = :box)

  pqm = plot(q̄, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-2.2, 2.2),
            xlabel = "zonal mean q",
            ylabel = "y")
  plot!(pqm, 0*y, y, linestyle=:dash, linecolor=:black)

  pum = plot(ū, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-0.55, 0.55),
            xlabel = "zonal mean u",
            ylabel = "y")
  plot!(pum, 0*y, y, linestyle=:dash, linecolor=:black)

  layout = @layout Plots.grid(2, 2)
  p = plot(pq, pqm, pψ, pum, layout = layout, size = (800, 720))
  
  return p
end
nothing # hide


# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

startwalltime = time()

p = plot_output(prob)

anim = @animate for j = 0:round(Int, nsteps/nsubs)

  if j % round(Int, nsteps/nsubs / 4) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
      clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)

    println(log)
  end  

  p[1][1][:z] = Array(vars.q)
  p[1][:title] = "vorticity, t="*@sprintf("%.2f", clock.t)
  p[3][1][:z] = Array(vars.ψ)
  p[2][1][:x] = Array(mean(vars.q, dims=1)')
  p[4][1][:x] = Array(mean(vars.u, dims=1)')

  stepforward!(prob, diags, nsubs)
  SingleLayerQG.updatevars!(prob)

end

mp4(anim, "singlelayerqg_betadecay.mp4", fps=8)

# ## Save

# Finally, we can save, e.g., the last snapshot via
# ```julia
# savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), clock.step)
# savefig(savename)
# ```
