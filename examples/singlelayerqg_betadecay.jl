# # Decaying barotropic QG beta-plane turbulence
#
#md # This example can be run online via [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/singlelayerqg_betadecay.ipynb). 
#md # Also, it can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/singlelayerqg_betadecay.ipynb).
# 
# An example of decaying barotropic quasi-geostrophic turbulence on a beta plane.

using FourierFlows, Plots, Printf, Random

using Statistics: mean
using FFTW: irfft

import GeophysicalFlows.SingleLayerQG
import GeophysicalFlows.SingleLayerQG: energy, enstrophy

# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
nothing # hide


# ## Numerical parameters and time-stepping parameters

      n = 128            # 2D resolution = n²
stepper = "FilteredRK4"  # timestepper
     dt = 0.05           # timestep
 nsteps = 1500           # total number of time-steps
 nsubs  = 10             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

L = 2π        # domain size
β = 10.0      # planetary PV gradient
μ = 0.0       # bottom drag
nothing # hide

# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. Not providing
# a viscosity coefficient `ν` leads to the module's default value: `ν=0`. In this
# example numerical instability due to accumulation of enstrophy at high wavenumbers
# is taken care with the `FilteredTimestepper` we picked. 
prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, β=β, μ=μ, dt=dt, stepper=stepper)
nothing # hide

# and define some shortcuts
sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y
nothing # hide


# ## Setting initial conditions

# Our initial condition consist of a flow that has power only at wavenumbers with
# ``8 < \frac{L}{2\pi} \sqrt{k_x^2 + k_y^2} < 10`` and initial energy ``E_0``:

E₀ = 0.1 # energy of initial condition

K = @. sqrt(grid.Krsq)                          # a 2D array with the total wavenumber
k = [grid.kr[i] for i=1:grid.nkr, j=1:grid.nl]  # a 2D array with the zonal wavenumber

Random.seed!(1234)
qih = randn(Complex{eltype(grid)}, size(sol))
@. qih = ifelse(K < 2  * 2π/L, 0, qih)
@. qih = ifelse(K > 10 * 2π/L, 0, qih)
@. qih = ifelse(k == 0 * 2π/L, 0, qih)            # no power at zonal wavenumber k=0 component
qih *= sqrt(E₀ / energy(qih, vars, params, grid)) # normalize qi to have energy E₀
qi = irfft(qih, grid.nx)

SingleLayerQG.set_ζ!(prob, qi)
nothing # hide

# Let's plot the initial vorticity field:

p1 = heatmap(x, y, vars.q',
         aspectratio = 1,
              c = :balance,
           clim = (-12, 12),
          xlims = (-grid.Lx/2, grid.Lx/2),
          ylims = (-grid.Ly/2, grid.Ly/2),
         xticks = -3:3,
         yticks = -3:3,
         xlabel = "x",
         ylabel = "y",
          title = "initial vorticity ζ=∂v/∂x-∂u/∂y",
     framestyle = :box)

p2 = contourf(x, y, vars.ψ',
        aspectratio = 1,
             c = :viridis,
        levels = range(-0.65, stop=0.65, length=10), 
          clim = (-0.65, 0.65),
         xlims = (-grid.Lx/2, grid.Lx/2),
         ylims = (-grid.Ly/2, grid.Ly/2),
        xticks = -3:3,
        yticks = -3:3,
        xlabel = "x",
        ylabel = "y",
         title = "initial streamfunction ψ",
    framestyle = :box)

l = @layout Plots.grid(1, 2)
p = plot(p1, p2, layout=l, size = (900, 800))


# ## Diagnostics

# Create Diagnostics -- `energy` and `enstrophy` functions are imported at the top.
E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
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
  ζ = prob.vars.ζ
  ψ = prob.vars.ψ
  ζ̄ = mean(ζ, dims=1)'
  ū = mean(prob.vars.u, dims=1)'

  pζ = heatmap(x, y, ζ',
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
             title = "vorticity ζ=∂v/∂x-∂u/∂y",
        framestyle = :box)

  pψ = contourf(x, y, ψ',
       aspectratio = 1,
            legend = false,
                 c = :viridis,
            levels = range(-0.65, stop=0.65, length=10), 
              clim = (-0.65, 0.65),
             xlims = (-grid.Lx/2, grid.Lx/2),
             ylims = (-grid.Ly/2, grid.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "streamfunction ψ",
        framestyle = :box)

  pζm = plot(ζ̄, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-2.2, 2.2),
            xlabel = "zonal mean ζ",
            ylabel = "y")
  plot!(pζm, 0*y, y, linestyle=:dash, linecolor=:black)

  pum = plot(ū, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-0.55, 0.55),
            xlabel = "zonal mean u",
            ylabel = "y")
  plot!(pum, 0*y, y, linestyle=:dash, linecolor=:black)

  l = @layout Plots.grid(2, 2)
  p = plot(pζ, pζm, pψ, pum, layout = l, size = (900, 800))
  
  return p
end
nothing # hide


# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

startwalltime = time()

p = plot_output(prob)

anim = @animate for j = 0:round(Int, nsteps/nsubs)

  if j % (500 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
      clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)

    println(log)
  end  

  p[1][1][:z] = vars.ζ
  p[1][:title] = "vorticity, t="*@sprintf("%.2f", clock.t)
  p[3][1][:z] = vars.ψ
  p[2][1][:x] = mean(vars.ζ, dims=1)'
  p[4][1][:x] = mean(vars.u, dims=1)'

  stepforward!(prob, diags, nsubs)
  SingleLayerQG.updatevars!(prob)

end

gif(anim, "barotropicqg_betadecay.gif", fps=8)

# ## Save

# Finally save the last snapshot.
savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), clock.step)
savefig(savename)
