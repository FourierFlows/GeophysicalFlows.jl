# # 2D decaying turbulence
#
#md # This example can be run online via [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/twodnavierstokes_decaying.ipynb).
#md # Also, it can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/twodnavierstokes_decaying.ipynb).
#
# A simulation of decaying two-dimensional turbulence.

using FourierFlows, Printf, Random, Plots
 
using Random: seed!
using FFTW: rfft, irfft

import GeophysicalFlows.TwoDNavierStokes
import GeophysicalFlows.TwoDNavierStokes: energy, enstrophy
import GeophysicalFlows: peakedisotropicspectrum


# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
nothing # hide


# ## Numerical, domain, and simulation parameters
#
# First, we pick some numerical and physical parameters for our model.

n, L  = 128, 2π             # grid resolution and domain length
nothing # hide

## Then we pick the time-stepper parameters
    dt = 1e-2  # timestep
nsteps = 4000  # total number of steps
 nsubs = 20    # number of steps between each plot
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. The
# `stepper` keyword defines the time-stepper to be used.
prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ny=n, Ly=L, dt=dt, stepper="FilteredRK4")
nothing # hide

# Next we define some shortcuts for convenience.
sol, clock, vars, grid = prob.sol, prob.clock, prob.vars, prob.grid
x, y = grid.x, grid.y
nothing # hide


# ## Setting initial conditions

# Our initial condition closely tries to reproduce the initial condition used
# in the paper by McWilliams (_JFM_, 1984)
seed!(1234)
k₀, E₀ = 6, 0.5
ζ₀ = peakedisotropicspectrum(grid, k₀, E₀, mask=prob.timestepper.filter)
TwoDNavierStokes.set_ζ!(prob, ζ₀)
nothing # hide

# Let's plot the initial vorticity field:
heatmap(x, y, vars.ζ',
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
            

# ## Diagnostics

# Create Diagnostics -- `energy` and `enstrophy` functions are imported at the top.
E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_decayingTwoDNavierStokes"
plotname = "snapshots"
filename = joinpath(filepath, "decayingTwoDNavierStokes.jld2")
nothing # hide

# Do some basic file management
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

# And then create Output
get_sol(prob) = prob.sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im * grid.l .* grid.invKrsq .* sol, grid.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
saveproblem(out)
nothing # hide


# ## Visualizing the simulation

# We initialize a plot with the vorticity field and the time-series of
# energy and enstrophy diagnostics.

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
               label = ["energy E(t)/E(0)" "enstrophy Z(t)/Z(0)"],
              legend = :right,
           linewidth = 2,
               alpha = 0.7,
              xlabel = "t",
               xlims = (0, 41),
               ylims = (0, 1.1))

l = @layout Plots.grid(1, 2)
p = plot(p1, p2, layout = l, size = (900, 400))


# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

startwalltime = time()

anim = @animate for j = 0:Int(nsteps/nsubs)
  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
    
    log = @sprintf("step: %04d, t: %d, cfl: %.2f, ΔE: %.4f, ΔZ: %.4f, walltime: %.2f min",
        clock.step, clock.t, cfl, E.data[E.i]/E.data[1], Z.data[Z.i]/Z.data[1], (time()-startwalltime)/60)

    println(log)
  end  

  p[1][1][:z] = vars.ζ
  p[1][:title] = "vorticity, t=" * @sprintf("%.2f", clock.t)
  push!(p[2][1], E.t[E.i], E.data[E.i]/E.data[1])
  push!(p[2][2], Z.t[Z.i], Z.data[Z.i]/Z.data[1])

  stepforward!(prob, diags, nsubs)
  TwoDNavierStokes.updatevars!(prob)  
end

gif(anim, "twodturb.gif", fps=18)


# Last we save the output.
saveoutput(out)


# ## Radial energy spectrum

# After the simulation is done we plot the instantaneous radial energy spectrum to illustrate
# how `FourierFlows.radialspectrum` can be used,

E  = @. 0.5 * (vars.u^2 + vars.v^2) # energy density
Eh = rfft(E)                  # Fourier transform of energy density
kr, Ehr = FourierFlows.radialspectrum(Eh, grid, refinement=1) # compute radial specturm of `Eh`
nothing # hide

# and we plot it.
plot(kr, abs.(Ehr),
    linewidth = 2,
        alpha = 0.7,
       xlabel = "kᵣ", ylabel = "∫ |Ê| kᵣ dk_θ",
        xlims = (5e-1, grid.nx),
       xscale = :log10, yscale = :log10,
        title = "Radial energy spectrum",
       legend = false)
