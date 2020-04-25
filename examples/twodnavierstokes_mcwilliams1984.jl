# # Two-dimensional turbulence example
#
# In this example, we simulate decaying two-dimensional turbulence by solving
# the two-dimensional vorticity equation.

using FourierFlows, PyPlot, JLD2, Printf, Random, FFTW

using Random: seed!

import GeophysicalFlows.TwoDNavierStokes
import GeophysicalFlows.TwoDNavierStokes: energy, enstrophy
import GeophysicalFlows: peakedisotropicspectrum

# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)

# ## Numerical, domain, and simulation parameters
#
# First, we pick some numerical and physical parameters for our model.

n = 256
L = 2π
nothing # hide

## Then we pick the time-stepper parameters
    dt = 1e-2  # timestep
nsteps = 5000  # total number of steps
 nsubs = 200   # number of steps between each plot
 nothing # hide

## We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_McWilliams1984"
plotname = "snapshots"
filename = joinpath(filepath, "McWilliams1984.jld2")
nothing # hide

# File management
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. The
# `stepper` keyword defines the time-stepper to be used.
prob = TwoDNavierStokes.Problem(; nx=n, Lx=L, ν=ν, nν=nν, dt=dt, stepper="FilteredRK4", dev=dev)

# and define some shortcuts
sol, cl, vs, gr, filter = prob.sol, prob.clock, prob.vars, prob.grid, prob.timestepper.filter
x, y = gridpoints(gr)

# ## Setting initial conditions

# Our initial condition closely tries to reproduce the initial condition used
# in the paper by McWilliams (_JFM_, 1984)
seed!(1234)
k0, E0 = 6, 0.5
zetai  = peakedisotropicspectrum(gr, k0, E0, mask=filter)
TwoDNavierStokes.set_zeta!(prob, zetai)

# ## Diagnostics

# Create Diagnostics -- `energy` and `enstrophy` functions are imported at the top.
E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.

# ## Output

# Create Output
get_sol(prob) = Array(prob.sol) # extracts the Fourier-transformed solution
get_u(prob) = Array(irfft(im*gr.l.*gr.invKrsq.*sol, gr.nx))
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
saveproblem(out)

# ## Visualizing the simulation

# We define a function that plots the vorticity field and the evolution of
# energy and enstrophy diagnostics.

function plot_output(prob, fig, axs; drawcolorbar=false)
  # Plot the vorticity field and the evolution of energy and enstrophy.
  TwoDNavierStokes.updatevars!(prob)
  sca(axs[1])
  pcolormesh(x, y, vs.zeta)
  clim(-40, 40)
  axis("off")
  axis("square")
  if drawcolorbar==true
    colorbar()
  end

  sca(axs[2])
  cla()
  plot(E.t[1:E.i], E.data[1:E.i]/E.data[1])
  plot(Z.t[1:Z.i], Z.data[1:E.i]/Z.data[1])
  xlabel(L"t")
  ylabel(L"\Delta E, \, \Delta Z")

  pause(0.01)
end

# ## Time-stepping the `Problem` forwward

# Finally, we time-step the `Problem` forward in time.

fig, axs = subplots(ncols=2, nrows=1, figsize=(12, 4))
plot_output(prob, fig, axs; drawcolorbar=true)

startwalltime = time()
while cl.step < nsteps
  stepforward!(prob, diags, nsubs)
  saveoutput(out)

  log = @sprintf("step: %04d, t: %d, ΔE: %.4f, ΔZ: %.4f, τ: %.2f min",
    cl.step, cl.t, E.data[E.i]/E.data[1], Z.data[Z.i]/Z.data[1], (time()-startwalltime)/60)

  println(log)
  plot_output(prob, fig, axs; drawcolorbar=false)
end
println("finished")
plot_output(prob, fig, axs; drawcolorbar=false)

savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename, dpi=240)
