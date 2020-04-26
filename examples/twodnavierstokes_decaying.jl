# # Two-dimensional turbulence example
#
# In this example, we simulate decaying two-dimensional turbulence by solving
# the two-dimensional vorticity equation.

using FourierFlows, PyPlot, JLD2, Printf, Random, FFTW

using Random: seed!
using FFTW: rfft

import GeophysicalFlows.TwoDNavierStokes
import GeophysicalFlows.TwoDNavierStokes: energy, enstrophy
import GeophysicalFlows: peakedisotropicspectrum

# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
nothing # hide

# ## Numerical, domain, and simulation parameters
#
# First, we pick some numerical and physical parameters for our model.

n, L  = 256, 2π             # grid resolution and domain length
nothing # hide

## Then we pick the time-stepper parameters
    dt = 5e-3  # timestep
nsteps = 5000  # total number of steps
 nsubs = 200   # number of steps between each plot
 nothing # hide

## We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_decayingTwoDNavierStokes"
plotname = "snapshots"
filename = joinpath(filepath, "decayingTwoDNavierStokes.jld2")
nothing # hide

# File management
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. The
# `stepper` keyword defines the time-stepper to be used,
prob = TwoDNavierStokes.Problem(; nx=n, Lx=L, ny=n, Ly=L, ν=ν, nν=nν, dt=dt, stepper="FilteredRK4", dev=dev)
nothing # hide

# and define some shortcuts
sol, cl, vs, gr, filter = prob.sol, prob.clock, prob.vars, prob.grid, prob.timestepper.filter
x, y = gridpoints(gr)
nothing # hide

# ## Setting initial conditions

# Our initial condition closely tries to reproduce the initial condition used
# in the paper by McWilliams (_JFM_, 1984)
seed!(1234)
k0, E0 = 6, 0.5
zetai  = peakedisotropicspectrum(gr, k0, E0, mask=filter)
TwoDNavierStokes.set_zeta!(prob, zetai)
nothing # hide

# ## Diagnostics

# Create Diagnostics -- `energy` and `enstrophy` functions are imported at the top.
E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide

# ## Output

# Create Output
get_sol(prob) = Array(prob.sol) # extracts the Fourier-transformed solution
get_u(prob) = Array(irfft(im*gr.l.*gr.invKrsq.*sol, gr.nx))
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
saveproblem(out)
nothing # hide

# ## Visualizing the simulation

# We define a function that plots the vorticity field and the evolution of
# energy and enstrophy diagnostics.

function plot_output(prob, fig, axs; drawcolorbar=false)
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
end

# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

startwalltime = time()
while cl.step < nsteps
  stepforward!(prob, diags, nsubs)
  saveoutput(out)

  log = @sprintf("step: %04d, t: %d, ΔE: %.4f, ΔZ: %.4f, τ: %.2f min",
    cl.step, cl.t, E.data[E.i]/E.data[1], Z.data[Z.i]/Z.data[1], (time()-startwalltime)/60)

  println(log)
end
println("finished")

# And now let's see what we got. We plot the output and save.

fig, axs = subplots(ncols=2, nrows=1, figsize=(12, 4))
plot_output(prob, fig, axs; drawcolorbar=true)

savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename, dpi=240)

gcf()

# ## Radial energy spectrum

# After the simulation is done we plot the radial energy spectrum to illustrate
# how `FourierFlows.radialspectrum` can be used,

E  = @. 0.5*(vs.u^2 + vs.v^2) # energy density
Eh = rfft(E)                  # Fourier transform of energy density
kr, Ehr = FourierFlows.radialspectrum(Eh, gr, refinement=1) # compute radial specturm of `Eh`
nothing # hide

# and we plot it.

fig2, axs = subplots(ncols=2, figsize=(8, 4))

sca(axs[1])
pcolormesh(x, y, vs.zeta)
xlabel(L"x")
ylabel(L"y")
title("Vorticity")
clim(-40, 40)
axis("off")
axis("square")

sca(axs[2])
plot(kr, abs.(Ehr))
xlabel(L"k_r")
ylabel(L"\int | \hat{E} | \, k_r \,\mathrm{d} k_{\theta}")
title("Radial energy spectrum")

xlim(0, gr.nx/4)
axs[2].set_yscale("log")

tight_layout(w_pad=0.1)
gcf()