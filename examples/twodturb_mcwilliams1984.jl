using FourierFlows, PyPlot, JLD2, Printf, Random

using Random: seed!

import GeophysicalFlows.TwoDTurb
import GeophysicalFlows.TwoDTurb: energy, enstrophy
import GeophysicalFlows: peakedisotropicspectrum

# Parameters
  n = 128
  L = 2π
nnu = 2
 nu = 0.0
 dt = 5e-3
nsteps = 8000
nsubs = 200

# Files
filepath = "."
plotpath = "./plots_McWilliams1984"
plotname = "snapshots"
filename = joinpath(filepath, "McWilliams1984.jld2")

# File management
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end

# Initialize problem
prob = TwoDTurb.Problem(; nx=n, Lx=L, ny=n, Ly=L, nu=nu, nnu=nnu, dt=dt, stepper="FilteredRK4")

sol, cl, vs, gr, filter = prob.sol, prob.clock, prob.vars, prob.grid, prob.timestepper.filter
x, y = gridpoints(gr)

# Initial condition closely following pyqg barotropic example
# that reproduces the results of the paper by McWilliams (1984)
seed!(1234)
k0, E0 = 6, 0.5
zetai = peakedisotropicspectrum(gr, k0, E0, mask=filter)
TwoDTurb.set_zeta!(prob, zetai)

# Create Diagnostic -- energy and enstrophy are functions imported at the top.
E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will
# be updated every timestep.

# Create Output
get_sol(prob) = prob.sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*gr.l.*gr.invKrsq.*sol, gr.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))


function plot_output(prob, fig, axs; drawcolorbar=false)
  # Plot the vorticity field and the evolution of energy and enstrophy.
  TwoDTurb.updatevars!(prob)
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

# Step forward

fig, axs = subplots(ncols=2, nrows=1, figsize=(12, 4))
plot_output(prob, fig, axs; drawcolorbar=true)

startwalltime = time()
while cl.step < nsteps
  stepforward!(prob, diags, nsubs)

  # Message
  log = @sprintf("step: %04d, t: %d, ΔE: %.4f, ΔZ: %.4f, τ: %.2f min",
    cl.step, cl.t, E.data[E.i]/E.data[1], Z.data[Z.i]/Z.data[1], (time()-startwalltime)/60)

  println(log)
  plot_output(prob, fig, axs; drawcolorbar=false)
end

plot_output(prob, fig, axs; drawcolorbar=false)

savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename, dpi=240)
