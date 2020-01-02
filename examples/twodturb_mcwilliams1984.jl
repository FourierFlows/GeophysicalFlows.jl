using FourierFlows, PyPlot, JLD2, Printf, Random, FFTW

using Random: seed!

import GeophysicalFlows.TwoDTurb
import GeophysicalFlows.TwoDTurb: energy, enstrophy
import GeophysicalFlows: peakedisotropicspectrum

dev = CPU()     # Device (CPU/GPU)

# Parameters
  n = 256
  L = 2π
 nν = 2
  ν = 0.0
 dt = 1e-2
nsteps = 5000
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
prob = TwoDTurb.Problem(; nx=n, Lx=L, ny=n, Ly=L, ν=ν, nν=nν, dt=dt, stepper="FilteredRK4", dev=dev)

sol, cl, vs, gr, filter = prob.sol, prob.clock, prob.vars, prob.grid, prob.timestepper.filter
x, y = gridpoints(gr)

# Initial condition closely following pyqg barotropic example
# that reproduces the results of the paper by McWilliams (1984)
seed!(1234)
k0, E0 = 6, 0.5
zetai  = peakedisotropicspectrum(gr, k0, E0, mask=filter)
TwoDTurb.set_zeta!(prob, zetai)

# Create Diagnostic -- energy and enstrophy are functions imported at the top.
E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will
# be updated every timestep.

# Create Output
get_sol(prob) = Array(prob.sol) # extracts the Fourier-transformed solution
get_u(prob) = Array(irfft(im*gr.l.*gr.invKrsq.*sol, gr.nx))
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
saveproblem(out)

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
  saveoutput(out)
  
  # Message
  log = @sprintf("step: %04d, t: %d, ΔE: %.4f, ΔZ: %.4f, τ: %.2f min",
    cl.step, cl.t, E.data[E.i]/E.data[1], Z.data[Z.i]/Z.data[1], (time()-startwalltime)/60)

  println(log)
  plot_output(prob, fig, axs; drawcolorbar=false)
end
println("finished")
plot_output(prob, fig, axs; drawcolorbar=false)

savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename, dpi=240)
