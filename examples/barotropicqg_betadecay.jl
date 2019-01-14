using
  PyPlot,
  JLD2,
  Printf,
  Random,
  FourierFlows

using Statistics: mean
using FFTW: irfft

import GeophysicalFlows.BarotropicQG
import GeophysicalFlows.BarotropicQG: energy, enstrophy

# Numerical parameters and time-stepping parameters
nx  = 256      # 2D resolution = nx^2
stepper = "FilteredETDRK4"   # timestepper
dt  = 0.02     # timestep
nsteps = 8000  # total number of time-steps
nsubs  = 500   # number of time-steps for plotting
               # (nsteps must be multiple of nsubs)

# Physical parameters
 Lx  = 2π      # domain size
 nu  = 0e-05   # viscosity
 nnu = 1       # viscosity order
beta = 15.0    # planetary PV gradient
mu   = 0e-1    # bottom drag

# Initialize problem
prob = BarotropicQG.InitialValueProblem(nx=nx, Lx=Lx, beta=beta, nu=nu,
                                        nnu=nnu, mu=mu, dt=dt, stepper=stepper)
sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

# Files
filepath = "."
plotpath = "./plots_decayingbetaturb"
plotname = "snapshots"
filename = joinpath(filepath, "decayingbetaturb.jld2")

# File management
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end




# Initial condition that has power only at wavenumbers with
# 8<L/(2π)*sqrt(kx^2+ky^2)<10 and initial energy E0
Random.seed!(1234)
E0 = 0.1
modk = ones(g.nkr, g.nl)
modk[real.(g.Krsq).<(8*2*pi/g.Lx)^2] .= 0
modk[real.(g.Krsq).>(10*2*pi/g.Lx)^2] .= 0
modk[1, :] .= 0
psih = (randn(Float64, size(sol)) .+ im*randn(Float64, size(sol))).*modk
psih = @. psih*prob.timestepper.filter
Ein = real(sum(g.Krsq.*abs2.(psih)/(g.nx*g.ny)^2))
psih = psih*sqrt(E0/Ein)
qi = -irfft(g.Krsq.*psih, g.nx)
E0 = FourierFlows.parsevalsum(g.Krsq.*abs2.(psih), g)

BarotropicQG.set_zeta!(prob, qi)

# Create Diagnostic -- "energy" and "enstrophy" are functions imported at the top.
E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will
# be updated every timestep. They should be efficient to calculate and
# have a small memory footprint. (For example, the domain-integrated kinetic
# energy is just a single number for each timestep). See the file in
# src/diagnostics.jl and the stepforward! function in timesteppers.jl.

# Create Output
get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*g.l.*g.invKrsq.*sol, g.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))

x, y = gridpoints(g)

function plot_output(prob, fig, axs; drawcolorbar=false)
  # Plot the vorticity and streamfunction fields as well as the zonal mean
  # vorticity and the zonal mean zonal velocity.

  sol, v, p, g = prob.sol, prob.vars, prob.params, prob.grid
  BarotropicQG.updatevars!(prob)

  sca(axs[1])
  cla()
  pcolormesh(x, y, v.q)
  axis("square")
  xticks(-2:2:2)
  yticks(-2:2:2)
  title(L"vorticity $\zeta = \partial_x v - \partial_y u$")
  if drawcolorbar==true
    colorbar()
  end

  sca(axs[2])
  cla()
  pcolormesh(x, y, v.psi)
  axis("square")
  xticks(-2:2:2)
  yticks(-2:2:2)
  title(L"streamfunction $\psi$")
  if drawcolorbar==true
    colorbar()
  end

  sca(axs[3])
  cla()
  plot(Array(transpose(mean(v.zeta, dims=1))), y[1,:])
  plot(0*y[1,:], y[1,:], "k--")
  ylim(-Lx/2, Lx/2)
  xlim(-2, 2)
  title(L"zonal mean $\zeta$")

  sca(axs[4])
  cla()
  plot(Array(transpose(mean(v.u, dims=1))), y[1,:])
  plot(0*y[1,:], y[1,:], "k--")
  ylim(-Lx/2, Lx/2)
  xlim(-0.5, 0.5)
  title(L"zonal mean $u$")

  pause(0.001)
end



fig, axs = subplots(ncols=2, nrows=2, figsize=(8, 8))
plot_output(prob, fig, axs; drawcolorbar=false)

# Step forward
startwalltime = time()

while cl.step < nsteps
  stepforward!(prob, diags, nsubs)

  # Message
  log = @sprintf("step: %04d, t: %d, E: %.4f, Q: %.4f, τ: %.2f min",
    cl.step, cl.t, E.data[E.i], Z.data[Z.i],
    (time()-startwalltime)/60)

  println(log)

  plot_output(prob, fig, axs; drawcolorbar=false)
end

# how long did it take?
println((time()-startwalltime))

plot_output(prob, fig, axs; drawcolorbar=false)

# save the figure as png
savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename)
