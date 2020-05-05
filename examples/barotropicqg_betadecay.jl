# # Decaying barotropic QG beta-plane turbulence
#
#md # This example can be run online via [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/barotropicqg_betadecay.ipynb). 
#md # Also, it can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/barotropicqg_betadecay.ipynb).
# 
# An example of decaying barotropic quasi-geostrophic turbulence on a beta plane.

using FourierFlows, PyPlot, Printf, Random

using Statistics: mean
using FFTW: irfft

import GeophysicalFlows.BarotropicQG
import GeophysicalFlows.BarotropicQG: energy, enstrophy


# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
nothing # hide


# ## Numerical parameters and time-stepping parameters

nx = 256       # 2D resolution = nx^2
stepper = "FilteredETDRK4"   # timestepper
dt = 0.02      # timestep 
nsteps = 8000  # total number of time-steps
nsubs  = 2000  # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

Lx = 2π        # domain size
 ν = 0e-05     # viscosity
nν = 1         # viscosity order
 β = 15.0      # planetary PV gradient
 μ = 0.0       # bottom drag
nothing # hide

# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments,
prob = BarotropicQG.Problem(nx=nx, Lx=Lx, β=β, ν=ν, nν=nν, μ=μ, dt=dt, stepper=stepper, dev=dev)
nothing # hide

# and define some shortcuts
sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = gridpoints(g)
nothing # hide


# ## Setting initial conditions

# Our initial condition consist of a flow that has power only at wavenumbers with
# $8<\frac{L}{2\pi}\sqrt{k_x^2+k_y^2}<10$ and initial energy $E_0$:

Random.seed!(1234)
E0 = 0.1
qih = randn(Complex{Float64}, size(sol))
qih[ g.Krsq .< (8*2π /g.Lx)^2] .= 0
qih[ g.Krsq .> (10*2π/g.Lx)^2] .= 0
Ein = energy(qih, g) # compute energy of qi
qih = qih*sqrt(E0/Ein) # normalize qi to have energy E0
qi  = irfft(qih, g.nx) 

BarotropicQG.set_zeta!(prob, qi)
nothing #hide

# Let's plot the initial vorticity field:

fig, axs = subplots(ncols=2, nrows=1, figsize=(10, 3.5), dpi=200)
sca(axs[1])
cla()
pcolormesh(x, y, v.q)
axis("square")
xticks(-2:2:2)
yticks(-2:2:2)
title(L"initial vorticity $\zeta = \partial_x v - \partial_y u$")
colorbar()
clim(-15, 15)
sca(axs[2])
cla()
contourf(x, y, v.psi)
colorbar()
clim(-0.2, 0.2)
contour(x, y, v.psi, colors="k")
axis("square")
xticks(-2:2:2)
yticks(-2:2:2)
title(L"initial streamfunction $\psi$")
gcf() # hide


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
get_u(prob) = irfft(im*g.l.*g.invKrsq.*sol, g.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
nothing # hide


# ## Visualizing the simulation

# We define a function that plots the vorticity and streamfunction fields and 
# their corresponding zonal mean structure.

function plot_output(prob, fig, axs; drawcolorbar=false)
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
  contourf(x, y, v.psi)
  axis("square")
  if drawcolorbar==true
    colorbar()
  end
  if maximum(abs.(v.psi))>0
    contour(x, y, v.psi, colors="k")
  end
  xticks(-2:2:2)
  yticks(-2:2:2)
  title(L"streamfunction $\psi$")

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
end
nothing # hide


# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

startwalltime = time()

while cl.step < nsteps
  stepforward!(prob, diags, nsubs)
  
  log = @sprintf("step: %04d, t: %d, E: %.4f, Q: %.4f, walltime: %.2f min",
    cl.step, cl.t, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)

  println(log)
end
println("finished")


# ## Plot
# Now let's see what we got. We plot the output,

fig, axs = subplots(ncols=2, nrows=2, figsize=(10, 7), dpi=200)
plot_output(prob, fig, axs; drawcolorbar=true)
gcf() # hide

# and finally save the figure
savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename)
