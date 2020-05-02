#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/examples/barotropicqg_betaforced.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/examples/barotropicqg_betaforced.ipynb)

# # Forced-dissipative barotropic QG beta-plane turbulence
#
# A simulation of forced-dissipative barotropic quasi-geostrophic turbulence on 
# a beta plane. The dynamics include linear drag and stochastic excitation.

using FourierFlows, PyPlot, Statistics, Printf, Random

using FFTW: irfft
using Statistics: mean
import Random: seed!

import GeophysicalFlows.BarotropicQG
import GeophysicalFlows.BarotropicQG: energy, enstrophy


# ## Choosing a device: CPU or GPU

dev = CPU()    # Device (CPU/GPU)
#md nothing # hide


# ## Numerical parameters and time-stepping parameters

nx = 128       # 2D resolution = nx^2
stepper = "FilteredRK4"   # timestepper
dt  = 0.05     # timestep
nsteps = 8000  # total number of time-steps
nsubs  = 2000  # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
#md nothing # hide


# ## Physical parameters

Lx = 2π        # domain size
 ν = 0.0       # viscosity
nν = 1         # viscosity order 
 β = 10.0      # planetary PV gradient
 μ = 0.01      # bottom drag
#md nothing # hide


# ## Forcing
#
# We force the vorticity equation with stochastic excitation that is delta-correlated
# in time and while spatially homogeneously and isotropically correlated. The forcing
# has a spectrum with power in a ring in wavenumber space of radious $k_f$ and 
# width $\delta k_f$, and it injects energy per unit area and per unit time equal 
# to $\varepsilon$.

kf, dkf = 14.0, 1.5     # forcing wavenumber and width of forcing ring in wavenumber space
ε = 0.001               # energy input rate by the forcing

gr  = TwoDGrid(nx, Lx)

x, y = gridpoints(gr)
Kr = [ gr.kr[i] for i=1:gr.nkr, j=1:gr.nl]

forcingcovariancespectrum = @. exp(-(sqrt(gr.Krsq)-kf)^2/(2*dkf^2))
@. forcingcovariancespectrum[gr.Krsq < 2.0^2 ] .= 0
@. forcingcovariancespectrum[gr.Krsq > 20.0^2 ] .= 0
forcingcovariancespectrum[Kr .< 2π/Lx] .= 0
ε0 = FourierFlows.parsevalsum(forcingcovariancespectrum.*gr.invKrsq/2.0, gr)/(gr.Lx*gr.Ly)
forcingcovariancespectrum .= ε/ε0 * forcingcovariancespectrum  # normalization so that forcing injects energy ε per domain area per unit time

seed!(1234) # reset of the random number generator for reproducibility
#md nothing # hide

# Next we construct function `calcF!` that computes a forcing realization every timestep
function calcFq!(Fh, sol, t, cl, v, p, g)
  ξ = ArrayType(dev)(exp.(2π*im*rand(Float64, size(sol)))/sqrt(cl.dt))
  ξ[1, 1] = 0
  @. Fh = ξ*sqrt(forcingcovariancespectrum)
  Fh[abs.(Kr).==0] .= 0
  nothing
end
#md nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments,
prob = BarotropicQG.Problem(nx=nx, Lx=Lx, β=β, ν=ν, nν=nν, μ=μ, dt=dt,
                            stepper=stepper, calcFq=calcFq!, stochastic=true, dev=dev)
#md nothing # hide

# and define some shortcuts.
sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
#md nothing # hide


# ## Setting initial conditions

# Our initial condition is simply fluid at rest.
BarotropicQG.set_zeta!(prob, 0*x)


# ## Diagnostics

# Create Diagnostic -- "energy" and "enstrophy" are functions imported at the top.
E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
#md nothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_forcedbetaturb"
plotname = "snapshots"
filename = joinpath(filepath, "forcedbetaturb.jld2")
#md nothing # hide

# Do some basic file management,
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
#md nothing # hide

# and then create Output.
get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*g.l.*g.invKrsq.*sol, g.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
#md nothing # hide


# ## Visualizing the simulation

# We define a function that plots the vorticity and streamfunction fields, their 
# corresponding zonal mean structure and timeseries of energy and enstrophy.

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
  if maximum(abs.(v.psi))>0
    contour(x, y, v.psi, colors="k")
  end
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
  xlim(-3, 3)
  title(L"zonal mean $\zeta$")

  sca(axs[4])
  cla()
  plot(Array(transpose(mean(v.u, dims=1))), y[1,:])
  plot(0*y[1,:], y[1,:], "k--")
  ylim(-Lx/2, Lx/2)
  xlim(-0.5, 0.5)
  title(L"zonal mean $u$")

  sca(axs[5])
  cla()
  plot(μ*E.t[1:E.i], E.data[1:E.i], label="energy")
  xlabel(L"\mu t")
  legend()

  sca(axs[6])
  cla()
  plot(μ*Z.t[1:Z.i], Z.data[1:E.i], label="enstrophy")
  xlabel(L"\mu t")
  legend()
end
#md nothing # hide


# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

startwalltime = time()

while cl.step < nsteps
  stepforward!(prob, diags, nsubs)

  BarotropicQG.updatevars!(prob)
  cfl = cl.dt*maximum([maximum(v.u)/g.dx, maximum(v.v)/g.dy])

  log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
    cl.step, cl.t, cfl, E.data[E.i], Z.data[Z.i], 
    (time()-startwalltime)/60)
  println(log)
end
println("finished")


# ## Plot
# Now let's see what we got. We plot the output,

fig, axs = subplots(ncols=3, nrows=2, figsize=(14, 8), dpi=200)
plot_output(prob, fig, axs; drawcolorbar=false)
gcf() # hide

# and finally save the figure.
savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename)
