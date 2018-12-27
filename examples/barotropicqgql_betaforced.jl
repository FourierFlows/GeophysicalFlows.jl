using PyPlot,
  JLD2,
  Statistics,
  Printf,
  Random,
  FourierFlows

import GeophysicalFlows.BarotropicQGQL
import GeophysicalFlows.BarotropicQGQL: energy, enstrophy

import FFTW: irfft
import Statistics: mean

# Numerical parameters and time-stepping parameters
nx = 256       # 2D resolution = nx^2
stepper = "FilteredRK4"   # timestepper
dt  = 0.01     # timestep
nsteps = 20000 # total number of time-steps
nsubs  = 500   # number of time-steps for plotting
               # (nsteps must be multiple of nsubs)

# Physical parameters
Lx  = 2π       # domain size
nu  = 0e-05    # viscosity
nnu = 1        # viscosity order
beta = 15.0    # planetary PV gradient
mu   = 0.02    # bottom drag


# Forcing
kf, dkf = 12.0, 2.0     # forcing wavenumber and width of
                        # forcing ring in wavenumber space
σ = 0.005               # energy input rate by the forcing
gr  = TwoDGrid(nx, Lx)
force2k = @. exp(-(sqrt(gr.Krsq)-kf)^2/(2*dkf^2))
force2k[gr.Krsq .< 2.0^2 ] .= 0
force2k[gr.Krsq .> 20.0^2 ] .= 0
force2k[gr.Kr.<2π/Lx] .= 0
σ0 = FourierFlows.parsevalsum(force2k.*gr.invKrsq/2.0, gr)/(gr.Lx*gr.Ly)
force2k .= σ/σ0 * force2k  # normalization so that forcing injects
                           # energy ε per domain area per unit time

# reset of the random number generator for reproducibility
Random.seed!(1234)

# the function that updates the forcing realization
function calcF!(Fh, t, s, v, p, g)
  ξ = exp.(2π*im*rand(Float64, size(s.sol)))/sqrt(cl.dt)
  ξ[1, 1] = 0
  @. Fh = ξ*sqrt(force2k)
  Fh[abs.(g.Kr).==0] .= 0
  nothing
end

# Initialize problem
prob = BarotropicQGQL.ForcedProblem(nx=nx, Lx=Lx, beta=beta, nu=nu, nnu=nnu,
                                  mu=mu, dt=dt, stepper=stepper, calcF=calcF!)
s, v, p, g, eq, ts = prob.state, prob.vars, prob.params, prob.grid, prob.eqn, prob.ts


# Files
filepath = "."
filename = joinpath(filepath, "forcedbetaturbQL.jld2")

# File management
if isfile(filename); rm(filename); end

# Zero initial condition
BarotropicQGQL.set_zeta!(prob, zeros(g.X))

function zetaMean(prob)
    sol = prob.state.sol
    sol[1, :]
end

# Create Diagnostic -- "energy" and "enstrophy" are functions imported at the top.
E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
zMean = Diagnostic(zetaMean, prob; nsteps=nsteps, freq=10)
diags = [E, Z, zMean] # A list of Diagnostics types passed to "stepforward!" will
# be updated every timestep. They should be efficient to calculate and
# have a small memory footprint. (For example, the domain-integrated kinetic
# energy is just a single number for each timestep).

# Create Output
get_sol(prob) = prob.vars.sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*g.Lr.*g.invKrsq.*prob.vars.sol, g.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))


function plot_output(prob, fig, axs; drawcolorbar=false)
  # Plot the vorticity and streamfunction fields as well as the zonal mean
  # vorticity and the zonal mean zonal velocity.

  s, v, p, g = prob.state, prob.vars, prob.params, prob.grid
  BarotropicQGQL.updatevars!(prob)

  sca(axs[1])
  cla()
  pcolormesh(g.X, g.Y, v.zeta .+ v.Zeta)
  axis("square")
  xticks(-3:1:3)
  yticks(-3:1:3)
  title(L"vorticity $\zeta = \partial_x v - \partial_y u$")
  if drawcolorbar==true
    colorbar()
  end

  sca(axs[2])
  cla()
  pcolormesh(g.X, g.Y, v.psi)
  axis("square")
  xticks(-3:1:3)
  yticks(-3:1:3)
  title(L"streamfunction $\psi$")
  if drawcolorbar==true
    colorbar()
  end

  sca(axs[3])
  cla()
  plot(transpose(mean(v.Zeta, dims=1)), g.Y[1,:])
  plot(0*g.Y[1,:], g.Y[1,:], "k--")
  yticks(-3:1:3)
  ylim(-Lx/2, Lx/2)
  xlim(-4, 4)
  title(L"zonal mean $\zeta$")

  sca(axs[4])
  cla()
  plot(mean(transpose(v.U), dims=2), g.Y[1,:])
  plot(0*g.Y[1,:], g.Y[1,:], "k--")
  yticks(-3:1:3)
  ylim(-Lx/2, Lx/2)
  xlim(-0.7, 0.7)
  title(L"zonal mean $u$")

  sca(axs[5])
  cla()
  plot(mu*E.time[1:E.prob.step], E.data[1:prob.step], label="energy")
  xlabel(L"\mu t")
  legend()

  sca(axs[6])
  cla()
  plot(mu*Z.time[1:E.prob.step], Z.data[1:prob.step], label="enstrophy")
  xlabel(L"\mu t")
  legend()

  pause(0.001)
end

fig, axs = subplots(ncols=3, nrows=2, figsize=(14, 8))
plot_output(prob, fig, axs; drawcolorbar=false)

# Step forward
startwalltime = time()

while prob.step < nsteps
  stepforward!(prob, diags, nsubs)

  BarotropicQGQL.updatevars!(prob)

  # Message
  cfl = prob.tcl.dt*maximum([maximum(v.v)/g.dy, maximum(v.u+v.U)/g.dx])
  log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, τ: %.2f min",
    prob.step, prob.t, cfl, E.value, Z.value,
    (time()-startwalltime)/60)

  println(log)

  plot_output(prob, fig, axs; drawcolorbar=false)
end

# how long did it take?
println((time()-startwalltime))

plot_output(prob, fig, axs; drawcolorbar=false)


UM = zeros(g.ny, length(zMean.time))
for in = 1:length(zMean.time)
    UM[:, in] = real(ifft(im*g.Lr[1, :].*zMean[in].*g.invKrsq[1, :]))
end
figure(2); pcolormesh(zMean.time, g.y.', UM)


# save the figure as png
# savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), prob.step)
# savefig(savename)
