using
  PyPlot,
  Printf,
  FourierFlows

using FFTW: rfft

import GeophysicalFlows.TwoDTurb


nx = 256        # Resolution
Lx = 2π         # Domain size
nu = 1e-6       # Viscosity
nnu = 1         # Order of (hyper-)viscosity. nnu=1 means Laplacian
dt = 1.0        # Timestep
nint = 100      # Number of steps between plots
ntot = 10nint   # Number of total timesteps

# Define problem
prob = TwoDTurb.Problem(nx=nx, Lx=Lx, nu=nu, nnu=nnu, dt=dt, stepper="FilteredRK4")
TwoDTurb.set_q!(prob, rand(nx, nx))

"Plot the vorticity of the twodturb problem `prob`."
function makeplot!(ax, prob)
  sca(ax)
  cla()
  pcolormesh(prob.grid.X, prob.grid.Y, prob.vars.q)
  title("Vorticity")
  xlabel(L"x")
  ylabel(L"y")
  pause(0.01)
  nothing
end

# Step forward
fig1, ax = subplots(figsize=(8, 8))

while prob.step < ntot
  @time begin
    stepforward!(prob, nint)
    @printf("step: %04d, t: %6.1f", prob.step, prob.t)
  end

  TwoDTurb.updatevars!(prob)
  makeplot!(ax, prob)
end

# Plot the radial energy spectrum
E = @. 0.5*(prob.vars.U^2 + prob.vars.V^2) # energy density
Eh = rfft(E)
kr, Ehr = FourierFlows.radialspectrum(Eh, prob.grid, refinement=1)

fig2, axs = subplots(ncols=2, figsize=(8, 4))

sca(axs[1])
pcolormesh(prob.grid.X, prob.grid.Y, prob.vars.q)
xlabel(L"x")
ylabel(L"y")
title("Vorticity")

sca(axs[2])
plot(kr, abs.(Ehr))
xlabel(L"k_r")
ylabel(L"\int | \hat{E} | k_r \mathrm{d} k_{\theta}")
title("Radial energy spectrum")

xlim(0, nx/4)
axs[2][:set_yscale]("log")

tight_layout(w_pad=0.1)
show()
