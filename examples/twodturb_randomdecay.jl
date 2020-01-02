using
  PyPlot,
  Printf,
  FourierFlows

using FFTW: rfft

import GeophysicalFlows.TwoDTurb

 dev = CPU()   # Device (CPU/GPU)

  nx = 256     # Resolution
  Lx = 2π      # Domain size
   ν = 1e-6    # Viscosity
  nν = 1       # Order of (hyper-)viscosity. nν=1 means Laplacian
  dt = 0.1     # Timestep
nint = 200     # Number of steps between plots
ntot = 10nint  # Number of total timesteps
 
# Define problem
prob = TwoDTurb.Problem(nx=nx, Lx=Lx, ν=ν, nν=nν, dt=dt, stepper="FilteredRK4", dev=dev)
TwoDTurb.set_zeta!(prob, ArrayType(dev)(rand(nx, nx)))

cl, vs, gr = prob.clock, prob.vars, prob.grid
x, y = gridpoints(gr)

"Plot the vorticity of the twodturb problem `prob`."
function makeplot!(ax, prob)
  sca(ax)
  cla()
  pcolormesh(x, y, vs.zeta)
  title("Vorticity")
  xlabel(L"x")
  ylabel(L"y")
  pause(0.01)
  nothing
end

# Step forward
fig1, ax = subplots(figsize=(8, 8))

while cl.step < ntot
  @time begin
    stepforward!(prob, nint)
    @printf("step: %04d, t: %6.1f", cl.step, cl.t)
  end

  TwoDTurb.updatevars!(prob)
  makeplot!(ax, prob)
end

# Plot the radial energy spectrum
E  = @. 0.5*(vs.u^2 + vs.v^2) # energy density
Eh = rfft(E)
kr, Ehr = FourierFlows.radialspectrum(Eh, gr, refinement=1)

fig2, axs = subplots(ncols=2, figsize=(8, 4))

sca(axs[1])
pcolormesh(x, y, vs.zeta)
xlabel(L"x")
ylabel(L"y")
title("Vorticity")

sca(axs[2])
plot(kr, abs.(Ehr))
xlabel(L"k_r")
ylabel(L"\int | \hat{E} | k_r \mathrm{d} k_{\theta}")
title("Radial energy spectrum")

xlim(0, nx/4)
axs[2].set_yscale("log")

tight_layout(w_pad=0.1)
show()
