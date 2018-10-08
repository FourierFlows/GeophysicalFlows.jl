using 
  GeophysicalFlows.NIWQG, 
  PyPlot,
  PyPlotPlus,
  Printf

using FourierFlows: lambdipole

# *** Parameters from Rocha, Wagner, and Young, JFM (2018) ***
 nx = 512
 Lx = 2π * 200e3
# Wave and dipole
 Uw = 5e-1
 Ue = 5e-2
 Re = 84e3
# Physical parameters
  f = 1e-4 
  N = 5e-3
  m = 2π/325
lam = N/(f*m)
eta = f*lam^2
# Dissipation
  nu = 1e7
 nnu = 2
 kap = 5e7
nkap = 2
# Timestepping
dt = 5e-2 * 2π/f
nsteps = 10
nsubsteps = 100
# Some calculated parameters
ke = 2π/Re
Ro = Ue*ke/f
amplitude = Ro*(Uw/Ue)^2
dispersivity = f*lam^2*ke/Ue

prob = NIWQG.Problem(nx=nx, Lx=Lx, dt=dt, f=f, eta=eta, nu=nu, nnu=nnu, kap=kap, nkap=nkap, Ub=-Ue,
                     stepper="FilteredETDRK4")
x, y = prob.grid.X, prob.grid.Y
xp, yp = x./Re, y./Re # for plotting

# Initial condition
q0 = lambdipole(Ue, Re, prob.grid)
phi0 = (1+im)/sqrt(2) * Uw * ones(nx, nx)

set_q!(prob, q0)
set_phi!(prob, phi0)

action0 = waveaction(prob)
ke0 = qgke(prob)
pe0 = wavepe(prob)
energy0 = coupledenergy(prob)

# Print a message
msg = """
Simulation of the interaction of the Lamb dipole with a 
uniform near-inertial wave.

Parameters:

       resolution = $nx
    Rossby number = $Ro
   wave amplitude = $amplitude
wave dispersivity = $dispersivity
"""
println(msg) #@printf(msg, nx, Ro, amplitude, dispersivity))

close("all")
fig, axs = subplots(ncols=2, figsize=(10, 6), sharey=true, sharex=true)
function makeplot!(axs, prob)
  sca(axs[1])
  pcolormesh(xp, yp, prob.vars.q)

  xlim(minimum(xp)/10, maximum(xp)/10)
  ylim(minimum(yp)/10, maximum(yp)/10)

  sca(axs[2])
  pcolormesh(xp, yp, abs.(prob.vars.phi))

  xlim(minimum(xp)/10, maximum(xp)/10)
  ylim(minimum(yp)/10, maximum(yp)/10)

  makesquare(axs)
  nothing
end

for i = 1:nsteps
  stepforward!(prob, nsubsteps)
  updatevars!(prob)

  println("step: $(i*nsubsteps), Δe: $(coupledenergy(prob)/energy0), Δe_w: $(wavepe(prob)/energy0)")
  makeplot!(axs, prob)
end
