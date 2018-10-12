using 
  GeophysicalFlows.NIWQG, 
  PyPlot,
  PyPlotPlus,
  Printf

using GeophysicalFlows: lambdipole

# *** Parameters from Rocha, Wagner, and Young, JFM (2018) ***
    nx = 512
    Lx = 2π * 200e3
# Wave and dipole
    Uw = 5e-1
    Ue = 5e-2
    Re = Lx/15 # = 83.8 km
    ke = 2π/Re
    te = 1/(ke*Ue) # eddy turnover time-scale
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
# Timestepping and jumping
    dt = 1e-2 * te
   nte = round(Int, te/dt)
njumps = 20
# Some calculated parameters
    Ro = Ue*ke/f
   amp = Ro*(Uw/Ue)^2
  disp = f*lam^2*ke/Ue

prob = NIWQG.Problem(nx=nx, Lx=Lx, dt=dt, f=f, eta=eta, nu=nu, nnu=nnu, kap=kap, nkap=nkap, 
                     Ub=-Ue, stepper="FilteredETDRK4")
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
   wave amplitude = $amp
wave dispersivity = $disp
"""
println(msg)

close("all")
fig, axs = subplots(ncols=2, figsize=(10, 6))
function makeplot!(axs, prob)
  sca(axs[1])
  cla()
  pcolormesh(xp, yp, prob.vars.q)

  fraction = 2.5
  xlim(minimum(xp)/fraction, maximum(xp)/fraction)
  ylim(minimum(yp)/fraction, maximum(yp)/fraction)

  sca(axs[2])
  cla()
  pcolormesh(xp, yp, abs.(prob.vars.phi))

  xlim(minimum(xp)/fraction, maximum(xp)/fraction)
  ylim(minimum(yp)/fraction, maximum(yp)/fraction)

  makesquare(axs)
  pause(0.01)
  nothing
end

for i = 1:njumps
  @time begin
    stepforward!(prob, nte)
    @printf("step: % 6d, t: %.1f, A: %.3f, P: %.3f, K+P: %.3f ", 
            prob.step, prob.t/te, waveaction(prob)/action0,
            wavepe(prob)/energy0, coupledenergy(prob)/energy0)
  end

  updatevars!(prob)
  makeplot!(axs, prob)
end
