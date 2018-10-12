using 
  GeophysicalFlows.NIWQG, 
  FFTW,
  PyPlot,
  PyPlotPlus,
  Printf

import GeophysicalFlows.TwoDTurb

# *** Parameters from Rocha, Wagner, and Young, JFM (2018) ***
     nx = 1024
     Lx = 2π * 200e3
# Waves and turbulence
     Ue = 5e-2            # magintude of inital eddy velocity
     ke = 2π / (Lx/10)    # energy containing length-scale
     te = 1/(ke*Ue)       # eddy turnover time-scale
     Uw = 1e-1            # wave field strength
# Physical parameters
      f = 1e-4
      N = 5e-3
      m = 2π/280
    lam = N/(f*m)
    eta = f*lam^2
# Dissipation
     nu = 5e6
    nnu = 2
    kap = 5e6
   nkap = 2
# Timestepping
     dt = 1e-2 * te
    nte = round(Int, te/dt)
  ninit = 20 
 ninter = 100
stepper = "ETDRK4"
# Some calculated parameters
     Ro = Ue*ke/f         # rossby number
    amp = Ro*(Uw/Ue)^2    # wave amplitude
   disp = f*lam^2*ke/Ue   # wave dispersivity

# Set up problem
initprob = TwoDTurb.Problem(nx=nx, Lx=Lx, dt=dt, nu=nu, nnu=nnu, stepper=stepper)
x, y = initprob.grid, initprob.grid.X, initprob.grid.Y

# 2D turbulence initial condition
nk = 32
kk = 2π/Lx*(1:nk)
k = Array(reshape(kk, (nk, 1)))
l = Array(reshape(kk, (1, nk)))
K = @. sqrt(k^2 + l^2)

psik = @. (K*(1 + K/ke)^4)^(-1/2)
psik *= 0.5*Ue^2 / sum(@.(K^2*psik^2)) # normalize

psi = zeros(nx, nx)
for (i, ki) in enumerate(k)
  for (j, li) in enumerate(l)
    @. psi += psik[i, j]*cos(ki*x + li*y + 2π*rand())
  end
end

psih = rfft(psi)
qh = @. -prob.grid.KKrsq*psih
q0 = irfft(qh, nx)
TwoDTurb.set_q!(initprob, q0)

close("all")
for i = 1:ninit
  stepforward!(initprob, nte)
  TwoDTurb.updatevars!(initprob)
  imshow(initprob.vars.q)
end

pause(1)
close("all")

#=
# Interruption problem!
prob = NIWQG.Problem(nx=nx, Lx=Lx, dt=dt, f=f, eta=eta, nu=nu, nnu=nnu, kap=kap, nkap=nkap, stepper=stepper)

set_q!(prob, initprob.vars.q)
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
  pcolormesh(xp, yp, prob.vars.q)

  fraction = 2.5
  xlim(minimum(xp)/fraction, maximum(xp)/fraction)
  ylim(minimum(yp)/fraction, maximum(yp)/fraction)

  sca(axs[2])
  pcolormesh(xp, yp, abs.(prob.vars.phi))

  xlim(minimum(xp)/fraction, maximum(xp)/fraction)
  ylim(minimum(yp)/fraction, maximum(yp)/fraction)

  makesquare(axs)
  pause(0.01)
  nothing
end

for i = 1:ninter
  @time begin
    stepforward!(prob, nte)
    @printf("step: % 6d, t: %.1f, A: %.3f, P: %.3f, K+P: %.3f ", prob.step, prob.t/te, waveaction(prob)/action0,
            wavepe(prob)/energy0, coupledenergy(prob)/energy0)
  end

  updatevars!(prob)
  makeplot!(axs, prob)
end
=#
