using PyPlot, FourierFlows

using Random: seed!
using Printf: @printf

import GeophysicalFlows.TwoDTurb
import GeophysicalFlows.TwoDTurb: energy, enstrophy, dissipation, work, drag

  n, L  = 256, 2π
nu, nnu = 1e-7, 2
mu, nmu = 1e-1, 0
dt, tf = 0.005, 0.2/mu
nt = round(Int, tf/dt)
ns = 4

# Forcing
kf, dkf = 12.0, 2.0     # forcing central wavenumber, wavenumber width
ε = 0.1                 # energy injection rate

gr  = TwoDGrid(n, L)
x, y = gridpoints(gr)

Kr = [ gr.kr[i] for i=1:gr.nkr, j=1:gr.nl]

force2k = @. exp(-(sqrt(gr.Krsq)-kf)^2/(2*dkf^2))
force2k[gr.Krsq .< 2.0^2 ] .= 0
force2k[gr.Krsq .> 20.0^2 ] .= 0
force2k[Kr .< 2π/L] .= 0
ε0 = FourierFlows.parsevalsum(force2k.*gr.invKrsq/2.0, gr)/(gr.Lx*gr.Ly)
force2k .= ε/ε0 * force2k

seed!(1234)

function calcF!(Fh, sol, t, cl, v, p, g)
  eta = exp.(2π*im*rand(Float64, size(sol)))/sqrt(cl.dt)
  eta[1, 1] = 0
  @. Fh = eta*sqrt(force2k)
  nothing
end

prob = TwoDTurb.Problem(nx=n, Lx=L, nu=nu, nnu=nnu, mu=mu, nmu=nmu, dt=dt, stepper="RK4",
                        calcF=calcF!, stochastic=true)

sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

TwoDTurb.set_zeta!(prob, 0*x)
E = Diagnostic(energy,      prob, nsteps=nt)
D = Diagnostic(dissipation, prob, nsteps=nt)
R = Diagnostic(drag,        prob, nsteps=nt)
W = Diagnostic(work,        prob, nsteps=nt)
diags = [E, D, W, R]


function makeplot(prob, diags)
  TwoDTurb.updatevars!(prob)
  E, D, W, R = diags

  t = round(mu*cl.t, digits=2)
  sca(axs[1]); cla()
  pcolormesh(x, y, v.zeta)
  xlabel(L"$x$")
  ylabel(L"$y$")
  title("\$\\nabla^2\\psi(x,y,\\mu t= $t )\$")
  axis("square")

  sca(axs[3]); cla()

  i₀ = 1
  dEdt = (E[(i₀+1):E.i] - E[i₀:E.i-1])/cl.dt
  ii = (i₀):E.i-1
  ii2 = (i₀+1):E.i

  # dEdt = W - D - R?

  # If the Ito interpretation was used for the work
  # then we need to add the drift term
  # total = W[ii2]+ε - D[ii] - R[ii]      # Ito
  total = W[ii2] - D[ii] - R[ii]        # Stratonovich
  residual = dEdt - total

  # If the Ito interpretation was used for the work
  # then we need to add the drift term: I[ii2] + ε
  plot(mu*E.t[ii], W[ii2], label=L"work ($W$)")   # Ito
  # plot(mu*E.t[ii], W[ii2] , label=L"work ($W$)")      # Stratonovich
  plot(mu*E.t[ii], ε .+ 0*E.t[ii], "--", label=L"ensemble mean  work ($\langle W\rangle $)")
  # plot(mu*E.t[ii], -D[ii], label="dissipation (\$D\$)")
  plot(mu*E.t[ii], -R[ii], label=L"drag ($D=2\mu E$)")
  plot(mu*E.t[ii], 0*E.t[ii], "k:", linewidth=0.5)
  ylabel("Energy sources and sinks")
  xlabel(L"$\mu t$")
  legend(fontsize=10)

  sca(axs[2]); cla()
  plot(mu*E.t[ii], total[ii], label=L"computed $W-D$")
  plot(mu*E.t[ii], dEdt, "--k", label=L"numerical $dE/dt$")
  ylabel(L"$dE/dt$")
  xlabel(L"$\mu t$")
  legend(fontsize=10)

  sca(axs[4]); cla()
  plot(mu*E.t[ii], residual, "c-", label=L"residual $dE/dt$ = computed $-$ numerical")
  xlabel(L"$\mu t$")
  legend(fontsize=10)

  residual
end

fig, axs = subplots(ncols=2, nrows=2, figsize=(12, 8))


# Step forward
startwalltime = time()
for i = 1:ns
  stepforward!(prob, diags, round(Int, nt/ns))

  TwoDTurb.updatevars!(prob)
  # saveoutput(out)

  cfl = cl.dt*maximum([maximum(v.u)/g.dx, maximum(v.v)/g.dy])
  res = makeplot(prob, diags)
  pause(0.01)

  @printf("step: %04d, t: %.1f, cfl: %.3f, time: %.2f s\n", cl.step, cl.t,
        cfl, (time()-startwalltime)/60)

  # savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
  # savefig(savename, dpi=240)
end

# savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
# savefig(savename, dpi=240)

# savediagnostic(E, "energy", out.filename)
# savediagnostic(D, "dissipation", out.filename)
# savediagnostic(W, "work", out.filename)
# savediagnostic(R, "drag", out.filename)
