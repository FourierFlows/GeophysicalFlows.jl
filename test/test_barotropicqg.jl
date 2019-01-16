"""
    test_bqg_rossbywave(; kwargs...)

Evolves a Rossby wave and compares with the analytic solution.
"""
function test_bqg_rossbywave(stepper, dt, nsteps)
    nx = 64
  beta = 2.0
    Lx = 2π
    mu = 0.0
    nu = 0.0

  gr  = TwoDGrid(nx, Lx)
  x, y = gridpoints(gr)

  # the following if statement is called so that all the cases of
  # Problem() fuction are tested
  if stepper=="ForwardEuler"
    eta = zeros(nx, nx)
  else
    eta(x, y) = 0*x
  end

  prob = BarotropicQG.InitialValueProblem(nx=nx, Lx=Lx, eta=eta, beta=beta, mu=mu, nu=nu, stepper=stepper, dt=dt)
  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  x, y = gridpoints(g)

  # the Rossby wave initial condition
   ampl = 1e-2
  kwave = 3.0*2π/g.Lx
  lwave = 2.0*2π/g.Ly
      ω = -p.beta*kwave/(kwave^2.0 + lwave^2.0)
     ζ0 = @. ampl*cos(kwave*x)*cos(lwave*y)
    ζ0h = rfft(ζ0)

  BarotropicQG.set_zeta!(prob, ζ0)

  stepforward!(prob, nsteps)
  dealias!(sol, g)
  BarotropicQG.updatevars!(prob)

  ζ_theory = @. ampl*cos(kwave*(x - ω/kwave*cl.t)) * cos(lwave*y)

  isapprox(ζ_theory, v.zeta, rtol=g.nx*g.ny*nsteps*1e-12)
end

"""
    test_stochasticforcingbudgets(; kwargs...)

Tests if the energy budgets are closed for BarotropicQG with stochastic forcing.
"""
function test_bqg_stochasticforcingbudgets(; n=256, dt=0.01, L=2π, nu=1e-7, nnu=2, mu=1e-1, message=false)
  n, L  = 256, 2π
  nu, nnu = 1e-7, 2
  mu = 1e-1
  dt, tf = 0.005, 0.1/mu
  nt = round(Int, tf/dt)
  ns = 1

  # Forcing
  kf, dkf = 12.0, 2.0
  ε = 0.1

  gr  = TwoDGrid(n, L)
  x, y = gridpoints(gr)

  Kr = [ gr.kr[i] for i=1:gr.nkr, j=1:gr.nl]

  force2k = zero(gr.Krsq)
  @. force2k = exp.(-(sqrt(gr.Krsq)-kf)^2/(2*dkf^2))
  @. force2k[gr.Krsq .< 2.0^2 ] = 0
  @. force2k[gr.Krsq .> 20.0^2 ] = 0
  @. force2k[Kr .< 2π/L] = 0
  ε0 = parsevalsum(force2k.*gr.invKrsq/2.0, gr)/(gr.Lx*gr.Ly)
  force2k .= ε/ε0 * force2k

  Random.seed!(1234)

  function calcFq!(Fqh, sol, t, cl, v, p, g)
    eta = exp.(2π*im*rand(Float64, size(sol)))/sqrt(cl.dt)
    eta[1, 1] = 0
    @. Fqh = eta * sqrt(force2k)
    nothing
  end

  prob = BarotropicQG.ForcedProblem(nx=n, Lx=L, nu=nu, nnu=nnu, mu=mu, dt=dt,
   stepper="RK4", calcFq=calcFq!, stochastic=true)

  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  BarotropicQG.set_zeta!(prob, 0*x)
  E = Diagnostic(BarotropicQG.energy,      prob, nsteps=nt)
  D = Diagnostic(BarotropicQG.dissipation, prob, nsteps=nt)
  R = Diagnostic(BarotropicQG.drag,        prob, nsteps=nt)
  W = Diagnostic(BarotropicQG.work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  # Step forward

  stepforward!(prob, diags, round(Int, nt))

  BarotropicQG.updatevars!(prob)

  cfl = cl.dt*maximum([maximum(v.v)/g.dx, maximum(v.u)/g.dy])

  E, D, W, R = diags

  t = round(mu*cl.t, digits=2)

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

  if message
    println("step: %04d, t: %.1f, cfl: %.3f, time: %.2f s\n", cl.step, cl.t, cfl, t)
  end
  # println(mean(abs.(residual)))
  isapprox(mean(abs.(residual)), 0, atol=1e-4)
end

"""
    test_stochasticforcingbudgets(; kwargs...)

Tests if the energy budgets are closed for BarotropicQG with stochastic forcing.
"""
function test_bqg_deterministicforcingbudgets(; n=256, dt=0.01, L=2π, nu=1e-7, nnu=2, mu=1e-1, message=false)
  n, L  = 256, 2π
  nu, nnu = 1e-7, 2
  mu = 1e-1
  dt, tf = 0.005, 0.1/mu
  nt = round(Int, tf/dt)
  ns = 1


  # Forcing = 0.01cos(4x)cos(5y)cos(2t)
  gr  = TwoDGrid(n, L)
  x, y = gridpoints(gr)
  k0, l0 = gr.kr[2], gr.l[2]

  f = @. 0.01*cos(4*k0*x)*cos(5*l0*y)
  fh = rfft(f)
  function calcFq!(Fqh, sol, t, cl, v, p, g)
    @. Fqh = fh*cos(2*t)
    nothing
  end

  prob = BarotropicQG.ForcedProblem(nx=n, Lx=L, nu=nu, nnu=nnu, mu=mu, dt=dt,
   stepper="RK4", calcFq=calcFq!, stochastic=false)

  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  BarotropicQG.set_zeta!(prob, 0*x)
  E = Diagnostic(BarotropicQG.energy,      prob, nsteps=nt)
  D = Diagnostic(BarotropicQG.dissipation, prob, nsteps=nt)
  R = Diagnostic(BarotropicQG.drag,        prob, nsteps=nt)
  W = Diagnostic(BarotropicQG.work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  # Step forward

  stepforward!(prob, diags, round(Int, nt))

  BarotropicQG.updatevars!(prob)

  cfl = cl.dt*maximum([maximum(v.v)/g.dx, maximum(v.u)/g.dy])

  E, D, W, R = diags

  t = round(mu*cl.t, digits=2)

  i₀ = 1
  dEdt = (E[(i₀+1):E.i] - E[i₀:E.i-1])/cl.dt
  ii = (i₀):E.i-1
  ii2 = (i₀+1):E.i

  # dEdt = W - D - R?
  total = W[ii2] - D[ii] - R[ii]

  residual = dEdt - total

  if message
    println("step: %04d, t: %.1f, cfl: %.3f, time: %.2f s\n", prob.step, cl.t, cfl, tc)
  end
  # println(mean(abs.(residual)))
  isapprox(mean(abs.(residual)), 0, atol=1e-8)
end

"""
    test_bqg_nonlinearadvection(dt, stepper; kwargs...)

Tests the advection term in the twodturb module by timestepping a
test problem with timestep dt and timestepper identified by the string stepper.
The test problem is derived by picking a solution ζf (with associated
streamfunction ψf) for which the advection term J(ψf, ζf) is non-zero. Next, a
forcing Ff is derived according to Ff = ∂ζf/∂t + J(ψf, ζf) - nuΔζf. One solution
to the vorticity equation forced by this Ff is then ζf. (This solution may not
be realized, at least at long times, if it is unstable.)
"""
function test_bqg_advection(dt, stepper; n=128, L=2π, nu=1e-2, nnu=1, mu=0.0, message=false)
  n, L  = 128, 2π
  nu, nnu = 1e-2, 1
  mu = 0.0
  tf = 1.0
  nt = round(Int, tf/dt)

  gr  = TwoDGrid(n, L)
  x, y = gridpoints(gr)

  psif = @. sin(2x)*cos(2y) + 2sin(x)*cos(3y)
  qf = @. -8sin(2x)*cos(2y) - 20sin(x)*cos(3y)

  Ff = @. -(
    nu*( 64sin(2x)*cos(2y) + 200sin(x)*cos(3y) )
    + 8*( cos(x)*cos(3y)*sin(2x)*sin(2y) - 3cos(2x)*cos(2y)*sin(x)*sin(3y) )
  )

  Ffh = rfft(Ff)

  # Forcing
  function calcFq!(Fqh, sol, t, cl, v, p, g)
    Fqh .= Ffh
    nothing
  end

  prob = BarotropicQG.ForcedProblem(nx=n, Lx=L, nu=nu, nnu=nnu, mu=mu, dt=dt, stepper=stepper, calcFq=calcFq!)
  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
  BarotropicQG.set_zeta!(prob, qf)

  # Step forward
  stepforward!(prob, round(Int, nt))
  BarotropicQG.updatevars!(prob)
  isapprox(v.q, qf, rtol=1e-13)
end

"""
    test_bqg_formstress(dt, stepper; kwargs...)

Tests the form stress term that forces the domain-averaged zonal flow U(t).
"""
function test_bqg_formstress(dt, stepper; n=128, L=2π, nu=0.0, nnu=1, mu=0.0, message=false)
  n, L  = 128, 2π
  nu, nnu = 1e-2, 1
  mu = 0.0
  tf = 1
  nt = 1

  gr  = TwoDGrid(n, L)
  x, y = gridpoints(gr)

  zetai = @. -20*sin(10*x)*cos(10*y)
  topoPV(x, y) = @. cos(10x)*cos(10y)
  F(t) = 0 #no forcing

  answer = 0.25 # this is what <v*eta> should be

  prob = BarotropicQG.ForcedProblem(nx=n, Lx=L, nu=nu, nnu=nnu, mu=mu, dt=dt, stepper=stepper, eta=topoPV, calcFU = F)
  BarotropicQG.set_zeta!(prob, zetai)
  BarotropicQG.updatevars!(prob)

  # Step forward
  stepforward!(prob, nt)
  isapprox(prob.timestepper.N[1, 1], answer, rtol=1e-13)
end

function test_bqg_energyenstrophy()
  nx, Lx  = 64, 2π
  ny, Ly  = 64, 3π
  g  = TwoDGrid(nx, Lx, ny, Ly)
  k0, l0 = g.k[2], g.l[2] # fundamental wavenumbers
  x, y = gridpoints(g)

    eta = @. cos(10k0*x)*cos(10l0*y)
   psi0 = @. sin(2k0*x)*cos(2l0*y) + 2sin(k0*x)*cos(3l0*y)
  zeta0 = @. -((2k0)^2+(2l0)^2)*sin(2k0*x)*cos(2l0*y) - (k0^2+(3l0)^2)*2sin(k0*x)*cos(3l0*y)

  prob = BarotropicQG.InitialValueProblem(nx=nx, Lx=Lx, ny=ny, Ly=Ly, eta = eta, stepper="ForwardEuler")
  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
  BarotropicQG.set_zeta!(prob, zeta0)
  BarotropicQG.updatevars!(prob)

  energyzeta0 = BarotropicQG.energy(prob)
  enstrophyzeta0 = BarotropicQG.enstrophy(prob)

  isapprox(energyzeta0, 29.0/9, rtol=1e-13) && isapprox(enstrophyzeta0, 2701.0/162, rtol=1e-13)
end

function test_bqg_meanenergyenstrophy()
  nx, Lx  = 64, 2π
  ny, Ly  = 96, 3π
  g  = TwoDGrid(nx, Lx, ny, Ly)
  k0, l0 = g.k[2], g.l[2] # fundamental wavenumbers
  x, y = gridpoints(g)

  calcFU(t) = 0.0
  eta(x, y) = @. cos(10x)*cos(10y)
  psi0 = @. sin(2k0*x)*cos(2l0*y) + 2sin(k0*x)*cos(3l0*y)
 zeta0 = @. -((2k0)^2+(2l0)^2)*sin(2k0*x)*cos(2l0*y) - (k0^2+(3l0)^2)*2sin(k0*x)*cos(3l0*y)
  beta = 10.0
  U = 1.2

  prob = BarotropicQG.ForcedProblem(nx=nx, Lx=Lx, ny=ny, Ly=Ly, beta=beta, eta=eta, calcFU = calcFU,
                                    stepper="ForwardEuler")

  BarotropicQG.set_zeta!(prob, zeta0)
  BarotropicQG.set_U!(prob, U)
  BarotropicQG.updatevars!(prob)

  energyU = BarotropicQG.meanenergy(prob)
  enstrophyU = BarotropicQG.meanenstrophy(prob)

  energyzeta0 = BarotropicQG.energy(prob)
  enstrophyzeta0 = BarotropicQG.enstrophy(prob)

  (isapprox(energyU, 0.5*U^2, rtol=1e-13) &&
    isapprox(enstrophyU, beta*U, rtol=1e-13) &&
    isapprox(energyzeta0, 29.0/9, rtol=1e-13) &&
    isapprox(enstrophyzeta0, 2701.0/162, rtol=1e-13)
  )
end
