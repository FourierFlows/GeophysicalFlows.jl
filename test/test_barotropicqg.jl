"""
    test_bqg_rossbywave(; kwargs...)

Evolves a Rossby wave and compares with the analytic solution.
"""
function test_bqg_rossbywave(stepper, dt, nsteps, dev::Device=CPU())
    nx = 64
    Lx = 2π
     β = 2.0
     μ = 0.0
     ν = 0.0
     T = Float64

  # the following if statement is called so that all the cases of
  # Problem() fuction are tested
  if stepper=="ForwardEuler"
    eta = zeros(dev, T, (nx, nx))
  else
    eta(x, y) = 0*x
  end

  prob = BarotropicQG.Problem(nx=nx, Lx=Lx, eta=eta, β=β, μ=μ, ν=ν, stepper=stepper, dt=dt, dev=dev)
  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  x, y = gridpoints(g)

  # the Rossby wave initial condition
   ampl = 1e-2
  kwave = 3.0*2π/g.Lx
  lwave = 2.0*2π/g.Ly
      ω = -p.β*kwave/(kwave^2.0 + lwave^2.0)
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
function test_bqg_stochasticforcingbudgets(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1, T=Float64)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ = 1e-1
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)
   
  # Forcing
  kf, dkf = 12.0, 2.0
  ε = 0.1

  gr = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

  Kr = ArrayType(dev)([ gr.kr[i] for i=1:gr.nkr, j=1:gr.nl])

  force2k = zeros(dev, T, (gr.nkr, gr.nl))
  @. force2k = exp.(-(sqrt(gr.Krsq)-kf)^2/(2*dkf^2))
  @. force2k[gr.Krsq .< 2.0^2 ] = 0
  @. force2k[gr.Krsq .> 20.0^2 ] = 0
  @. force2k[Kr .< 2π/L] = 0
  ε0 = parsevalsum(force2k.*gr.invKrsq/2.0, gr)/(gr.Lx*gr.Ly)
  force2k .= ε/ε0 * force2k

  Random.seed!(1234)

  function calcFq!(Fqh, sol, t, cl, v, p, g)
    eta = ArrayType(dev)(exp.(2π*im*rand(T, size(sol)))/sqrt(cl.dt))
    eta[1, 1] = 0
    @. Fqh = eta * sqrt(force2k)
    nothing
  end

  prob = BarotropicQG.Problem(nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt,
   stepper="RK4", calcFq=calcFq!, stochastic=true, dev=dev)

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

  E, D, W, R = diags

  t = round(μ*cl.t, digits=2)

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

  isapprox(mean(abs.(residual)), 0, atol=1e-4)
end

"""
    test_stochasticforcingbudgets(; kwargs...)

Tests if the energy budgets are closed for BarotropicQG with stochastic forcing.
"""
function test_bqg_deterministicforcingbudgets(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ = 1e-1
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  gr  = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)
  k0, l0 = gr.kr[2], gr.l[2]

  # Forcing = 0.01cos(4x)cos(5y)cos(2t)
  f = @. 0.01*cos(4k0*x)*cos(5l0*y)
  fh = rfft(f)
  function calcFq!(Fqh, sol, t, cl, v, p, g)
    Fqh = fh*cos(2t)
    nothing
  end

  prob = BarotropicQG.Problem(nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt,
   stepper="RK4", calcFq=calcFq!, stochastic=false, dev=dev)

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

  t = round(μ*cl.t, digits=2)

  i₀ = 1
  dEdt = (E[(i₀+1):E.i] - E[i₀:E.i-1])/cl.dt
  ii = (i₀):E.i-1
  ii2 = (i₀+1):E.i

  # dEdt = W - D - R?
  total = W[ii2] - D[ii] - R[ii]

  residual = dEdt - total

  isapprox(mean(abs.(residual)), 0, atol=1e-8)
end

"""
    test_bqg_nonlinearadvection(dt, stepper; kwargs...)

Tests the advection term in the TwoDNavierStokes module by timestepping a
test problem with timestep dt and timestepper identified by the string stepper.
The test problem is derived by picking a solution ζf (with associated
streamfunction ψf) for which the advection term J(ψf, ζf) is non-zero. Next, a
forcing Ff is derived according to Ff = ∂ζf/∂t + J(ψf, ζf) - ν∇²ζf. One solution
to the vorticity equation forced by this Ff is then ζf. (This solution may not
be realized, at least at long times, if it is unstable.)
"""
function test_bqg_advection(dt, stepper, dev::Device=CPU(); n=128, L=2π, ν=1e-2, nν=1, μ=0.0)
  n, L  = 128, 2π
  ν, nν = 1e-2, 1
  μ = 0.0
  tf = 1.0
  nt = round(Int, tf/dt)

  gr  = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

  psif = @. sin(2x)*cos(2y) + 2sin(x)*cos(3y)
    qf = @. -8sin(2x)*cos(2y) - 20sin(x)*cos(3y)

  Ff = @. -(
    ν*( 64sin(2x)*cos(2y) + 200sin(x)*cos(3y) )
    + 8*( cos(x)*cos(3y)*sin(2x)*sin(2y) - 3cos(2x)*cos(2y)*sin(x)*sin(3y) )
  )

  Ffh = rfft(Ff)

  # Forcing
  function calcFq!(Fqh, sol, t, cl, v, p, g)
    Fqh .= Ffh
    nothing
  end

  prob = BarotropicQG.Problem(nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt, stepper=stepper, calcFq=calcFq!, dev=dev)
  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
  BarotropicQG.set_zeta!(prob, qf)

  # Step forward
  stepforward!(prob, round(Int, nt))
  BarotropicQG.updatevars!(prob)
  
  isapprox(v.q, qf, rtol=rtol_barotropicQG)
end

"""
    test_bqg_formstress(dt, stepper; kwargs...)

Tests the form stress term that forces the domain-averaged zonal flow U(t).
"""
function test_bqg_formstress(dt, stepper, dev::Device=CPU(); n=128, L=2π, ν=0.0, nν=1, μ=0.0)
  n, L  = 128, 2π
  ν, nν = 1e-2, 1
  μ = 0.0
  tf = 1
  nt = 1

  gr  = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

  zetai = @. -20*sin(10x)*cos(10y)
  topoPV(x, y) = @. cos(10x)*cos(10y)
  F(t) = 0 #no forcing

  answer = 0.25 # this is what <v*eta> should be

  prob = BarotropicQG.Problem(nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt, stepper=stepper, eta=topoPV, calcFU=F, dev=dev)
  BarotropicQG.set_zeta!(prob, zetai)
  BarotropicQG.updatevars!(prob)

  # Step forward
  stepforward!(prob, nt)
  isapprox(prob.timestepper.N[1, 1], answer, rtol=rtol_barotropicQG)
end

function test_bqg_energyenstrophy(dev::Device=CPU())
  nx, Lx  = 64, 2π
  ny, Ly  = 64, 3π
  g  = TwoDGrid(dev, nx, Lx, ny, Ly)
  k0, l0 = g.k[2], g.l[2] # fundamental wavenumbers
  x, y = gridpoints(g)

  energy_calc = 29/9
  enstrophy_calc = 2701/162

    eta = @. cos(10k0*x)*cos(10l0*y)
   psi0 = @. sin(2k0*x)*cos(2l0*y) + 2sin(k0*x)*cos(3l0*y)
  zeta0 = @. -((2k0)^2+(2l0)^2)*sin(2k0*x)*cos(2l0*y) - (k0^2+(3l0)^2)*2sin(k0*x)*cos(3l0*y)

  prob = BarotropicQG.Problem(nx=nx, Lx=Lx, ny=ny, Ly=Ly, eta = eta, stepper="ForwardEuler", dev=dev)
  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
  BarotropicQG.set_zeta!(prob, zeta0)
  BarotropicQG.updatevars!(prob)

  energyzeta0 = BarotropicQG.energy(prob)
  enstrophyzeta0 = BarotropicQG.enstrophy(prob)

  isapprox(energyzeta0, energy_calc, rtol=rtol_barotropicQG) && isapprox(enstrophyzeta0, enstrophy_calc, rtol=rtol_barotropicQG) &&
  BarotropicQG.addforcing!(prob.timestepper.N, sol, cl.t, cl, v, p, g)==nothing
end

function test_bqg_meanenergyenstrophy(dev::Device=CPU())
  nx, Lx  = 64, 2π
  ny, Ly  = 96, 3π
  g = TwoDGrid(dev, nx, Lx, ny, Ly)
  k0, l0 = g.k[2], g.l[2] # fundamental wavenumbers
  x, y = gridpoints(g)

  calcFU(t) = 0.0
  eta(x, y) = @. cos(10x)*cos(10y)
  psi0 = @. sin(2k0*x)*cos(2l0*y) + 2sin(k0*x)*cos(3l0*y)
 zeta0 = @. -((2k0)^2+(2l0)^2)*sin(2k0*x)*cos(2l0*y) - (k0^2+(3l0)^2)*2sin(k0*x)*cos(3l0*y)
  β = 10.0
  U = 1.2

  energy_calc = 29/9
  enstrophy_calc = 2701/162

  prob = BarotropicQG.Problem(nx=nx, Lx=Lx, ny=ny, Ly=Ly, β=β, eta=eta, calcFU = calcFU,
                                    stepper="ForwardEuler", dev=dev)

  BarotropicQG.set_zeta!(prob, zeta0)
  BarotropicQG.set_U!(prob, U)
  BarotropicQG.updatevars!(prob)

  energyU = BarotropicQG.meanenergy(prob)
  enstrophyU = BarotropicQG.meanenstrophy(prob)

  energyzeta0 = BarotropicQG.energy(prob)
  enstrophyzeta0 = BarotropicQG.enstrophy(prob)

  (isapprox(energyU, 0.5*U^2, rtol=rtol_barotropicQG) &&
    isapprox(enstrophyU, β*U, rtol=rtol_barotropicQG) &&
    isapprox(energyzeta0, energy_calc, rtol=rtol_barotropicQG) &&
    isapprox(enstrophyzeta0, enstrophy_calc, rtol=rtol_barotropicQG)
  )
end
