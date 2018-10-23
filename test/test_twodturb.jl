function test_twodturb_lambdipole(n, dt; L=2π, Ue=1, Re=L/20, nu=0.0, nnu=1, ti=L/Ue*0.01, nm=3)
  nt = round(Int, ti/dt)
  prob = TwoDTurb.Problem(nx=n, Lx=L, nu=nu, nnu=nnu, dt=dt, stepper="FilteredRK4")
  zeta₀ = lambdipole(Ue, Re, prob.grid)
  TwoDTurb.set_zeta!(prob, zeta₀)

  xzeta = zeros(nm)   # centroid of abs(zeta)
  Ue_m = zeros(nm) # measured dipole speed
  x, y = gridpoints(prob.grid)
  zeta = prob.vars.zeta

  for i = 1:nm # step forward
    stepforward!(prob, nt)
    TwoDTurb.updatevars!(prob)
    xzeta[i] = mean(abs.(zeta).*x) / mean(abs.(zeta))
    if i > 1
      Ue_m[i] = (xzeta[i]-xzeta[i-1]) / ((nt-1)*dt)
    end
  end
  isapprox(Ue, mean(Ue_m[2:end]), rtol=rtol_lambdipole)
end

function test_twodturb_stochasticforcingbudgets(; n=256, L=2π, dt=0.005, nu=1e-7, nnu=2, mu=1e-1, nmu=0, tf=0.1/mu)
  nt = round(Int, tf/dt)

  # Forcing
  kf, dkf = 12, 2
  σ = 0.1
  g  = TwoDGrid(n, L)

  force2k = @. exp( -(sqrt(g.Krsq)-kf)^2 / (2dkf^2) )
  @. force2k[g.Krsq < 2^2] = 0
  @. force2k[g.Krsq > 20^2] = 0
  @. force2k[1, :] = 0

  σ0 = parsevalsum(force2k.*g.invKrsq/2, g) / (g.Lx*g.Ly)
  @. force2k *= σ/σ0

  Random.seed!(1234)

  function calcF!(F, sol, t, cl, v, p, g)
    eta = exp.(2π*im*rand(Float64, size(sol))) / sqrt(cl.dt)
    eta[1, 1] = 0
    @. F = eta * sqrt(force2k)
    nothing
  end

  prob = TwoDTurb.Problem(nx=n, Lx=L, nu=nu, nnu=nnu, mu=mu, nmu=nmu, dt=dt,
                          stepper="RK4", calcF=calcF!, stochastic=true)

  TwoDTurb.set_zeta!(prob, zeros(g.nx, g.ny))
  E = Diagnostic(TwoDTurb.energy,      prob, nsteps=nt)
  D = Diagnostic(TwoDTurb.dissipation, prob, nsteps=nt)
  R = Diagnostic(TwoDTurb.drag,        prob, nsteps=nt)
  W = Diagnostic(TwoDTurb.work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  stepforward!(prob, diags, nt)
  TwoDTurb.updatevars!(prob)

  i₀ = 1
  dEdt = (E[(i₀+1):E.i] - E[i₀:E.i-1])/prob.clock.dt
  ii = (i₀):E.i-1
  ii2 = (i₀+1):E.i

  # dEdt = W - D - R?
  # If the Ito interpretation was used for the work
  # then we need to add the drift term
  # total = W[ii2]+σ - D[ii] - R[ii]      # Ito
  total = W[ii2] - D[ii] - R[ii]        # Stratonovich

  residual = dEdt - total
  isapprox(mean(abs.(residual)), 0, atol=1e-4)
end


function test_twodturb_deterministicforcingbudgets(; n=256, L=2π, dt=0.005, nu=1e-7, nnu=2, mu=1e-1, nmu=0, tf=0.1/mu)
  nt = round(Int, tf/dt)
  g = TwoDGrid(n, L)

  f = @. 0.01*cos(4*g.x)*cos(5*g.y)
  fh = rfft(f)
  function calcF!(F, sol, t, cl, v, p, g)
    @. F = fh*cos(2t)
    nothing
  end

  prob = TwoDTurb.Problem(nx=n, Lx=L, nu=nu, nnu=nnu, mu=mu, nmu=nmu, dt=dt, stepper="RK4", calcF=calcF!)
  TwoDTurb.set_zeta!(prob, zeros(n, n))

  E = Diagnostic(TwoDTurb.energy,      prob, nsteps=nt)
  D = Diagnostic(TwoDTurb.dissipation, prob, nsteps=nt)
  R = Diagnostic(TwoDTurb.drag,        prob, nsteps=nt)
  W = Diagnostic(TwoDTurb.work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  # Step forward
  stepforward!(prob, diags, nt)
  TwoDTurb.updatevars!(prob)

  i₀ = 1
  dEdt = (E[i₀+1:E.i] - E[i₀:E.i-1])/prob.clock.dt
  ii = i₀:(E.i-1)
  ii2 = (i₀+1):E.i

  # dEdt = W - D - R?
  total = W[ii2] - D[ii] - R[ii]
  residual = dEdt - total

  isapprox(mean(abs.(residual)), 0, atol=1e-8)
end

"""
    testnonlinearterms(dt, stepper; kwargs...)

Tests the advection term in the twodturb module by timestepping a
test problem with timestep dt and timestepper identified by the string stepper.
The test problem is derived by picking a solution ζf (with associated
streamfunction ψf) for which the advection term J(ψf, ζf) is non-zero. Next, a
forcing Ff is derived according to Ff = ∂ζf/∂t + J(ψf, ζf) - nuΔζf. One solution
to the vorticity equation forced by this Ff is then ζf. (This solution may not
be realized, at least at long times, if it is unstable.)
"""
function test_twodturb_advection(dt, stepper; n=128, L=2π, nu=1e-2, nnu=1, mu=0.0, nmu=0)
  n, L  = 128, 2π
  nu, nnu = 1e-2, 1
  mu, nmu = 0.0, 0
  tf = 1.0
  nt = round(Int, tf/dt)

  gr  = TwoDGrid(n, L)
  x, y = gridpoints(gr)

  psif = @. sin(2x)*cos(2y) + 2sin(x)*cos(3y)
  zetaf = @. -8sin(2x)*cos(2y) - 20sin(x)*cos(3y)

  Ff = @. -(
    nu*( 64sin(2x)*cos(2y) + 200sin(x)*cos(3y) )
    + 8*( cos(x)*cos(3y)*sin(2x)*sin(2y) - 3cos(2x)*cos(2y)*sin(x)*sin(3y) )
  )

  Ffh = rfft(Ff)
  function calcF!(F, sol, t, cl, v, p, g)
    F .= Ffh
    nothing
  end

  prob = TwoDTurb.Problem(nx=n, Lx=L, nu=nu, nnu=nnu, mu=mu, nmu=nmu, dt=dt, stepper=stepper, calcF=calcF!)
  TwoDTurb.set_zeta!(prob, zetaf)

  stepforward!(prob, nt)
  TwoDTurb.updatevars!(prob)

  isapprox(prob.vars.zeta, zetaf, rtol=rtol_twodturb)
end

function test_twodturb_energyenstrophy()
  nx, Lx  = 128, 2π
  ny, Ly  = 128, 3π
  g  = TwoDGrid(nx, Lx, ny, Ly)
  k0 = g.k[2] # fundamental wavenumber
  l0 = g.l[2] # fundamental wavenumber
  x, y = gridpoints(g)

   psi0 = @. sin(2*k0*x)*cos(2*l0*y) + 2sin(k0*x)*cos(3*l0*y)
  zeta0 = @. -((2*k0)^2+(2*l0)^2)*sin(2*k0*x)*cos(2*l0*y) - (k0^2+(3*l0)^2)*2sin(k0*x)*cos(3*l0*y)

  energy_calc = 29/9
  enstrophy_calc = 2701/162

  prob = TwoDTurb.Problem(nx=nx, Lx=Lx, ny=ny, Ly=Ly, stepper="ForwardEuler")
  TwoDTurb.set_zeta!(prob, zeta0)
  TwoDTurb.updatevars!(prob)

  energy_num = TwoDTurb.energy(prob)
  enstrophy_num = TwoDTurb.enstrophy(prob)

  isapprox(energy_num, energy_calc, rtol=rtol_twodturb) && isapprox(enstrophy_num, enstrophy_calc, rtol=rtol_twodturb)
end
