function test_twodnavierstokes_lambdipole(n, dt, dev::Device=CPU(); L=2π, Ue=1, Re=L/20, ν=0.0, nν=1, ti=L/Ue*0.01, nm=3)
  nt = round(Int, ti/dt)
  prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, dt=dt, stepper="FilteredRK4")
  zeta₀ = lambdipole(Ue, Re, prob.grid)
  TwoDNavierStokes.set_zeta!(prob, zeta₀)

  xzeta = zeros(nm) # centroid of abs(zeta)
  Ue_m = zeros(nm)  # measured dipole speed
  x, y = gridpoints(prob.grid)
  zeta = prob.vars.zeta

  for i = 1:nm # step forward
    stepforward!(prob, nt)
    TwoDNavierStokes.updatevars!(prob)
    xzeta[i] = mean(@. abs(zeta)*x) / mean(abs.(zeta))
    if i > 1
      Ue_m[i] = (xzeta[i]-xzeta[i-1]) / ((nt-1)*dt)
    end
  end
  isapprox(Ue, mean(Ue_m[2:end]), rtol=rtol_lambdipole)
end

function test_twodnavierstokes_stochasticforcing_energybudget(dev::Device=CPU(); n=256, L=2π, dt=0.005, ν=1e-7, nν=2, μ=1e-1, nμ=0, tf=0.1/μ)
  nt = round(Int, tf/dt)

  # Forcing parameters
  kf, dkf = 12.0, 2.0
  ε = 0.1

  gr  = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

  Kr = ArrayType(dev)([CUDA.@allowscalar gr.kr[i] for i=1:gr.nkr, j=1:gr.nl])

  forcingcovariancespectrum = ArrayType(dev)(zero(gr.Krsq))
  @. forcingcovariancespectrum = exp(-(sqrt(gr.Krsq) - kf)^2 / (2 * dkf^2))
  @. forcingcovariancespectrum = ifelse(gr.Krsq < 2^2, 0, forcingcovariancespectrum)
  @. forcingcovariancespectrum = ifelse(gr.Krsq > 20^2, 0, forcingcovariancespectrum)
  @. forcingcovariancespectrum = ifelse(Kr < 2π/L, 0, forcingcovariancespectrum)
  ε0 = parsevalsum(forcingcovariancespectrum .* gr.invKrsq / 2, gr) / (gr.Lx * gr.Ly)
  forcingcovariancespectrum .= ε / ε0 * forcingcovariancespectrum

  Random.seed!(1234)

  function calcF!(Fh, sol, t, clock, vars, params, grid)
    eta = ArrayType(dev)(exp.(2π * im * rand(Float64, size(sol))) / sqrt(clock.dt))
    CUDA.@allowscalar eta[1, 1] = 0.0
    @. Fh = eta * sqrt(forcingcovariancespectrum)
    return nothing
  end

  prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=true)

  TwoDNavierStokes.set_zeta!(prob, 0*x)
  
  E = Diagnostic(TwoDNavierStokes.energy,      prob, nsteps=nt)
  D = Diagnostic(TwoDNavierStokes.energy_dissipation, prob, nsteps=nt)
  R = Diagnostic(TwoDNavierStokes.energy_drag,        prob, nsteps=nt)
  W = Diagnostic(TwoDNavierStokes.energy_work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  stepforward!(prob, diags, nt)
  
  TwoDNavierStokes.updatevars!(prob)

  dEdt_numerical = (E[2:E.i] - E[1:E.i-1]) / prob.clock.dt

  # If the Ito interpretation was used for the work
  # then we need to add the drift term
  # dEdt_computed = W[2:E.i] + ε - D[1:E.i-1] - R[1:E.i-1]      # Ito
  dEdt_computed = W[2:E.i] - D[1:E.i-1] - R[1:E.i-1]        # Stratonovich

  return isapprox(dEdt_numerical, dEdt_computed, rtol = 1e-3)
end

function test_twodnavierstokes_stochasticforcing_enstrophybudget(dev::Device=CPU(); n=256, L=2π, dt=0.005, ν=1e-7, nν=2, μ=1e-1, nμ=0, tf=0.1/μ)
  nt = round(Int, tf/dt)

  # Forcing parameters
  kf, dkf = 12.0, 2.0
  εᶻ = 0.1

  gr  = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

  Kr = ArrayType(dev)([CUDA.@allowscalar gr.kr[i] for i=1:gr.nkr, j=1:gr.nl])

  forcingcovariancespectrum = ArrayType(dev)(zero(gr.Krsq))
  @. forcingcovariancespectrum = exp(-(sqrt(gr.Krsq) - kf)^2 / (2 * dkf^2))
  @. forcingcovariancespectrum = ifelse(gr.Krsq < 2^2, 0, forcingcovariancespectrum)
  @. forcingcovariancespectrum = ifelse(gr.Krsq > 20^2, 0, forcingcovariancespectrum)
  @. forcingcovariancespectrum = ifelse(Kr < 2π/L, 0, forcingcovariancespectrum)
  εᶻ0 = parsevalsum(forcingcovariancespectrum / 2, gr) / (gr.Lx * gr.Ly)
  forcingcovariancespectrum .= εᶻ / εᶻ0 * forcingcovariancespectrum
  
  Random.seed!(1234)

  function calcF!(Fh, sol, t, cl, v, p, g)
    eta = ArrayType(dev)(exp.(2π * im * rand(Float64, size(sol))) / sqrt(cl.dt))
    CUDA.@allowscalar eta[1, 1] = 0.0
    @. Fh = eta * sqrt(forcingcovariancespectrum)
    nothing
  end

  prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=true)

  TwoDNavierStokes.set_zeta!(prob, 0*x)
  Z = Diagnostic(TwoDNavierStokes.enstrophy,      prob, nsteps=nt)
  D = Diagnostic(TwoDNavierStokes.enstrophy_dissipation, prob, nsteps=nt)
  R = Diagnostic(TwoDNavierStokes.enstrophy_drag,        prob, nsteps=nt)
  W = Diagnostic(TwoDNavierStokes.enstrophy_work,        prob, nsteps=nt)
  diags = [Z, D, W, R]

  stepforward!(prob, diags, nt)

  TwoDNavierStokes.updatevars!(prob)

  dZdt_numerical = (Z[2:Z.i] - Z[1:Z.i-1]) / prob.clock.dt

  # If the Ito interpretation was used for the work
  # then we need to add the drift term
  # dZdt_computed = W[2:Z.i] + εᶻ - D[1:Z.i-1] - R[1:Z.i-1]      # Ito
  dZdt_computed = W[2:Z.i] - D[1:Z.i-1] - R[1:Z.i-1]        # Stratonovich

  return isapprox(dZdt_numerical, dZdt_computed, rtol = 1e-3)
end

function test_twodnavierstokes_deterministicforcing_energybudget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1, nμ=0)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ, nμ = 1e-1, 0
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  gr  = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

  # Forcing = 0.01cos(4x)cos(5y)cos(2t)
  f = @. 0.01cos(4x) * cos(5y)
  fh = rfft(f)
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    @. Fh = fh * cos(2t)
    return nothing
  end

  prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=false)

  TwoDNavierStokes.set_zeta!(prob, 0*x)

  E = Diagnostic(TwoDNavierStokes.energy,      prob, nsteps=nt)
  D = Diagnostic(TwoDNavierStokes.energy_dissipation, prob, nsteps=nt)
  R = Diagnostic(TwoDNavierStokes.energy_drag,        prob, nsteps=nt)
  W = Diagnostic(TwoDNavierStokes.energy_work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  stepforward!(prob, diags, nt)
  
  TwoDNavierStokes.updatevars!(prob)

  dEdt_numerical = (E[3:E.i] - E[1:E.i-2]) / (2 * prob.clock.dt)
  dEdt_computed  = W[2:E.i-1] - D[2:E.i-1] - R[2:E.i-1]

  return isapprox(dEdt_numerical, dEdt_computed, rtol = 1e-4)
end

function test_twodnavierstokes_deterministicforcing_enstrophybudget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1, nμ=0)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ, nμ = 1e-1, 0
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  gr  = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

  # Forcing = 0.01cos(4x)cos(5y)cos(2t)
  f = @. 0.01cos(4x) * cos(5y)
  fh = rfft(f)
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    @. Fh = fh * cos(2t)
    return nothing
  end

  prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=false)

  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  TwoDNavierStokes.set_zeta!(prob, 0*x)

  Z = Diagnostic(TwoDNavierStokes.enstrophy,      prob, nsteps=nt)
  D = Diagnostic(TwoDNavierStokes.enstrophy_dissipation, prob, nsteps=nt)
  R = Diagnostic(TwoDNavierStokes.enstrophy_drag,        prob, nsteps=nt)
  W = Diagnostic(TwoDNavierStokes.enstrophy_work,        prob, nsteps=nt)
  diags = [Z, D, W, R]

  stepforward!(prob, diags, nt)
  
  TwoDNavierStokes.updatevars!(prob)

  dZdt_numerical = (Z[3:Z.i] - Z[1:Z.i-2]) / (2 * prob.clock.dt)
  dZdt_computed  = W[2:Z.i-1] - D[2:Z.i-1] - R[2:Z.i-1]

  return isapprox(dZdt_numerical, dZdt_computed, rtol = 1e-4)
end

"""
    testnonlinearterms(dt, stepper; kwargs...)

Tests the advection term in the twodnavierstokes module by timestepping a
test problem with timestep dt and timestepper identified by the string stepper.
The test problem is derived by picking a solution ζf (with associated
streamfunction ψf) for which the advection term J(ψf, ζf) is non-zero. Next, a
forcing Ff is derived according to Ff = ∂ζf/∂t + J(ψf, ζf) - ν∇²ζf. One solution
to the vorticity equation forced by this Ff is then ζf. (This solution may not
be realized, at least at long times, if it is unstable.)
"""
function test_twodnavierstokes_advection(dt, stepper, dev::Device=CPU(); n=128, L=2π, ν=1e-2, nν=1, μ=0.0, nμ=0)
  n, L  = 128, 2π
  ν, nν = 1e-2, 1
  μ, nμ = 0.0, 0
  tf = 1.0
  nt = round(Int, tf/dt)

  gr = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

   psif = @.   sin(2x)*cos(2y) +  2sin(x)*cos(3y)
  zetaf = @. -8sin(2x)*cos(2y) - 20sin(x)*cos(3y)

  Ff = @. -(
    ν*( 64sin(2x)*cos(2y) + 200sin(x)*cos(3y) )
    + 8*( cos(x)*cos(3y)*sin(2x)*sin(2y) - 3cos(2x)*cos(2y)*sin(x)*sin(3y) )
  )

  Ffh = rfft(Ff)

  # Forcing
  function calcF!(Fh, sol, t, cl, v, p, g)
    Fh .= Ffh
    return nothing
  end

  prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt, stepper=stepper, calcF=calcF!, stochastic=false)
  sol, cl, p, v, g = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
  TwoDNavierStokes.set_zeta!(prob, zetaf)

  stepforward!(prob, nt)
  TwoDNavierStokes.updatevars!(prob)

  isapprox(prob.vars.zeta, zetaf, rtol=rtol_twodnavierstokes)
end

function test_twodnavierstokes_energyenstrophy(dev::Device=CPU())
  nx, Lx  = 128, 2π
  ny, Ly  = 128, 3π
  gr = TwoDGrid(dev, nx, Lx, ny, Ly)
  x, y = gridpoints(gr)

  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly # fundamental wavenumbers
   ψ₀ = @. sin(2k₀*x)*cos(2l₀*y) + 2sin(k₀*x)*cos(3l₀*y)
   ζ₀ = @. -((2k₀)^2+(2l₀)^2)*sin(2k₀*x)*cos(2l₀*y) - (k₀^2+(3l₀)^2)*2sin(k₀*x)*cos(3l₀*y)

  energy_calc = 29/9
  enstrophy_calc = 2701/162

  prob = TwoDNavierStokes.Problem(dev; nx=nx, Lx=Lx, ny=ny, Ly=Ly, stepper="ForwardEuler")

  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid;

  TwoDNavierStokes.set_zeta!(prob, ζ₀)
  TwoDNavierStokes.updatevars!(prob)

  energyζ₀ = TwoDNavierStokes.energy(prob)
  enstrophyζ₀ = TwoDNavierStokes.enstrophy(prob)

  params = TwoDNavierStokes.Params(p.ν, p.nν)

  (isapprox(energyζ₀, energy_calc, rtol=rtol_twodnavierstokes) &&
   isapprox(enstrophyζ₀, enstrophy_calc, rtol=rtol_twodnavierstokes) &&
   TwoDNavierStokes.addforcing!(prob.timestepper.N, sol, cl.t, cl, v, p, g)==nothing && p == params)
end

function test_twodnavierstokes_problemtype(dev, T)
  prob = TwoDNavierStokes.Problem(dev; T=T)

  A = ArrayType(dev)

  (typeof(prob.sol)<:A{Complex{T},2} && typeof(prob.grid.Lx)==T && eltype(prob.grid.x)==T && typeof(prob.vars.u)<:A{T,2})
end
