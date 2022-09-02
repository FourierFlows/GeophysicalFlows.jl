function test_twodnavierstokes_lambdipole(n, dt, dev::Device=CPU(); L=2π, Ue=1, Re=L/20, ν=0.0, nν=1, ti=L/Ue*0.01, nm=3)
  nt = round(Int, ti/dt)
  prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν, nν, dt, stepper="FilteredRK4")
  x, y = gridpoints(prob.grid)
  ζ = prob.vars.ζ

  ζ₀ = lambdipole(Ue, Re, prob.grid)
  TwoDNavierStokes.set_ζ!(prob, ζ₀)

  xζ = zeros(nm)    # centroid of abs(ζ)
  Ue_m = zeros(nm)  # measured dipole speed
  
  for i = 1:nm # step forward
    stepforward!(prob, nt)
    TwoDNavierStokes.updatevars!(prob)
    
    xζ[i] = mean(@. abs(ζ) * x) / mean(abs.(ζ))
    if i > 1
      Ue_m[i] = (xζ[i] - xζ[i-1]) / ((nt-1)*dt)
    end
  end
  isapprox(Ue, mean(Ue_m[2:end]), rtol=rtol_lambdipole)
end

function test_twodnavierstokes_stochasticforcing_energybudget(dev::Device=CPU(); n=256, L=2π, dt=0.005, ν=1e-7, nν=2, μ=1e-1, nμ=0, tf=0.1/μ)
  nt = round(Int, tf/dt)

  # Forcing parameters
  kf, dkf = 12.0, 2.0
  ε = 0.1

  grid = TwoDGrid(dev; nx=n, Lx=L)
  x, y = gridpoints(grid)

  Kr = device_array(dev)([CUDA.@allowscalar grid.kr[i] for i=1:grid.nkr, j=1:grid.nl])

  forcing_spectrum = device_array(dev)(zero(grid.Krsq))
  @. forcing_spectrum = exp(-(sqrt(grid.Krsq) - kf)^2 / (2 * dkf^2))
  @. forcing_spectrum = ifelse(grid.Krsq < 2^2, 0, forcing_spectrum)
  @. forcing_spectrum = ifelse(grid.Krsq > 20^2, 0, forcing_spectrum)
  @. forcing_spectrum = ifelse(Kr < 2π/L, 0, forcing_spectrum)
  ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
  forcing_spectrum .= ε / ε0 * forcing_spectrum

  Random.seed!(1234)

  function calcF!(Fh, sol, t, clock, vars, params, grid)
    eta = device_array(dev)(exp.(2π * im * rand(Float64, size(sol))) / sqrt(clock.dt))
    CUDA.@allowscalar eta[1, 1] = 0.0
    @. Fh = eta * sqrt(forcing_spectrum)
    
    return nothing
  end

  prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=true)

  TwoDNavierStokes.set_ζ!(prob, 0*x)
  
  E = Diagnostic(TwoDNavierStokes.energy,                            prob, nsteps=nt)
  D = Diagnostic(TwoDNavierStokes.energy_dissipation_hyperviscosity, prob, nsteps=nt)
  R = Diagnostic(TwoDNavierStokes.energy_dissipation_hypoviscosity,  prob, nsteps=nt)
  W = Diagnostic(TwoDNavierStokes.energy_work,                       prob, nsteps=nt)
  diags = [E, D, W, R]

  stepforward!(prob, diags, nt)
  
  TwoDNavierStokes.updatevars!(prob)

  dEdt_numerical = (E[2:E.i] - E[1:E.i-1]) / prob.clock.dt

  dEdt_computed = W[2:E.i] + D[1:E.i-1] + R[1:E.i-1]

  return isapprox(dEdt_numerical, dEdt_computed, rtol = 1e-3)
end

function test_twodnavierstokes_stochasticforcing_enstrophybudget(dev::Device=CPU(); n=256, L=2π, dt=0.005, ν=1e-7, nν=2, μ=1e-1, nμ=0, tf=0.1/μ)
  nt = round(Int, tf/dt)

  # Forcing parameters
  kf, dkf = 12.0, 2.0
  εᶻ = 0.1

  grid = TwoDGrid(dev; nx=n, Lx=L)
  x, y = gridpoints(grid)

  Kr = device_array(dev)([CUDA.@allowscalar grid.kr[i] for i=1:grid.nkr, j=1:grid.nl])

  forcing_spectrum = device_array(dev)(zero(grid.Krsq))
  @. forcing_spectrum = exp(-(sqrt(grid.Krsq) - kf)^2 / (2 * dkf^2))
  @. forcing_spectrum = ifelse(grid.Krsq < 2^2, 0, forcing_spectrum)
  @. forcing_spectrum = ifelse(grid.Krsq > 20^2, 0, forcing_spectrum)
  @. forcing_spectrum = ifelse(Kr < 2π/L, 0, forcing_spectrum)
  εᶻ0 = parsevalsum(forcing_spectrum / 2, grid) / (grid.Lx * grid.Ly)
  forcing_spectrum .= εᶻ / εᶻ0 * forcing_spectrum
  
  Random.seed!(1234)

  function calcF!(Fh, sol, t, cl, v, p, g)
    eta = device_array(dev)(exp.(2π * im * rand(Float64, size(sol))) / sqrt(cl.dt))
    CUDA.@allowscalar eta[1, 1] = 0.0
    @. Fh = eta * sqrt(forcing_spectrum)
    
    nothing
  end

  prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=true)

  TwoDNavierStokes.set_ζ!(prob, 0*x)
  
  Z = Diagnostic(TwoDNavierStokes.enstrophy,                            prob, nsteps=nt)
  D = Diagnostic(TwoDNavierStokes.enstrophy_dissipation_hyperviscosity, prob, nsteps=nt)
  R = Diagnostic(TwoDNavierStokes.enstrophy_dissipation_hypoviscosity,  prob, nsteps=nt)
  W = Diagnostic(TwoDNavierStokes.enstrophy_work,                       prob, nsteps=nt)
  diags = [Z, D, W, R]

  stepforward!(prob, diags, nt)

  TwoDNavierStokes.updatevars!(prob)

  dZdt_numerical = (Z[2:Z.i] - Z[1:Z.i-1]) / prob.clock.dt

  dZdt_computed = W[2:Z.i] + D[1:Z.i-1] + R[1:Z.i-1]

  return isapprox(dZdt_numerical, dZdt_computed, rtol = 1e-3)
end

function test_twodnavierstokes_deterministicforcing_energybudget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1, nμ=0)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ, nμ = 1e-1, 0
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev; nx=n, Lx=L)
  x, y = gridpoints(grid)

  # Forcing = 0.01cos(4x)cos(5y)cos(2t)
  f = @. 0.01cos(4x) * cos(5y)
  fh = rfft(f)
  
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    @. Fh = fh * cos(2t)
    
    return nothing
  end

  prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=false)

  TwoDNavierStokes.set_ζ!(prob, 0*x)

  E = Diagnostic(TwoDNavierStokes.energy,                            prob, nsteps=nt)
  D = Diagnostic(TwoDNavierStokes.energy_dissipation_hyperviscosity, prob, nsteps=nt)
  R = Diagnostic(TwoDNavierStokes.energy_dissipation_hypoviscosity,  prob, nsteps=nt)
  W = Diagnostic(TwoDNavierStokes.energy_work,                       prob, nsteps=nt)
  diags = [E, D, W, R]

  stepforward!(prob, diags, nt)
  
  TwoDNavierStokes.updatevars!(prob)

  dEdt_numerical = (E[3:E.i] - E[1:E.i-2]) / (2 * prob.clock.dt)
  dEdt_computed  = W[2:E.i-1] + D[2:E.i-1] + R[2:E.i-1]

  return isapprox(dEdt_numerical, dEdt_computed, rtol = 1e-4)
end

function test_twodnavierstokes_deterministicforcing_enstrophybudget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1, nμ=0)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ, nμ = 1e-1, 0
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev; nx=n, Lx=L)
  x, y = gridpoints(grid)

  # Forcing = 0.01cos(4x)cos(5y)cos(2t)
  f = @. 0.01cos(4x) * cos(5y)
  fh = rfft(f)
  
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    @. Fh = fh * cos(2t)
    
    return nothing
  end

  prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=false)

  TwoDNavierStokes.set_ζ!(prob, 0*x)

  Z = Diagnostic(TwoDNavierStokes.enstrophy,                            prob, nsteps=nt)
  D = Diagnostic(TwoDNavierStokes.enstrophy_dissipation_hyperviscosity, prob, nsteps=nt)
  R = Diagnostic(TwoDNavierStokes.enstrophy_dissipation_hypoviscosity,  prob, nsteps=nt)
  W = Diagnostic(TwoDNavierStokes.enstrophy_work,                       prob, nsteps=nt)
  diags = [Z, D, W, R]

  stepforward!(prob, diags, nt)
  
  TwoDNavierStokes.updatevars!(prob)

  dZdt_numerical = (Z[3:Z.i] - Z[1:Z.i-2]) / (2 * prob.clock.dt)
  dZdt_computed  = W[2:Z.i-1] + D[2:Z.i-1] + R[2:Z.i-1]

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

  grid = TwoDGrid(dev; nx=n, Lx=L)
  x, y = gridpoints(grid)

  ψf = @.   sin(2x)*cos(2y) +  2sin(x)*cos(3y)
  ζf = @. -8sin(2x)*cos(2y) - 20sin(x)*cos(3y)

  Ff = @. -(
    ν*( 64sin(2x)*cos(2y) + 200sin(x)*cos(3y) )
    + 8*( cos(x)*cos(3y)*sin(2x)*sin(2y) - 3cos(2x)*cos(2y)*sin(x)*sin(3y) )
  )

  Ffh = rfft(Ff)

  # Forcing
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    Fh .= Ffh
    
    return nothing
  end

  prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν, nν, μ, nμ, dt, stepper, calcF=calcF!, stochastic=false)
  
  TwoDNavierStokes.set_ζ!(prob, ζf)

  stepforward!(prob, nt)
  
  TwoDNavierStokes.updatevars!(prob)

  isapprox(prob.vars.ζ, ζf, rtol=rtol_twodnavierstokes)
end

function test_twodnavierstokes_energyenstrophy(dev::Device=CPU())
  nx, Lx  = 128, 2π
  ny, Ly  = 126, 3π
  
  grid = TwoDGrid(dev; nx, Lx, ny, Ly)
  x, y = gridpoints(grid)

  k₀, l₀ = 2π/grid.Lx, 2π/grid.Ly # fundamental wavenumbers
  ψ₀ = @. sin(2k₀*x)*cos(2l₀*y) + 2sin(k₀*x)*cos(3l₀*y)
  ζ₀ = @. -((2k₀)^2+(2l₀)^2)*sin(2k₀*x)*cos(2l₀*y) - (k₀^2+(3l₀)^2)*2sin(k₀*x)*cos(3l₀*y)

  energy_calc = 29/9
  enstrophy_calc = 2701/162

  prob = TwoDNavierStokes.Problem(dev; nx, Lx, ny, Ly, stepper="ForwardEuler")

  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid;

  TwoDNavierStokes.set_ζ!(prob, ζ₀)
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

  A = device_array(dev)

  (typeof(prob.sol)<:A{Complex{T},2} && typeof(prob.grid.Lx)==T && eltype(prob.grid.x)==T && typeof(prob.vars.u)<:A{T,2})
end
