function test_sqg_problemtype(dev, T)
  prob = SurfaceQG.Problem(dev; T=T)

  A = device_array(dev)

  (typeof(prob.sol)<:A{Complex{T},2} && typeof(prob.grid.Lx)==T && eltype(prob.grid.x)==T && typeof(prob.vars.u)<:A{T,2})
end

function test_sqg_stochasticforcedproblemconstructor(dev::Device=CPU())
  
  function calcF!(Fqh, sol, t, clock, vars, params, grid)
    Fqh .= Ffh
    return nothing
  end
       
  prob = SurfaceQG.Problem(dev; calcF=calcF!, stochastic=true)
  
  return typeof(prob.vars.prevsol) == typeof(prob.sol)
end

"""
    test_sqg_advection(dt, stepper; kwargs...)

Tests the advection term in the SurfaceQG module by timestepping a
test problem with timestep dt and timestepper identified by the string stepper.
The test problem is derived by picking a solution bf (with associated
streamfunction ψf) for which the advection term J(ψf, bf) is non-zero. Next, a
forcing Ff is derived according to Ff = ∂bf/∂t + J(ψf, bf) - ν∇²bf. One solution
to the buoyancy equation forced by this Ff is then bf. (This solution may not
be realized, at least at long times, if it is unstable.)
"""
function test_sqg_advection(dt, stepper, dev::Device=CPU(); n=128, L=2π, ν=1e-2, nν=1)
  n, L  = 128, 2π
  ν, nν = 1e-2, 1
  tf = 0.5  # SQG piles up energy at small scales so running for t ⪆ 0.5 brings instability
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev, n, L)
  x, y = gridpoints(grid)

  ψf = @.             sin(2x)*cos(2y) +            2sin(x)*cos(3y)
  bf = @. - sqrt(8) * sin(2x)*cos(2y) - sqrt(10) * 2sin(x)*cos(3y)

  Ff = @. (
    - ν*(16*sqrt(2) * sin(2x)*cos(2y) + 10*sqrt(10) * 2sin(x)*cos(3y) )
    - 4*sqrt(2)*(sqrt(5)-2) * (cos(x)*cos(3y)*sin(2x)*sin(2y) 
      - 3cos(2x)*cos(2y)*sin(x)*sin(3y))
  )

  Ffh = rfft(Ff)

  # Forcing
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    Fh .= Ffh
    return nothing
  end

  prob = SurfaceQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, dt=dt, stepper=stepper, calcF=calcF!, stochastic=false)
  
  SurfaceQG.set_b!(prob, bf)

  stepforward!(prob, nt)
  
  SurfaceQG.updatevars!(prob)

  isapprox(prob.vars.b, bf, rtol=rtol_surfaceqg)
end

function test_sqg_kineticenergy_buoyancyvariance(dev::Device=CPU())
  nx, Lx  = 128, 2π
  ny, Ly  = 128, 3π
  
  grid = TwoDGrid(dev, nx, Lx, ny, Ly)
  x, y = gridpoints(grid)

  k₀, l₀ = 2π/grid.Lx, 2π/grid.Ly # fundamental wavenumbers
  ψ₀ = @. sin(2k₀*x)*cos(2l₀*y) + 2sin(k₀*x)*cos(3l₀*y)
  b₀ = @. - sqrt(8) * sin(2k₀*x)*cos(2l₀*y) - sqrt(10) * 2sin(k₀*x)*cos(3l₀*y)

  kinetic_energy_calc = 6
  buoyancy_variance_calc = 12

  prob = SurfaceQG.Problem(dev; nx=nx, Lx=Lx, ny=ny, Ly=Ly, stepper="ForwardEuler")

  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid;

  SurfaceQG.set_b!(prob, b₀)
  SurfaceQG.updatevars!(prob)

  kinetic_energy_b₀ = SurfaceQG.kinetic_energy(prob)
  buoyancy_variance_b₀ = SurfaceQG.buoyancy_variance(prob)

  params = SurfaceQG.Params(p.ν, p.nν)

  return (isapprox(kinetic_energy_b₀, kinetic_energy_calc, rtol=rtol_surfaceqg) &&
   isapprox(buoyancy_variance_b₀, buoyancy_variance_calc, rtol=rtol_surfaceqg))
end

function test_sqg_paramsconstructor(dev::Device=CPU())
  n, L = 128, 2π
  ν, nν = 1e-3, 4
  
  prob = SurfaceQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, stepper="ForwardEuler")
  
  params = SurfaceQG.Params(ν, nν)

  return prob.params == params
end

function test_sqg_noforcing(dev::Device=CPU())
  n, L = 16, 2π
  
  prob_unforced = SurfaceQG.Problem(dev; nx=n, Lx=L, stepper="ForwardEuler")

  SurfaceQG.addforcing!(prob_unforced.timestepper.N, prob_unforced.sol, prob_unforced.clock.t, prob_unforced.clock, prob_unforced.vars, prob_unforced.params, prob_unforced.grid)
    
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    Fh .= 2 * device_array(dev)(ones(size(sol)))
    return nothing
  end
  
  prob_forced = SurfaceQG.Problem(dev; nx=n, Lx=L, stepper="ForwardEuler", calcF=calcF!)

  SurfaceQG.addforcing!(prob_forced.timestepper.N, prob_forced.sol, prob_forced.clock.t, prob_forced.clock, prob_forced.vars, prob_forced.params, prob_forced.grid)
  
  return prob_unforced.timestepper.N == Complex.(device_array(dev)(zeros(size(prob_unforced.sol)))) && prob_forced.timestepper.N == Complex.(2*device_array(dev)(ones(size(prob_unforced.sol))))
end

function test_sqg_deterministicforcing_buoyancy_variance_budget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, tf=10.0)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  dt, tf = 0.005, 10
  nt = round(Int, tf/dt)

  grid  = TwoDGrid(dev, n, L)
  x, y = gridpoints(grid)

  # buoyancy forcing = 0.01cos(4x)cos(5y)cos(2t)
  f = @. 0.01cos(4x) * cos(5y)
  fh = rfft(f)
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    @. Fh = fh * cos(2t)
    return nothing
  end

  prob = SurfaceQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, dt=dt, stepper="RK4",
    calcF=calcF!, stochastic=false)

  SurfaceQG.set_b!(prob, 0*x)

  B = Diagnostic(SurfaceQG.buoyancy_variance,    prob, nsteps=nt)
  D = Diagnostic(SurfaceQG.buoyancy_dissipation, prob, nsteps=nt)
  W = Diagnostic(SurfaceQG.buoyancy_work,        prob, nsteps=nt)
  diags = [B, D, W]

  stepforward!(prob, diags, nt)
  
  SurfaceQG.updatevars!(prob)

  dBdt_numerical = (B[3:B.i] - B[1:B.i-2]) / (2 * prob.clock.dt)
  dBdt_computed  = W[2:B.i-1] - D[2:B.i-1]

  return isapprox(dBdt_numerical, dBdt_computed, rtol = 1e-4)
end

function test_sqg_stochasticforcing_buoyancy_variance_budget(dev::Device=CPU(); n=256, L=2π, dt=0.005, ν=1e-7, nν=2, tf=2.0)
  nt = round(Int, tf/dt)

  # Forcing parameters
  kf, dkf = 12.0, 2.0
  εᵇ = 0.01

  grid = TwoDGrid(dev, n, L)
  x, y = gridpoints(grid)

  Kr = device_array(dev)([CUDA.@allowscalar grid.kr[i] for i=1:grid.nkr, j=1:grid.nl])

  forcing_spectrum = device_array(dev)(zero(grid.Krsq))
  @. forcing_spectrum = exp(-(sqrt(grid.Krsq) - kf)^2 / (2 * dkf^2))
  @. forcing_spectrum = ifelse(grid.Krsq < 2^2, 0, forcing_spectrum)
  @. forcing_spectrum = ifelse(grid.Krsq > 20^2, 0, forcing_spectrum)
  @. forcing_spectrum = ifelse(Kr < 2π/L, 0, forcing_spectrum)
  εᵇ0 = parsevalsum(forcing_spectrum, grid) / (grid.Lx * grid.Ly)
  forcing_spectrum .= εᵇ / εᵇ0 * forcing_spectrum
  
  Random.seed!(1234)

  function calcF!(Fh, sol, t, clock, vars, params, grid)
    eta = device_array(dev)(exp.(2π * im * rand(Float64, size(sol))) / sqrt(clock.dt))
    CUDA.@allowscalar eta[1, 1] = 0.0
    @. Fh = eta * sqrt(forcing_spectrum)
    nothing
  end

  prob = SurfaceQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, dt=dt, stepper="RK4",
    calcF=calcF!, stochastic=true)

  SurfaceQG.set_b!(prob, 0*x)
  
  B = Diagnostic(SurfaceQG.buoyancy_variance,    prob, nsteps=nt)
  D = Diagnostic(SurfaceQG.buoyancy_dissipation, prob, nsteps=nt)
  W = Diagnostic(SurfaceQG.buoyancy_work,        prob, nsteps=nt)
  diags = [B, D, W]

  stepforward!(prob, diags, nt)

  SurfaceQG.updatevars!(prob)

  dBdt_numerical = (B[2:B.i] - B[1:B.i-1]) / prob.clock.dt

  dBdt_computed = W[2:B.i] - D[1:B.i-1]
  
  return isapprox(dBdt_numerical, dBdt_computed, rtol = 1e-4)
end
