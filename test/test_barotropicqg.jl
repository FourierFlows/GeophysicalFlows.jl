"""
    test_bqg_rossbywave(stepper, dt, nsteps, dev; kwargs...)

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
    eta(x, y) = 0 * x
  end

  prob = BarotropicQG.Problem(dev; nx=nx, Lx=Lx, eta=eta, β=β, μ=μ, ν=ν, stepper=stepper, dt=dt)
  sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  x, y = gridpoints(grid)

  # the Rossby wave initial condition
   ampl = 1e-2
  kwave = 3 * 2π/grid.Lx
  lwave = 2 * 2π/grid.Ly
      ω = -params.β * kwave / (kwave^2 + lwave^2)
     ζ0 = @. ampl * cos(kwave * x) * cos(lwave * y)
    ζ0h = rfft(ζ0)

  BarotropicQG.set_zeta!(prob, ζ0)

  stepforward!(prob, nsteps)
  dealias!(sol, grid)
  BarotropicQG.updatevars!(prob)

  ζ_theory = @. ampl * cos(kwave * (x - ω/kwave * clock.t)) * cos(lwave * y)

  return isapprox(ζ_theory, vars.zeta, rtol=grid.nx * grid.ny * nsteps * 1e-12)
end

"""
    test_bqg_stochasticforcing_energy_budgets(dev; kwargs...)

Tests if the energy budget is closed for a BarotropicQG problem with stochastic forcing.
"""
function test_bqg_stochasticforcing_energybudget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1, T=Float64)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ = 1e-1
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev, n, L)
  x, y = gridpoints(grid)
  K  = @. sqrt(grid.Krsq)                      # a 2D array with the total wavenumber
  
  # Forcing with spectrum ~ exp(-(k-kf)²/(2 dkf²))
  kf, dkf = 12.0, 2.0
  ε = 0.1
  
  forcing_spectrum = zeros(dev, T, (grid.nkr, grid.nl))
  @. forcing_spectrum = exp(-(K - kf)^2 / (2 * dkf^2))
  @. forcing_spectrum = ifelse(grid.Krsq < 2^2,  0, forcing_spectrum)   # no power at low wavenumbers
  @. forcing_spectrum = ifelse(grid.Krsq > 20^2, 0, forcing_spectrum)   # no power at high wavenumbers
  ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
  forcing_spectrum .= ε / ε0 * forcing_spectrum

  Random.seed!(1234)

  function calcF!(Fh, sol, t, clock, vars, params, grid)
    eta = ArrayType(dev)(exp.(2π * im * rand(T, size(sol))) / sqrt(clock.dt))
    CUDA.@allowscalar eta[1, 1] = 0
    @. Fh = eta * sqrt(forcing_spectrum)
    return nothing
  end

  prob = BarotropicQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=true)

  BarotropicQG.set_zeta!(prob, 0*x)
  
  E = Diagnostic(BarotropicQG.energy,             prob, nsteps=nt)
  D = Diagnostic(BarotropicQG.energy_dissipation, prob, nsteps=nt)
  R = Diagnostic(BarotropicQG.energy_drag,        prob, nsteps=nt)
  W = Diagnostic(BarotropicQG.energy_work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  stepforward!(prob, diags, nt)

  BarotropicQG.updatevars!(prob)

  dEdt_numerical = (E[2:E.i] - E[1:E.i-1]) / prob.clock.dt

  # If the Ito interpretation was used for the work
  # then we need to add the drift term
  # dEdt_computed = W[2:E.i] + ε - D[1:E.i-1] - R[1:E.i-1]      # Ito
  dEdt_computed = W[2:E.i] - D[1:E.i-1] - R[1:E.i-1]        # Stratonovich

  return isapprox(dEdt_numerical, dEdt_computed, rtol=1e-3)
end

"""
    test_bqg_deterministicforcing_energy_budgets(dev; kwargs...)

Tests if the energy budget is closed for a BarotropicQG problem with deterministic forcing.
"""
function test_bqg_deterministicforcing_energybudget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ = 1e-1
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev, n, L)
  x, y = gridpoints(grid)
  k₀, l₀ = 2π/grid.Lx, 2π/grid.Ly

  # Forcing = 0.01cos(4x)cos(5y)cos(2t)
  f = @. 0.01 * cos(4k₀*x) * cos(5l₀*y)
  fh = rfft(f)
  
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    @. Fh = fh * cos(2t)
    return nothing
  end

  prob = BarotropicQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=false)

  BarotropicQG.set_zeta!(prob, 0*x)
  
  E = Diagnostic(BarotropicQG.energy,             prob, nsteps=nt)
  D = Diagnostic(BarotropicQG.energy_dissipation, prob, nsteps=nt)
  R = Diagnostic(BarotropicQG.energy_drag,        prob, nsteps=nt)
  W = Diagnostic(BarotropicQG.energy_work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  stepforward!(prob, diags, round(Int, nt))

  BarotropicQG.updatevars!(prob)

  dEdt_numerical = (E[3:E.i] - E[1:E.i-2]) / (2 * prob.clock.dt)
  dEdt_computed  = W[2:E.i-1] - D[2:E.i-1] - R[2:E.i-1]
  
  return isapprox(dEdt_numerical, dEdt_computed, rtol=1e-4)
end

"""
    test_bqg_stochasticforcing_enstrophy_budgets(dev; kwargs...)

Tests if the enstrophy budget is closed for a BarotropicQG problem with stochastic forcing.
"""
function test_bqg_stochasticforcing_enstrophybudget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1, T=Float64)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ = 1e-1
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev, n, L)
  x, y = gridpoints(grid)
  K  = @. sqrt(grid.Krsq)                      # a 2D array with the total wavenumber

  # Forcing with spectrum ~ exp(-(k-kf)²/(2 dkf²))
  kf, dkf = 12.0, 2.0
  εᶻ = 0.1
  
  forcing_spectrum = zeros(dev, T, (grid.nkr, grid.nl))
  @. forcing_spectrum = exp(-(K - kf)^2 / (2 * dkf^2))
  @. forcing_spectrum = ifelse(grid.Krsq < 2^2,  0, forcing_spectrum)   # no power at low wavenumbers
  @. forcing_spectrum = ifelse(grid.Krsq > 20^2, 0, forcing_spectrum)   # no power at high wavenumbers
  εᶻ0 = parsevalsum(forcing_spectrum / 2, grid) / (grid.Lx * grid.Ly)
  forcing_spectrum .= εᶻ / εᶻ0 * forcing_spectrum

  Random.seed!(1234)

  function calcF!(Fh, sol, t, clock, vars, params, grid)
    eta = ArrayType(dev)(exp.(2π * im * rand(T, size(sol))) / sqrt(clock.dt))
    CUDA.@allowscalar eta[1, 1] = 0
    @. Fh = eta * sqrt(forcing_spectrum)
    return nothing
  end

  prob = BarotropicQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=true)

  BarotropicQG.set_zeta!(prob, 0*x)
  
  Z = Diagnostic(BarotropicQG.enstrophy,             prob, nsteps=nt)
  D = Diagnostic(BarotropicQG.enstrophy_dissipation, prob, nsteps=nt)
  R = Diagnostic(BarotropicQG.enstrophy_drag,        prob, nsteps=nt)
  W = Diagnostic(BarotropicQG.enstrophy_work,        prob, nsteps=nt)
  diags = [Z, D, W, R]

  stepforward!(prob, diags, nt)

  BarotropicQG.updatevars!(prob)

  dZdt_numerical = (Z[2:Z.i] - Z[1:Z.i-1]) / prob.clock.dt

  # If the Ito interpretation was used for the work
  # then we need to add the drift term
  # dZdt_computed = W[2:Z.i] + εᶻ - D[1:Z.i-1] - R[1:Z.i-1]      # Ito
  dZdt_computed = W[2:Z.i] - D[1:Z.i-1] - R[1:Z.i-1]        # Stratonovich

  return isapprox(dZdt_numerical, dZdt_computed, rtol=1e-3)
end

"""
    test_bqg_deterministicforcing_enstrophy_budgets(dev; kwargs...)

Tests if the enstrophy budget is closed for a BarotropicQG problem with deterministic forcing.
"""
function test_bqg_deterministicforcing_enstrophybudget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ = 1e-1
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev, n, L)
  x, y = gridpoints(grid)
  k₀, l₀ = 2π/grid.Lx, 2π/grid.Ly

  # Forcing = 0.01cos(4x)cos(5y)cos(2t)
  f = @. 0.01 * cos(4k₀*x) * cos(5l₀*y)
  fh = rfft(f)
  
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    @. Fh = fh * cos(2t)
    return nothing
  end

  prob = BarotropicQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=false)

  BarotropicQG.set_zeta!(prob, 0*x)
  
  Z = Diagnostic(BarotropicQG.enstrophy,             prob, nsteps=nt)
  D = Diagnostic(BarotropicQG.enstrophy_dissipation, prob, nsteps=nt)
  R = Diagnostic(BarotropicQG.enstrophy_drag,        prob, nsteps=nt)
  W = Diagnostic(BarotropicQG.enstrophy_work,        prob, nsteps=nt)
  diags = [Z, D, W, R]

  stepforward!(prob, diags, round(Int, nt))

  BarotropicQG.updatevars!(prob)

  dZdt_numerical = (Z[3:Z.i] - Z[1:Z.i-2]) / (2 * prob.clock.dt)
  dZdt_computed  = W[2:Z.i-1] - D[2:Z.i-1] - R[2:Z.i-1]
  
  return isapprox(dZdt_numerical, dZdt_computed, rtol=1e-4)
end

"""
    test_bqg_nonlinearadvection(dt, stepper, dev; kwargs...)

Tests the advection term in the BarotropicQG module by timestepping a
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

  grid = TwoDGrid(dev, n, L)
  x, y = gridpoints(grid)

  ψf = @. sin(2x) * cos(2y) + 2sin(x) * cos(3y)
  qf = @. -8sin(2x) * cos(2y) - 20sin(x) * cos(3y)

  Ff = @. -(
    ν*( 64sin(2x) * cos(2y) + 200sin(x) * cos(3y) )
    + 8 * ( cos(x) * cos(3y) * sin(2x) * sin(2y) - 3cos(2x) * cos(2y) * sin(x) * sin(3y) )
    )

  Ffh = rfft(Ff)

  function calcF!(Fh, sol, t, cl, v, p, g)
    Fh .= Ffh
    return nothing
  end

  prob = BarotropicQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt, stepper=stepper, calcF=calcF!)

  BarotropicQG.set_zeta!(prob, qf)

  stepforward!(prob, round(Int, nt))

  BarotropicQG.updatevars!(prob)
  
  return isapprox(prob.vars.q, qf, rtol=rtol_barotropicQG)
end

"""
    test_bqg_energyenstrophy(dev)

Tests the energy and enstrophy function for a BarotropicQG problem.
"""
function test_bqg_energyenstrophy(dev::Device=CPU())
  nx, Lx  = 64, 2π
  ny, Ly  = 64, 3π
  grid = TwoDGrid(dev, nx, Lx, ny, Ly)
  k₀, l₀ = 2π/Lx, 2π/Ly # fundamental wavenumbers
  x, y = gridpoints(grid)

  energy_calc = 29/9
  enstrophy_calc = 2701/162

    eta = @. cos(10k₀*x) * cos(10l₀*y)
   psi0 = @. sin(2k₀*x) * cos(2l₀*y) + 2sin(k₀*x) * cos(3l₀*y)
  zeta0 = @. - ((2k₀)^2+(2l₀)^2) * sin(2k₀*x) * cos(2l₀*y) - (k₀^2+(3l₀)^2) * 2sin(k₀*x) * cos(3l₀*y)

  prob = BarotropicQG.Problem(dev; nx=nx, Lx=Lx, ny=ny, Ly=Ly, eta = eta, stepper="ForwardEuler")
  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
  BarotropicQG.set_zeta!(prob, zeta0)
  BarotropicQG.updatevars!(prob)

  energyzeta0 = BarotropicQG.energy(prob)
  enstrophyzeta0 = BarotropicQG.enstrophy(prob)

  return isapprox(energyzeta0, energy_calc, rtol=rtol_barotropicQG) && isapprox(enstrophyzeta0, enstrophy_calc, rtol=rtol_barotropicQG) &&
  BarotropicQG.addforcing!(prob.timestepper.N, sol, cl.t, cl, v, p, g)==nothing
end

"""
    test_bqg_problemtype(dev, T)

Tests the BarotropicQG problem constructor for different DataType `T`.
"""
function test_bqg_problemtype(dev, T)
  prob = BarotropicQG.Problem(dev; T=T)

  A = ArrayType(dev)
  
  return (typeof(prob.sol)<:A{Complex{T}, 2} && typeof(prob.grid.Lx)==T && eltype(prob.grid.x)==T && typeof(prob.vars.u)<:A{T, 2})
end
