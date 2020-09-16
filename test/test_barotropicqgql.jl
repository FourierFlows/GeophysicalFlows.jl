"""
    test_bqgql_rossbywave(; kwargs...)

Evolvesa a Rossby wave and compares with the analytic solution.
"""
function test_bqgql_rossbywave(stepper, dt, nsteps, dev::Device=CPU())
  nx = 64
   β = 2.0
  Lx = 2π
   μ = 0.0
   ν = 0.0
   T = Float64

  # the following if statement is called so that all the cases of
  # Problem() function are tested
  if stepper=="ForwardEuler"
    eta = zeros(dev, T, (nx, nx))
  else
    eta(x, y) = 0*x
  end

  prob = BarotropicQGQL.Problem(dev; nx=nx, Lx=Lx, eta=eta, β=β, μ=μ, ν=ν, stepper=stepper, dt=dt)
  sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  x, y = gridpoints(grid)

  # the Rossby wave initial condition
   ampl = 1e-2
  kwave, lwave = 3 * 2π / grid.Lx, 2 * 2π / grid.Ly
      ω = - params.β * kwave / (kwave^2 + lwave^2)
     ζ0 = @. ampl * cos(kwave * x) * cos(lwave * y)
    ζ0h = rfft(ζ0)

  BarotropicQGQL.set_zeta!(prob, ζ0)

  stepforward!(prob, nsteps)
  dealias!(sol, grid)
  BarotropicQGQL.updatevars!(prob)

  ζ_theory = @. ampl * cos(kwave * (x - ω / kwave * clock.t)) * cos(lwave * y)

  return isapprox(ζ_theory, vars.zeta, rtol = grid.nx * grid.ny * nsteps * 1e-12)
end

"""
    test_stochasticforcingbudgets(dev; kwargs...)

Tests if the energy budget is closed for BarotropicQG problem with stochastic forcing.
"""
function test_bqgql_stochasticforcingbudgets(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1, T=Float64)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ = 1e-1
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  gr = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

  # Forcing
  kf, dkf = 12.0, 2.0
  ε = 0.1
  
  CUDA.@allowscalar Kr = ArrayType(dev)([ gr.kr[i] for i=1:gr.nkr, j=1:gr.nl])

  forcingcovariancespectrum = zeros(dev, T, (gr.nkr, gr.nl))
  @. forcingcovariancespectrum = exp(-(sqrt(gr.Krsq) - kf)^2 / (2 * dkf^2))
  CUDA.@allowscalar @. forcingcovariancespectrum[gr.Krsq .< 2^2] = 0
  CUDA.@allowscalar @. forcingcovariancespectrum[gr.Krsq .> 20^2] = 0
  CUDA.@allowscalar @. forcingcovariancespectrum[Kr .< 2π/L] = 0
  ε0 = parsevalsum(forcingcovariancespectrum .* gr.invKrsq / 2, gr) / (gr.Lx * gr.Ly)
  forcingcovariancespectrum .= ε / ε0 * forcingcovariancespectrum

  Random.seed!(1234)

  function calcF!(F, sol, t, clock, vars, params, grid)
    eta = ArrayType(dev)(exp.(2π * im * rand(T, size(sol))) / sqrt(clock.dt))
    CUDA.@allowscalar eta[1, 1] = 0
    @. F = eta * sqrt(forcingcovariancespectrum)
    return nothing
  end

  prob = BarotropicQGQL.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=true)

  BarotropicQGQL.set_zeta!(prob, 0*x)
  E = Diagnostic(BarotropicQGQL.energy,      prob, nsteps=nt)
  D = Diagnostic(BarotropicQGQL.dissipation, prob, nsteps=nt)
  R = Diagnostic(BarotropicQGQL.drag,        prob, nsteps=nt)
  W = Diagnostic(BarotropicQGQL.work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  stepforward!(prob, diags, round(Int, nt))

  BarotropicQGQL.updatevars!(prob)
  
  dEdt_numerical = (E[2:E.i] - E[1:E.i-1]) / prob.clock.dt

  # If the Ito interpretation was used for the work
  # then we need to add the drift term
  # dEdt_computed = W[2:E.i] + ε - D[1:E.i-1] - R[1:E.i-1]      # Ito
  dEdt_computed = W[2:E.i] - D[1:E.i-1] - R[1:E.i-1]        # Stratonovich

  return isapprox(dEdt_numerical, dEdt_computed, rtol=1e-3)
end

"""
    test_bqgql_deterministicforcingbudgets(dev ; kwargs...)

Tests if the energy budget is closed for BarotropicQGQL problem with deterministic forcing.
"""
function test_bqgql_deterministicforcingbudgets(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ = 1e-1
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  gr = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)
  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly

  # Forcing = 0.01cos(4x)cos(5y)cos(2t)
  f = @. 0.01 * cos(4k₀*x) * cos(5l₀*y)
  fh = rfft(f)

  function calcF!(Fh, sol, t, clock, vars, params, grid)
    @. Fh = fh*cos(2t)
    return nothing
  end

  prob = BarotropicQGQL.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=false)

  BarotropicQGQL.set_zeta!(prob, 0*x)
  
  E = Diagnostic(BarotropicQGQL.energy,      prob, nsteps=nt)
  D = Diagnostic(BarotropicQGQL.dissipation, prob, nsteps=nt)
  R = Diagnostic(BarotropicQGQL.drag,        prob, nsteps=nt)
  W = Diagnostic(BarotropicQGQL.work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  stepforward!(prob, diags, nt)

  BarotropicQGQL.updatevars!(prob)

  dEdt_numerical = (E[3:E.i] - E[1:E.i-2]) / (2 * prob.clock.dt)
  dEdt_computed  = W[2:E.i-1] - D[2:E.i-1] - R[2:E.i-1]
  
  residual = dEdt_numerical - dEdt_computed

  return isapprox(dEdt_numerical, dEdt_computed, atol=1e-10)
end

"""
    test_bqgql_nonlinearadvection(dt, stepper, dev; kwargs...)

Tests the advection term by timestepping a test problem with timestep `dt` and timestepper 
`stepper`. The test problem is derived by picking a solution ζf (with associated
streamfunction ψf) for which the advection term J(ψf, ζf) is non-zero. Next, a
forcing Ff is derived according to Ff = ∂ζf/∂t + J(ψf, ζf) - νΔζf. One solution
to the vorticity equation forced by this Ff is then ζf. (This solution may not
be realized, at least at long times, if it is unstable.)
"""
function test_bqgql_advection(dt, stepper, dev::Device=CPU(); n=128, L=2π, ν=1e-2, nν=1, μ=0.0)
  n, L  = 128, 2π
  ν, nν = 1e-2, 1
   μ = 0.0
  tf = 1.0
  nt = round(Int, tf/dt)

  gr = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

  ψf = @.    cos(3y) +  sin(2x)*cos(2y) +  2sin(x)*cos(3y)
  qf = @. - 9cos(3y) - 8sin(2x)*cos(2y) - 20sin(x)*cos(3y)

  Ff = @. ν*( 81cos(3y) + 200cos(3y)*sin(x) + 64cos(2y)*sin(2x) ) -
    3sin(3y)*(-16cos(2x)*cos(2y) - 20cos(x)*cos(3y)) -
 27sin(3y)*(2cos(2x)*cos(2y) + 2cos(x)*cos(3y)) + 0*(-8cos(x)*cos(3y)*sin(2x)*sin(2y) +
 24*cos(2x)*cos(2y)*sin(x)*sin(3y))

  Ffh = -rfft(Ff)

  # Forcing
  function calcF!(Fh, sol, t, cl, v, p, g)
    Fh .= Ffh
    return nothing
  end

  prob = BarotropicQGQL.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt, stepper=stepper, calcF=calcF!)
  
  BarotropicQGQL.set_zeta!(prob, qf)

  stepforward!(prob, round(Int, nt))
  
  BarotropicQGQL.updatevars!(prob)
  
  return isapprox(prob.vars.zeta + prob.vars.Zeta, qf, rtol = 1e-13)
end

"""
    test_bqgql_energyenstrophy(dev)

Tests the energy and enstrophy function for a BarotropicQGQL problem.
"""
function test_bqgql_energyenstrophy(dev::Device=CPU())
  nx, Lx  = 64, 2π
  ny, Ly  = 64, 3π
  gr = TwoDGrid(dev, nx, Lx, ny, Ly)
  k₀, l₀ = 2π/Lx, 2π/Ly # fundamental wavenumbers
  x, y = gridpoints(gr)

  energy_calc = 29/9
  enstrophy_calc = 2701/162

    eta = @. cos(10k₀*x)*cos(10l₀*y)
  ψ₀ = @. sin(2k₀*x)*cos(2l₀*y) + 2sin(k₀*x)*cos(3l₀*y)
  ζ₀ = @. -((2k₀)^2+(2l₀)^2)*sin(2k₀*x)*cos(2l₀*y) - (k₀^2+(3l₀)^2)*2sin(k₀*x)*cos(3l₀*y)

  prob = BarotropicQGQL.Problem(dev; nx=nx, Lx=Lx, ny=ny, Ly=Ly, eta=eta, stepper="ForwardEuler")
  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
  
  BarotropicQGQL.set_zeta!(prob, ζ₀)
  BarotropicQGQL.updatevars!(prob)

  energyζ₀ = BarotropicQGQL.energy(prob)
  enstrophyζ₀ = BarotropicQGQL.enstrophy(prob)

  return isapprox(energyζ₀, energy_calc, rtol=1e-13) && isapprox(enstrophyζ₀, enstrophy_calc, rtol=1e-13) && BarotropicQGQL.addforcing!(prob.timestepper.N, sol, cl.t, cl, v, p, g)==nothing
end

function test_bqgql_problemtype(dev, T)
  prob = BarotropicQGQL.Problem(dev; T=T)
  
  A = ArrayType(dev)
  
  return (typeof(prob.sol)<:A{Complex{T}, 2} && typeof(prob.grid.Lx)==T && eltype(prob.grid.x)==T && typeof(prob.vars.u)<:A{T, 2})
end
