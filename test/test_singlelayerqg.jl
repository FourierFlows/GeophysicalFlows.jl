using LinearAlgebra

"""
    test_1layerqg_rossbywave(stepper, dt, nsteps, dev; kwargs...)

Evolves a Rossby wave and compares with the analytic solution.
"""
function test_1layerqg_rossbywave(stepper, dt, nsteps, dev::Device=CPU(), nx=64; deformation_radius=Inf, U = 0)

  Lx = 2π
   β = 2.0
   μ = 0.0
   ν = 0.0
   T = Float64

  # the following if statement is called so that all the cases of
  # Problem() fuction are tested
  if stepper == "ForwardEuler"
    eta = zeros(dev, T, (nx, nx))
  else
    eta(x, y) = 0 * x
  end

  prob = SingleLayerQG.Problem(dev; nx=nx, Lx=Lx, eta=eta, deformation_radius=deformation_radius, β=β, U=U, μ=μ, ν=ν, stepper=stepper, dt=dt)
  sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  x, y = gridpoints(grid)

  # the Rossby wave initial condition
   ampl = 1e-2
  kwave = 3 * 2π / grid.Lx
  lwave = 2 * 2π / grid.Ly
      ω = -params.β * kwave / (kwave^2 + lwave^2 + 1 / deformation_radius^2)
    q₀  = @. ampl * cos(kwave * x) * cos(lwave * y)
    q₀h = rfft(q₀)

  SingleLayerQG.set_q!(prob, q₀)

  stepforward!(prob, nsteps)
  dealias!(sol, grid)
  SingleLayerQG.updatevars!(prob)

  q_theory = @CUDA.allowscalar @. ampl * cos(kwave * (x - (U[1] + ω / kwave) * clock.t)) * cos(lwave * y)

  return isapprox(q_theory, vars.q, rtol=grid.nx * grid.ny * nsteps * 1e-12)
end

"""
    test_1layerqg_stochasticforcing_energy_budgets(dev; kwargs...)

Tests if the energy budget is closed for a SingleLayerQG problem with stochastic forcing.
"""
function test_1layerqg_stochasticforcing_energybudget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1, T=Float64)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ = 1e-1
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev; nx=n, Lx=L)
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
  @. forcing_spectrum = ε / ε0 * forcing_spectrum

  Random.seed!(1234)

  function calcF!(Fh, sol, t, clock, vars, params, grid)
    eta = device_array(dev)(exp.(2π * im * rand(T, size(sol))) / sqrt(clock.dt))
    CUDA.@allowscalar eta[1, 1] = 0
    @. Fh = eta * sqrt(forcing_spectrum)
    
    return nothing
  end

  prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=true)

  SingleLayerQG.set_q!(prob, 0*x)
  
  E = Diagnostic(SingleLayerQG.kinetic_energy,     prob, nsteps=nt)
  D = Diagnostic(SingleLayerQG.energy_dissipation, prob, nsteps=nt)
  R = Diagnostic(SingleLayerQG.energy_drag,        prob, nsteps=nt)
  W = Diagnostic(SingleLayerQG.energy_work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  stepforward!(prob, diags, nt)

  SingleLayerQG.updatevars!(prob)

  dEdt_numerical = (E[2:E.i] - E[1:E.i-1]) / prob.clock.dt

  dEdt_computed = W[2:E.i] - D[1:E.i-1] - R[1:E.i-1]

  return isapprox(dEdt_numerical, dEdt_computed, rtol=1e-3)
end

"""
    test_1layerqg_deterministicforcing_energy_budgets(dev; kwargs...)

Tests if the energy budget is closed for a SingleLayerQG problem with deterministic forcing.
"""
function test_1layerqg_deterministicforcing_energybudget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ = 1e-1
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev; nx=n, Lx=L)
  x, y = gridpoints(grid)
  k₀, l₀ = 2π / grid.Lx, 2π/grid.Ly

  # Forcing = 0.01cos(4x)cos(5y)cos(2t)
  f = @. 0.01 * cos(4k₀*x) * cos(5l₀*y)
  fh = rfft(f)
  
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    @. Fh = fh * cos(2t)
    
    return nothing
  end

  prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=false)

  SingleLayerQG.set_q!(prob, 0*x)
  
  E = Diagnostic(SingleLayerQG.kinetic_energy,     prob, nsteps=nt)
  D = Diagnostic(SingleLayerQG.energy_dissipation, prob, nsteps=nt)
  R = Diagnostic(SingleLayerQG.energy_drag,        prob, nsteps=nt)
  W = Diagnostic(SingleLayerQG.energy_work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  stepforward!(prob, diags, round(Int, nt))

  SingleLayerQG.updatevars!(prob)

  dEdt_numerical = (E[3:E.i] - E[1:E.i-2]) / (2 * prob.clock.dt)
  dEdt_computed  = W[2:E.i-1] - D[2:E.i-1] - R[2:E.i-1]
  
  return isapprox(dEdt_numerical, dEdt_computed, rtol=1e-4)
end

"""
    test_1layerqg_stochasticforcing_enstrophy_budgets(dev; kwargs...)

Tests if the enstrophy budget is closed for a SingleLayerQG problem with stochastic forcing.
"""
function test_1layerqg_stochasticforcing_enstrophybudget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1, T=Float64)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ = 1e-1
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev; nx=n, Lx=L)
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
    eta = device_array(dev)(exp.(2π * im * rand(T, size(sol))) / sqrt(clock.dt))
    CUDA.@allowscalar eta[1, 1] = 0
    @. Fh = eta * sqrt(forcing_spectrum)
    
    return nothing
  end

  prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=true)

  SingleLayerQG.set_q!(prob, 0*x)
  
  Z = Diagnostic(SingleLayerQG.enstrophy,             prob, nsteps=nt)
  D = Diagnostic(SingleLayerQG.enstrophy_dissipation, prob, nsteps=nt)
  R = Diagnostic(SingleLayerQG.enstrophy_drag,        prob, nsteps=nt)
  W = Diagnostic(SingleLayerQG.enstrophy_work,        prob, nsteps=nt)
  diags = [Z, D, W, R]

  stepforward!(prob, diags, nt)

  SingleLayerQG.updatevars!(prob)

  dZdt_numerical = (Z[2:Z.i] - Z[1:Z.i-1]) / prob.clock.dt

  dZdt_computed = W[2:Z.i] - D[1:Z.i-1] - R[1:Z.i-1]

  return isapprox(dZdt_numerical, dZdt_computed, rtol=1e-3)
end

"""
    test_1layerqg_deterministicforcing_enstrophy_budgets(dev; kwargs...)

Tests if the enstrophy budget is closed for a SingleLayerQG problem with deterministic forcing.
"""
function test_1layerqg_deterministicforcing_enstrophybudget(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ = 1e-1
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev; nx=n, Lx=L)
  x, y = gridpoints(grid)
  k₀, l₀ = 2π/grid.Lx, 2π/grid.Ly

  # Forcing = 0.01cos(4x)cos(5y)cos(2t)
  f = @. 0.01 * cos(4k₀ * x) * cos(5l₀ * y)
  fh = rfft(f)
  
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    @. Fh = fh * cos(2t)
    return nothing
  end

  prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=false)

  SingleLayerQG.set_q!(prob, 0*x)
  
  Z = Diagnostic(SingleLayerQG.enstrophy,             prob, nsteps=nt)
  D = Diagnostic(SingleLayerQG.enstrophy_dissipation, prob, nsteps=nt)
  R = Diagnostic(SingleLayerQG.enstrophy_drag,        prob, nsteps=nt)
  W = Diagnostic(SingleLayerQG.enstrophy_work,        prob, nsteps=nt)
  diags = [Z, D, W, R]

  stepforward!(prob, diags, round(Int, nt))

  SingleLayerQG.updatevars!(prob)

  dZdt_numerical = (Z[3:Z.i] - Z[1:Z.i-2]) / (2 * prob.clock.dt)
  dZdt_computed  = W[2:Z.i-1] - D[2:Z.i-1] - R[2:Z.i-1]
  
  return isapprox(dZdt_numerical, dZdt_computed, rtol=1e-4)
end

"""
    test_1layerqg_nonlinearadvection(dt, stepper, dev::Device=CPU();
                                     add_topography = false,
                                     add_background_flow = false,
                                     background_flow_vary_in_y = true)

Tests the advection term in the `SingleLayerQG` module by timestepping a
test problem with timestep dt and timestepper identified by the string stepper.
The test problem is derived by picking a solution qf (with associated
streamfunction ψf) for which the advection term J(qf, ζf) is non-zero. Next, a
forcing Ff is derived according to Ff = ∂qf/∂t + J(ψf, qf) - ν∇²qf. One solution
to the vorticity equation forced by this Ff is then qf. (This solution may not
be realized, at least at long times, if it is unstable.)

We can optionally add an imposed mean flow `U(y)` via the kwarg `add_background_flow = true`
or add topography via kwarg `add_topography = true`.
"""
function test_1layerqg_nonlinearadvection(dt, stepper, dev::Device=CPU();
                                          add_topography = false,
                                          add_background_flow = false,
                                          background_flow_vary_in_y = true)
  n, L  = 128, 2π
  ν, nν = 1e-2, 1
   μ = 0.0
  tf = 1.0
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev; nx=n, Lx=L)
  x, y = gridpoints(grid)

  if add_topography == true
    η₀ = 0.4
  elseif add_topography == false
    η₀ = 0
  end

  η(x, y) = η₀ * cos(10x) * cos(10y)

  ψf = @. sin(2x) * cos(2y) + 2sin(x) * cos(3y)
  qf = @. -8sin(2x) * cos(2y) - 20sin(x) * cos(3y)

  # F = J(ψ, q + η) - ν ∇²q + U ∂(q+η)/∂x - v ∂²U/∂y²
  # where q = ∇²ψ
  Ff = @. (- ν * ( 64sin(2x) * cos(2y) + 200sin(x) * cos(3y) )
           - 8 * (cos(x) * cos(3y) * sin(2x) * sin(2y) - 3cos(2x) * cos(2y) * sin(x) * sin(3y))
           - 20η₀ * (sin(10x) * cos(10y) * (sin(2x) * sin(2y) + 3sin(x) * sin(3y))
                     + cos(10x) * sin(10y) * (cos(2x) * cos(2y) + cos(x) * cos(3y))))

  U_amplitude = 0.2

  U = 0
  Uyy = 0

  if add_background_flow == true && background_flow_vary_in_y == false
    U = U_amplitude
  elseif add_background_flow == true && background_flow_vary_in_y == true
    U = @. U_amplitude / cosh(2y)^2
    U = device_array(dev)(U)
    Uh = rfft(U)
    Uyy = irfft(- grid.l.^2 .* Uh, grid.nx)   # ∂²U/∂y²
  end

  @. Ff += (- U * (16cos(2x) * cos(2y) + 20cos(x) * cos(3y))
            - Uyy * (2cos(2x) * cos(2y) +  2cos(x) * cos(3y))
            - η₀ * U * 10sin(10x) * cos(10y))

  Ffh = rfft(Ff)

  function calcF!(Fh, sol, t, clock, vars, params, grid)
    Fh .= Ffh
    return nothing
  end

  if add_background_flow == false
    U₀ = 0
  elseif add_background_flow == true && background_flow_vary_in_y == true
    U₀ = U[1, :]
  elseif add_background_flow == true && background_flow_vary_in_y == false
    U₀ = U_amplitude
  end

  prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, U = U₀, eta = η, ν=ν, nν=nν, μ=μ, dt=dt, stepper=stepper, calcF=calcF!)

  SingleLayerQG.set_q!(prob, qf)

  stepforward!(prob, round(Int, nt))

  SingleLayerQG.updatevars!(prob)

  return isapprox(prob.vars.q, qf, rtol=rtol_singlelayerqg)
end

"""
    test_1layerqg_nonlinearadvection_deformation(dt, stepper, dev::Device=CPU();
                                                 deformation_radius=1.23)

Same as `test_1layerqg_nonlinearadvection` but with finite deformation radius.
"""
function test_1layerqg_nonlinearadvection_deformation(dt, stepper, dev::Device=CPU();
                                                      deformation_radius=1.23)
  n, L  = 128, 2π
  ν, nν = 1e-2, 1
   μ = 0.0
  tf = 1.0
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev; nx=n, Lx=L)
  k₀, l₀ = 2π / grid.Lx, 2π / grid.Ly # fundamental wavenumbers
  x, y = gridpoints(grid)

  η₀ = 0.4
  η(x, y) = η₀ * cos(10x) * cos(10y)
  ψf = @. sin(2x) * cos(2y) + 2sin(x) * cos(3y)
  qf = @. - (2^2 + 2^2) * sin(2x) * cos(2y) - (1^2 + 3^2) * 2sin(x) * cos(3y) - 1/deformation_radius^2 * ψf

  # F = J(ψ, q + η) - ν ∇²q
  # where q = ∇²ψ - ψ/ℓ²
  Ff = @. (- ν * (64sin(2x) * cos(2y) + 200sin(x) * cos(3y)
                  + (8sin(2x) * cos(2y) + 20sin(x) * cos(3y)) / deformation_radius^2)
           - 8 * (cos(x) * cos(3y) * sin(2x) * sin(2y) - 3cos(2x) * cos(2y) * sin(x) * sin(3y))
           - 20η₀ * (cos(10y) * sin(10x) * (sin(2x) * sin(2y) + 3sin(x) * sin(3y))
                     + cos(10x) * sin(10y) * (cos(2x) * cos(2y) + cos(x) * cos(3y))))

  Ffh = rfft(Ff)

  function calcF!(Fh, sol, t, clock, vars, params, grid)
    Fh .= Ffh
    return nothing
  end

  prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, eta=η, ν=ν, nν=nν, μ=μ, dt=dt, deformation_radius=deformation_radius, stepper=stepper, calcF=calcF!)

  SingleLayerQG.set_q!(prob, qf)

  stepforward!(prob, round(Int, nt))

  SingleLayerQG.updatevars!(prob)

  return isapprox(prob.vars.q, qf, rtol=rtol_singlelayerqg)
end

"""
    test_1layerqg_energyenstrophy_BarotropicQG(dev)

Tests the energy and enstrophy function for a SingleLayerQG problem.
"""
function test_1layerqg_energyenstrophy_BarotropicQG(dev::Device=CPU())
  nx, Lx  = 64, 2π
  ny, Ly  = 64, 3π

  prob = SingleLayerQG.Problem(dev; nx=nx, Lx=Lx, ny=ny, Ly=Ly, stepper="ForwardEuler")
  grid = prob.grid

  k₀, l₀ = 2π / grid.Lx, 2π / grid.Ly # fundamental wavenumbers
  x, y = gridpoints(grid)

  energy_calc = 29/9
  enstrophy_calc = 10885/648

  η(x, y) = cos(10k₀ * x) * cos(10l₀ * y)
  ψ₀ = @. sin(2k₀ * x) * cos(2l₀ * y) + 2sin(k₀ * x) * cos(3l₀ * y)
  q₀ = @. - ((2k₀)^2 + (2l₀)^2) * sin(2k₀ * x) * cos(2l₀ * y) - (k₀^2 + (3l₀)^2) * 2sin(k₀ * x) * cos(3l₀*y)

  prob = SingleLayerQG.Problem(dev; nx=nx, Lx=Lx, ny=ny, Ly=Ly, eta=η, stepper="ForwardEuler")
  sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  SingleLayerQG.set_q!(prob, q₀)
  SingleLayerQG.updatevars!(prob)

  energyq₀ = SingleLayerQG.energy(prob)
  enstrophyq₀ = SingleLayerQG.enstrophy(prob)

  return (isapprox(energyq₀, energy_calc, rtol=rtol_singlelayerqg) &&
          isapprox(enstrophyq₀, enstrophy_calc, rtol=rtol_singlelayerqg) &&
          SingleLayerQG.potential_energy(prob)==0 &&
          SingleLayerQG.addforcing!(prob.timestepper.N, sol, clock.t, clock, vars, params, grid) === nothing)
end

"""
    test_1layerqg_energies_EquivalentBarotropicQG(dev; deformation_radius=1.23)

Tests the kinetic and potential energy for an equivalent barotropic SingleLayerQG problem.
"""
function test_1layerqg_energies_EquivalentBarotropicQG(dev; deformation_radius=1.23)
  nx, Lx  = 64, 2π
  ny, Ly  = 64, 3π

  prob = SingleLayerQG.Problem(dev; nx, Lx, ny, Ly, deformation_radius, stepper="ForwardEuler")
  grid = prob.grid

  k₀, l₀ = 2π / grid.Lx, 2π / grid.Ly # fundamental wavenumbers
  x, y = gridpoints(grid)

  kinetic_energy_calc = 29/9
  potential_energy_calc = 5/(8*deformation_radius^2)
  energy_calc = kinetic_energy_calc + potential_energy_calc

  η(x, y) = cos(10k₀ * x) * cos(10l₀ * y)
  ψ₀ = @. sin(2k₀ * x) * cos(2l₀ * y) + 2sin(k₀ * x) * cos(3l₀ * y)
  q₀ = @. - ((2k₀)^2 + (2l₀)^2) * sin(2k₀ * x) * cos(2l₀ * y) - (k₀^2 + (3l₀)^2) * 2sin(k₀ * x) * cos(3l₀*y) - 1/deformation_radius^2 * ψ₀

  prob = SingleLayerQG.Problem(dev; nx, Lx, ny, Ly, eta=η, deformation_radius, stepper="ForwardEuler")
  sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
  SingleLayerQG.set_q!(prob, q₀)
  SingleLayerQG.updatevars!(prob)

  kinetic_energyq₀ = SingleLayerQG.kinetic_energy(prob)
  potential_energyq₀ = SingleLayerQG.potential_energy(prob)
  energyq₀ = SingleLayerQG.energy(prob)

  return (isapprox(kinetic_energyq₀, kinetic_energy_calc, rtol=rtol_singlelayerqg) &&
          isapprox(potential_energyq₀, potential_energy_calc, rtol=rtol_singlelayerqg) &&
          isapprox(energyq₀, energy_calc, rtol=rtol_singlelayerqg) &&
          SingleLayerQG.addforcing!(prob.timestepper.N, sol, clock.t, clock, vars, params, grid) === nothing)
end

"""
    test_1layerqg_problemtype(dev, T; deformation_radius=Inf, U=0)

Test the SingleLayerQG problem constructor for different DataType `T`.
"""
function test_1layerqg_problemtype(dev, T; deformation_radius=Inf, U=0)
  prob = SingleLayerQG.Problem(dev; T, deformation_radius, U)

  A = device_array(dev)

  return (typeof(prob.sol)<:A{Complex{T}, 2} && typeof(prob.grid.Lx)==T &&
          eltype(prob.grid.x)==T && typeof(prob.vars.u)<:A{T, 2} && typeof(prob.params.U)==T)
end

function test_streamfunctionfrompv(dev; deformation_radius=1.23)
  prob_barotropicQG = SingleLayerQG.Problem(dev; nx=64, deformation_radius=Inf)
  prob_equivalentbarotropicQG = SingleLayerQG.Problem(dev; nx=64, deformation_radius=deformation_radius)
  
  grid = prob_barotropicQG.grid
  k₀, l₀ = 2π/grid.Lx, 2π/grid.Ly # fundamental wavenumbers
  x, y = gridpoints(grid)
  
  ψ = @. sin(2k₀ * x) * cos(3l₀ * y) + 0*sin(3k₀ * x)
     
  q_barotropic = @. -((2k₀)^2 + (3l₀)^2) * sin(2k₀ * x) * cos(3l₀ * y) - 0*(3k₀)^2 * sin(3k₀ * x)
  q_equivalentbarotropic = @. -((2k₀)^2 + (3l₀)^2 + 1/deformation_radius^2) * sin(2k₀ * x) * cos(3l₀ * y) - 0*((3k₀)^2+ 1/deformation_radius^2) * sin(3k₀ * x)
  
  SingleLayerQG.set_q!(prob_barotropicQG, q_barotropic)
  SingleLayerQG.set_q!(prob_equivalentbarotropicQG, q_equivalentbarotropic)
  
  SingleLayerQG.streamfunctionfrompv!(prob_barotropicQG.vars.ψh, prob_barotropicQG.vars.qh, prob_barotropicQG.params, prob_barotropicQG.grid)
  SingleLayerQG.streamfunctionfrompv!(prob_equivalentbarotropicQG.vars.ψh, prob_equivalentbarotropicQG.vars.qh, prob_equivalentbarotropicQG.params, prob_equivalentbarotropicQG.grid)
  
  return (prob_barotropicQG.vars.ψ ≈ ψ && prob_equivalentbarotropicQG.vars.ψ ≈ ψ)
end

function test_1layerqg_energy_dissipation(dev; deformation_radius=2.23)
  prob = SingleLayerQG.Problem(dev; nx=16, deformation_radius=deformation_radius)
  SingleLayerQG.energy_dissipation(prob)
  return nothing
end

function test_1layerqg_enstrophy_dissipation(dev; deformation_radius=2.23)
  prob = SingleLayerQG.Problem(dev; nx=16, deformation_radius=deformation_radius)
  SingleLayerQG.enstrophy_dissipation(prob)
  return nothing
end

function test_1layerqg_energy_work(dev; deformation_radius=2.23)
  prob = SingleLayerQG.Problem(dev; nx=16, deformation_radius=deformation_radius)
  SingleLayerQG.energy_work(prob)
  return nothing
end

function test_1layerqg_enstrophy_work(dev; deformation_radius=2.23)
  prob = SingleLayerQG.Problem(dev; nx=16, deformation_radius=deformation_radius)
  SingleLayerQG.enstrophy_work(prob)
  return nothing
end

function test_1layerqg_energy_drag(dev; deformation_radius=2.23)
  prob = SingleLayerQG.Problem(dev; nx=16, deformation_radius=deformation_radius)
  SingleLayerQG.energy_drag(prob)
  return nothing
end

function test_1layerqg_enstrophy_drag(dev; deformation_radius=2.23)
  prob = SingleLayerQG.Problem(dev; nx=16, deformation_radius=deformation_radius)
  SingleLayerQG.enstrophy_drag(prob)
  return nothing
end
