"""
    test_shallowwater_problemtype(dev, T)

Test that the `Problem` constructor returns arrays of the correct type.
"""
function test_shallowwater_problemtype(dev, T)
  prob = ShallowWater.Problem(dev; nx=16, T)

  A = device_array(dev)

  return (typeof(prob.sol) <: A{Complex{T}, 3} && size(prob.sol, 3) == 3 &&
          typeof(prob.grid.Lx) == T && eltype(prob.grid.x) == T && typeof(prob.vars.u) <: A{T, 2})
end

"""
    test_shallowwater_inertiagravitywave(dev; kwargs...)

Test that a small-amplitude perturbation ``h = H + ε \\cos(k x)`` evolves according to the
linear inertia-gravity wave solution,

```math
h = H + ε \\cos(k x) \\left[1 - \\frac{g H k^2}{ω^2} (1 - \\cos ωt) \\right] , \\quad
u = \\frac{g k ε}{ω} \\sin ωt \\, \\sin(k x) ,
```

with ``ω^2 = f^2 + g H k^2``.
"""
function test_shallowwater_inertiagravitywave(dev; n=32, L=2π, f=1.0, g=1.0, H=1.0, ε=1e-6, dt=0.005, nsteps=400)
  prob = ShallowWater.Problem(dev; nx=n, Lx=L, f, g, dt, stepper="RK4")
  x, y = gridpoints(prob.grid)

  k = 2 * 2π / L
  ω = sqrt(f^2 + g * H * k^2)

  ShallowWater.set_uvh!(prob, zero(x), zero(x), @. H + ε * cos(k * x))

  stepforward!(prob, nsteps)
  ShallowWater.updatevars!(prob)

  t = prob.clock.t
  h_analytic = @. H + ε * cos(k * x) * (1 - g * H * k^2 / ω^2 * (1 - cos(ω * t)))
  u_analytic = @. g * k * ε / ω * sin(ω * t) * sin(k * x)

  h, u = Array(prob.vars.h), Array(prob.vars.u)

  return isapprox(h .- H, Array(h_analytic) .- H, rtol=1e-4) && isapprox(u, Array(u_analytic), rtol=1e-4)
end

"""
    test_shallowwater_lakeatrest(dev; kwargs...)

Test that a lake at rest, ``u = v = 0`` and ``h + b = `` constant, remains at rest over
non-trivial bathymetry ``b``.
"""
function test_shallowwater_lakeatrest(dev; n=64, L=2π, f=1.0, g=1.0, H=1.0, dt=0.01, nsteps=100)
  grid = TwoDGrid(dev; nx=n, Lx=L)
  x, y = gridpoints(grid)

  b = @. - H + 0.2 * exp(-(x^2 + 2y^2)) + 0.1 * sin(2x) * cos(y)

  prob = ShallowWater.Problem(dev; nx=n, Lx=L, f, g, b, dt, stepper="RK4")

  ShallowWater.set_uvh!(prob, zero(x), zero(x), - prob.params.b)

  stepforward!(prob, nsteps)
  ShallowWater.updatevars!(prob)

  return maximum(abs, prob.vars.u) < 1e-12 && maximum(abs, prob.vars.v) < 1e-12 &&
         isapprox(prob.vars.h, - prob.params.b, rtol=1e-12)
end

"""
    test_shallowwater_conservation(dev; kwargs...)

Test that mass and energy are conserved for an inviscid, unforced flow over bathymetry.
"""
function test_shallowwater_conservation(dev; n=64, L=2π, f=1.0, g=1.0, H=1.0, dt=0.005, nsteps=200)
  grid = TwoDGrid(dev; nx=n, Lx=L)
  x, y = gridpoints(grid)

  b = @. - H + 0.1 * exp(-((x - 1)^2 + y^2))

  prob = ShallowWater.Problem(dev; nx=n, Lx=L, f, g, b, dt, stepper="RK4")

  u₀ = @. 0.05 * sin(x) * cos(2y)
  v₀ = @. 0.05 * cos(x + y)
  h₀ = @. 0.1 * exp(-(x^2 + y^2)) - prob.params.b

  ShallowWater.set_uvh!(prob, u₀, v₀, h₀)

  M₀, E₀ = ShallowWater.mass(prob), ShallowWater.energy(prob)

  stepforward!(prob, nsteps)

  M, E = ShallowWater.mass(prob), ShallowWater.energy(prob)

  return isapprox(M, M₀, rtol=1e-13) && isapprox(E, E₀, rtol=1e-8)
end

"""
    test_shallowwater_deterministicforcing_mass(dev; kwargs...)

Test that a uniform mass source ``F_h`` increases the domain-averaged depth as ``F_h t``.
"""
function test_shallowwater_deterministicforcing_mass(dev; n=32, L=2π, Fh=0.1, dt=0.01, nsteps=100)
  function calcF!(Fh_sol, sol, t, clock, vars, params, grid)
    Fh_sol .= 0
    CUDA.@allowscalar Fh_sol[1, 1, 3] = Fh * grid.nx * grid.ny

    return nothing
  end

  prob = ShallowWater.Problem(dev; nx=n, Lx=L, f=1.0, dt, stepper="RK4", calcF=calcF!)
  x, y = gridpoints(prob.grid)

  ShallowWater.set_uvh!(prob, zero(x), zero(x), @. 1 + 0.1 * cos(x))

  M₀ = ShallowWater.mass(prob)

  stepforward!(prob, nsteps)

  return isapprox(ShallowWater.mass(prob), M₀ + Fh * prob.clock.t, rtol=1e-12)
end

"""
    test_shallowwater_viscosity(dev; kwargs...)

Test that hyperviscosity damps a single Fourier mode of ``h`` at rate ``ν |𝐤|^{2 n_ν}``.
"""
function test_shallowwater_viscosity(dev; n=32, L=2π, ν=1e-2, nν=2, dt=0.01, nsteps=100)
  # use g = 0 so that the mode does not oscillate
  prob = ShallowWater.Problem(dev; nx=n, Lx=L, g=0, ν, nν, dt, stepper="ETDRK4")
  x, y = gridpoints(prob.grid)

  k = 3 * 2π / L
  ε = 1e-2

  ShallowWater.set_uvh!(prob, zero(x), zero(x), @. 1 + ε * cos(k * x))

  stepforward!(prob, nsteps)
  ShallowWater.updatevars!(prob)

  h_analytic = @. 1 + ε * cos(k * x) * exp(- ν * k^(2nν) * prob.clock.t)

  return isapprox(Array(prob.vars.h), Array(h_analytic), rtol=1e-12)
end

"""
    test_shallowwater_stochasticforcedvars(dev)

Test that a stochastically forced `Problem` is constructed with the right `Vars`.
"""
function test_shallowwater_stochasticforcedvars(dev)
  calcF!(Fh, sol, t, clock, vars, params, grid) = nothing

  prob = ShallowWater.Problem(dev; nx=16, calcF=calcF!, stochastic=true)

  return prob.vars isa ShallowWater.StochasticForcedVars && size(prob.vars.prevsol) == size(prob.sol)
end
