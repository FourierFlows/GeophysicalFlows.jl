module ShallowWater

export
  Problem,
  set_uvh!,
  updatevars!,

  mass,
  kinetic_energy,
  potential_energy,
  energy

using
  CUDA,
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum

nothingfunction(args...) = nothing

"""
    Problem(dev::Device = CPU();
                     nx = 256,
                     ny = nx,
                     Lx = 2π,
                     Ly = Lx,
                      f = 0,
                      g = 1,
                      b = nothing,
                      ν = 0,
                     nν = 1,
                     dt = 0.01,
                stepper = "RK4",
                  calcF = nothingfunction,
             stochastic = false,
       aliased_fraction = 1/3,
                      T = Float64)

Construct a two-dimensional shallow water problem on device `dev`.

Arguments
=========
  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.

Keyword arguments
=================
  - `nx`: Number of grid points in ``x``-domain.
  - `ny`: Number of grid points in ``y``-domain.
  - `Lx`: Extent of the ``x``-domain.
  - `Ly`: Extent of the ``y``-domain.
  - `f`: Coriolis parameter.
  - `g`: Gravitational acceleration.
  - `b`: Bottom bathymetry ``b(x, y)``, measured from the rest height of the free surface;
    an array of size `(nx, ny)` or `nothing` for a flat bottom.
  - `ν`: (Hyper)-viscosity coefficient.
  - `nν`: (Hyper)-viscosity order, `nν```≥ 1``.
  - `dt`: Time-step.
  - `stepper`: Time-stepping method.
  - `calcF`: Function that calculates the Fourier transform of the forcing, ``F̂``.
  - `stochastic`: `true` or `false`; boolean denoting whether `calcF` is temporally stochastic.
  - `aliased_fraction`: the fraction of high wavenumbers that are zero-ed out by `dealias!()`.
  - `T`: `Float32` or `Float64`; floating point type used for `problem` data.
"""
function Problem(dev::Device=CPU();
  # Numerical parameters
                nx = 256,
                ny = nx,
                Lx = 2π,
                Ly = Lx,
  # Physical parameters
                 f = 0,
                 g = 1,
                 b = nothing,
  # Hyper-viscosity
                 ν = 0,
                nν = 1,
  # Timestepper and equation options
                dt = 0.01,
           stepper = "RK4",
             calcF = nothingfunction,
        stochastic = false,
  # Float type and dealiasing
  aliased_fraction = 1/3,
                 T = Float64)

  grid = TwoDGrid(dev; nx, Lx, ny, Ly, aliased_fraction, T)

  params = Params(grid, T(f), T(g), b, T(ν), nν, calcF)

  vars = calcF == nothingfunction ? DecayingVars(grid) : (stochastic ? StochasticForcedVars(grid) : ForcedVars(grid))

  equation = Equation(params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params)
end


# ----------
# Parameters
# ----------

"""
    struct Params{T, Aphys, Atrans} <: AbstractParams

The parameters for a two-dimensional shallow water problem.

$(TYPEDFIELDS)
"""
struct Params{T, Aphys, Atrans} <: AbstractParams
    "Coriolis parameter"
       f :: T
    "gravitational acceleration"
       g :: T
    "bottom bathymetry, ``b``"
       b :: Aphys
    "Fourier transform of bottom bathymetry, ``b̂``"
      bh :: Atrans
    "(hyper)-viscosity coefficient"
       ν :: T
    "(hyper)-viscosity order, `nν```≥ 1``"
      nν :: Int
    "function that calculates the Fourier transform of the forcing, ``F̂``"
  calcF! :: Function
end

"""
    Params(grid, f, g, b, ν, nν, calcF)

Return the parameters for a shallow water problem on `grid`. The bathymetry `b` is dealiased
so that a lake-at-rest state, ``h + b = `` constant, is represented exactly on the grid.
"""
function Params(grid::AbstractGrid, f, g, b, ν, nν, calcF)
  T = eltype(grid)
  A = device_array(grid.device)

  b === nothing && (b = zeros(grid.device, T, (grid.nx, grid.ny)))

  # remove high-wavenumbers from bathytmetry
  bh = rfft(A{T}(b))
  dealias!(bh, grid)
  b = irfft(bh, grid.nx)

  return Params(f, g, b, bh, ν, nν, calcF)
end


# ---------
# Equations
# ---------

"""
    Equation(params, grid)

Return the `equation` for two-dimensional shallow water with `params` and `grid`. The linear
operator ``L`` includes (hyper)-viscosity of order ``n_ν`` with coefficient ``ν``,

```math
L = - ν |𝐤|^{2 n_ν} ,
```

which is applied to each of ``q_u``, ``q_v``, and ``h``. The nonlinear term is computed via
[`calcN!`](@ref GeophysicalFlows.ShallowWater.calcN!).
"""
function Equation(params::Params, grid::AbstractGrid)
  T = eltype(grid)

  D = @. - params.ν * grid.Krsq^params.nν
  CUDA.@allowscalar D[1, 1] = 0

  L = zeros(grid.device, T, (grid.nkr, grid.nl, 3))

  for j in 1:3
    @views @. L[:, :, j] = D
  end

  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

abstract type ShallowWaterVars <: AbstractVars end

"""
    struct Vars{Aphys, Atrans, F, P} <: ShallowWaterVars

The variables for a two-dimensional shallow water problem.

$(FIELDS)
"""
struct Vars{Aphys, Atrans, F, P} <: ShallowWaterVars
    "``x``-volume flux, ``q_u = h u``"
       qu :: Aphys
    "``y``-volume flux, ``q_v = h v``"
       qv :: Aphys
    "total fluid depth"
        h :: Aphys
    "``x``-component of velocity"
        u :: Aphys
    "``y``-component of velocity"
        v :: Aphys
    "Fourier transform of ``x``-volume flux"
      quh :: Atrans
    "Fourier transform of ``y``-volume flux"
      qvh :: Atrans
    "Fourier transform of total fluid depth"
       hh :: Atrans
    "Fourier transform of forcing"
       Fh :: F
    "`sol` at previous time-step"
  prevsol :: P
end

const DecayingVars = Vars{<:AbstractArray, <:AbstractArray, Nothing, Nothing}
const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    DecayingVars(grid)

Return the variables for unforced two-dimensional shallow water on `grid`.
"""
function DecayingVars(grid::AbstractGrid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) qu qv h u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) quh qvh hh

  return Vars(qu, qv, h, u, v, quh, qvh, hh, nothing, nothing)
end

"""
    ForcedVars(grid)

Return the variables for forced two-dimensional shallow water on `grid`.
"""
function ForcedVars(grid::AbstractGrid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) qu qv h u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) quh qvh hh
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, 3) Fh

  return Vars(qu, qv, h, u, v, quh, qvh, hh, Fh, nothing)
end

"""
    StochasticForcedVars(grid)

Return the variables for stochastically forced two-dimensional shallow water on `grid`.
"""
function StochasticForcedVars(grid::AbstractGrid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) qu qv h u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) quh qvh hh
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, 3) Fh prevsol

  return Vars(qu, qv, h, u, v, quh, qvh, hh, Fh, prevsol)
end


# -------
# Solvers
# -------

"""
    calcN_advection!(N, sol, t, clock, vars, params, grid)

Calculate the Fourier transform of the advection, Coriolis, and pressure-gradient terms
of the shallow water equations in conservative form and store them in `N`:

```math
\\begin{aligned}
N_{q_u} & = - i k_x \\widehat{q_u^2 / h}   - i k_y \\widehat{q_u q_v / h} + f \\widehat{q_v} - g \\widehat{h ∂_x η} , \\\\
N_{q_v} & = - i k_x \\widehat{q_u q_v / h} - i k_y \\widehat{q_v^2 / h}   - f \\widehat{q_u} - g \\widehat{h ∂_y η} , \\\\
N_h     & = - i k_x \\widehat{q_u}         - i k_y \\widehat{q_v} ,
\\end{aligned}
```

where ``η = h + b`` is the free-surface elevation.
"""
function calcN_advection!(N, sol, t, clock, vars, params, grid)
  @views begin
    q̂u, q̂v, ĥ = sol[:, :, 1], sol[:, :, 2], sol[:, :, 3]
    Nqu, Nqv, Nh = N[:, :, 1], N[:, :, 2], N[:, :, 3]
  end

  @. Nqu =   params.f * q̂v
  @. Nqv = - params.f * q̂u
  @. Nh  = - im * grid.kr * q̂u - im * grid.l * q̂v

  @. vars.quh = q̂u
  @. vars.qvh = q̂v
  @. vars.hh  = ĥ

  ldiv!(vars.qu, grid.rfftplan, vars.quh)
  ldiv!(vars.qv, grid.rfftplan, vars.qvh)
  ldiv!(vars.h,  grid.rfftplan, vars.hh)

  @. vars.u = vars.qu / vars.h
  @. vars.v = vars.qv / vars.h

  # from here on, vars.qu and vars.quh are used as scratch variables
  flux, fluxh = vars.qu, vars.quh

  @. flux = vars.h * vars.u * vars.u      # h u u
  mul!(fluxh, grid.rfftplan, flux)
  @. Nqu -= im * grid.kr * fluxh

  @. flux = vars.h * vars.u * vars.v      # h u v
  mul!(fluxh, grid.rfftplan, flux)
  @. Nqu -= im * grid.l  * fluxh
  @. Nqv -= im * grid.kr * fluxh

  @. flux = vars.h * vars.v * vars.v      # h v v
  mul!(fluxh, grid.rfftplan, flux)
  @. Nqv -= im * grid.l  * fluxh

  # pressure gradient -g h ∇η, with η = h + b
  @. fluxh = im * grid.kr * (ĥ + params.bh) # ∂η/∂x
  ldiv!(flux, grid.rfftplan, fluxh)
  @. flux *= vars.h
  mul!(fluxh, grid.rfftplan, flux)
  @. Nqu -= params.g * fluxh

  @. fluxh = im * grid.l * (ĥ + params.bh)  # ∂η/∂y
  ldiv!(flux, grid.rfftplan, fluxh)
  @. flux *= vars.h
  mul!(fluxh, grid.rfftplan, flux)
  @. Nqv -= params.g * fluxh

  return nothing
end

"""
    calcN!(N, sol, t, clock, vars, params, grid)

Calculate the nonlinear term, that is the advection, Coriolis, and pressure-gradient terms
(see [`calcN_advection!`](@ref GeophysicalFlows.ShallowWater.calcN_advection!)) and the forcing.
"""
function calcN!(N, sol, t, clock, vars, params, grid)
  dealias!(sol, grid)

  calcN_advection!(N, sol, t, clock, vars, params, grid)

  addforcing!(N, sol, t, clock, vars, params, grid)

  return nothing
end

"""
    addforcing!(N, sol, t, clock, vars, params, grid)

When the problem includes forcing, calculate the forcing term ``F̂`` for each of
``q_u``, ``q_v``, and ``h`` and add it to the nonlinear term ``N``.
"""
addforcing!(N, sol, t, clock, vars::DecayingVars, params, grid) = nothing

function addforcing!(N, sol, t, clock, vars::ForcedVars, params, grid)
  params.calcF!(vars.Fh, sol, t, clock, vars, params, grid)

  @. N += vars.Fh

  return nothing
end

function addforcing!(N, sol, t, clock, vars::StochasticForcedVars, params, grid)
  if t == clock.t # not a substep
    @. vars.prevsol = sol # sol at previous time-step is needed to compute budgets for stochastic forcing
    params.calcF!(vars.Fh, sol, t, clock, vars, params, grid)
  end

  @. N += vars.Fh

  return nothing
end


# ----------------
# Helper functions
# ----------------

"""
    updatevars!(prob)

Update problem's variables in `prob.vars` using the state in `prob.sol`.
"""
function updatevars!(prob)
  vars, grid, sol = prob.vars, prob.grid, prob.sol

  dealias!(sol, grid)

  @views @. vars.quh = sol[:, :, 1]
  @views @. vars.qvh = sol[:, :, 2]
  @views @. vars.hh  = sol[:, :, 3]

  ldiv!(vars.qu, grid.rfftplan, deepcopy(vars.quh)) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.qv, grid.rfftplan, deepcopy(vars.qvh)) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.h,  grid.rfftplan, deepcopy(vars.hh))  # deepcopy() since inverse real-fft destroys its input

  @. vars.u = vars.qu / vars.h
  @. vars.v = vars.qv / vars.h

  return nothing
end

"""
    set_uvh!(prob, u, v, h)

Set the solution `sol` from the velocity components `u`, `v` and the total fluid depth `h`
and then update variables in `prob.vars`.
"""
function set_uvh!(prob, u, v, h)
  sol, vars, grid = prob.sol, prob.vars, prob.grid
  A = device_array(grid.device)

  u, v, h = A(u), A(v), A(h)

  @. vars.h  = h
  @. vars.qu = h * u
  @. vars.qv = h * v

  mul!(vars.quh, grid.rfftplan, vars.qu)
  mul!(vars.qvh, grid.rfftplan, vars.qv)
  mul!(vars.hh,  grid.rfftplan, vars.h)

  @views @. sol[:, :, 1] = vars.quh
  @views @. sol[:, :, 2] = vars.qvh
  @views @. sol[:, :, 3] = vars.hh

  updatevars!(prob)

  return nothing
end

"""
    mass(prob)

Return the domain-averaged fluid depth,

```math
\\int h \\frac{𝖽x 𝖽y}{L_x L_y} .
```
"""
@inline function mass(prob)
  grid = prob.grid

  return real(CUDA.@allowscalar prob.sol[1, 1, 3]) / (grid.nx * grid.ny)
end

"""
    kinetic_energy(prob)

Return the domain-averaged kinetic energy,

```math
\\int \\frac1{2} h (u² + v²) \\frac{𝖽x 𝖽y}{L_x L_y} .
```
"""
@inline function kinetic_energy(prob)
  updatevars!(prob)
  vars, grid = prob.vars, prob.grid

  return sum(@. vars.h * (vars.u^2 + vars.v^2) / 2) / (grid.nx * grid.ny)
end

"""
    potential_energy(prob)

Return the domain-averaged potential energy,

```math
\\int \\frac1{2} g η² \\frac{𝖽x 𝖽y}{L_x L_y} ,
```

where ``η = h + b`` is the free-surface elevation.
"""
@inline function potential_energy(prob)
  updatevars!(prob)
  vars, params, grid = prob.vars, prob.params, prob.grid

  return sum(@. params.g * (vars.h + params.b)^2 / 2) / (grid.nx * grid.ny)
end

"""
    energy(prob)

Return the domain-averaged total energy, i.e., the sum of the
[`kinetic_energy`](@ref GeophysicalFlows.ShallowWater.kinetic_energy) and the
[`potential_energy`](@ref GeophysicalFlows.ShallowWater.potential_energy).
"""
@inline energy(prob) = kinetic_energy(prob) + potential_energy(prob)

end # module
