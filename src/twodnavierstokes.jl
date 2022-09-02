module TwoDNavierStokes

export
  Problem,
  set_ζ!,
  updatevars!,

  energy,
  energy_dissipation_hyperviscosity,
  energy_dissipation_hypoviscosity,
  energy_work,
  enstrophy,
  enstrophy_dissipation_hyperviscosity,
  enstrophy_dissipation_hypoviscosity,
  enstrophy_work

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
                      ν = 0,
                     nν = 1,
                      μ = 0,
                     nμ = 0,
                     dt = 0.01,
                stepper = "RK4",
                  calcF = nothingfunction,
             stochastic = false,
       aliased_fraction = 1/3,
                      T = Float64)

Construct a two-dimensional Navier-Stokes problem on device `dev`.

Arguments
=========
  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.

Keyword arguments
=================
  - `nx`: Number of grid points in ``x``-domain.
  - `ny`: Number of grid points in ``y``-domain.
  - `Lx`: Extent of the ``x``-domain.
  - `Ly`: Extent of the ``y``-domain.
  - `ν`: Small-scale (hyper)-viscosity coefficient.
  - `nν`: (Hyper)-viscosity order, `nν```≥ 1``.
  - `μ`: Large-scale (hypo)-viscosity coefficient.
  - `nμ`: (Hypo)-viscosity order, `nμ```≤ 0``.
  - `dt`: Time-step.
  - `stepper`: Time-stepping method.
  - `calcF`: Function that calculates the Fourier transform of the forcing, ``F̂``.
  - `stochastic`: `true` or `false`; boolean denoting whether `calcF` is temporally stochastic.
  - `aliased_fraction`: the fraction of high-wavenumbers that are zero-ed out by `dealias!()`.
  - `T`: `Float32` or `Float64`; floating point type used for `problem` data.
"""
function Problem(dev::Device=CPU();
  # Numerical parameters
                nx = 256,
                ny = nx,
                Lx = 2π,
                Ly = Lx,
  # Drag and/or hyper-/hypo-viscosity
                 ν = 0,
                nν = 1,
                 μ = 0,
                nμ = 0,
  # Timestepper and equation options
                dt = 0.01,
           stepper = "RK4",
             calcF = nothingfunction,
        stochastic = false,
  # Float type and dealiasing
  aliased_fraction = 1/3,
                 T = Float64)

  grid = TwoDGrid(dev; nx, Lx, ny, Ly, aliased_fraction, T)

  params = Params(T(ν), nν, T(μ), nμ, calcF)

  vars = calcF == nothingfunction ? DecayingVars(grid) : (stochastic ? StochasticForcedVars(grid) : ForcedVars(grid))

  equation = Equation(params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params)
end


# ----------
# Parameters
# ----------

"""
    struct Params{T} <: AbstractParams

The parameters for a two-dimensional Navier-Stokes problem:

$(TYPEDFIELDS)
"""
struct Params{T} <: AbstractParams
    "small-scale (hyper)-viscosity coefficient"
       ν :: T
    "(hyper)-viscosity order, `nν```≥ 1``"
      nν :: Int
    "large-scale (hypo)-viscosity coefficient"
       μ :: T
    "(hypo)-viscosity order, `nμ```≤ 0``"
      nμ :: Int
    "function that calculates the Fourier transform of the forcing, ``F̂``"
  calcF! :: Function
end

Params(ν, nν) = Params(ν, nν, typeof(ν)(0), 0, nothingfunction)


# ---------
# Equations
# ---------

"""
    Equation(params, grid)

Return the `equation` for two-dimensional Navier-Stokes with `params` and `grid`. The linear
operator ``L`` includes (hyper)-viscosity of order ``n_ν`` with coefficient ``ν`` and 
hypo-viscocity of order ``n_μ`` with coefficient ``μ``,

```math
L = - ν |𝐤|^{2 n_ν} - μ |𝐤|^{2 n_μ} .
```

Plain-old viscocity corresponds to ``n_ν = 1`` while ``n_μ = 0`` corresponds to linear drag.

The nonlinear term is computed via the function `calcN!`.
"""
function Equation(params::Params, grid::AbstractGrid)
  L = @. - params.ν * grid.Krsq^params.nν - params.μ * grid.Krsq^params.nμ
  CUDA.@allowscalar L[1, 1] = 0
  
  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

abstract type TwoDNavierStokesVars <: AbstractVars end

"""
    struct Vars{Aphys, Atrans, F, P} <: TwoDNavierStokesVars

The variables for two-dimensional Navier-Stokes problem:

$(FIELDS)
"""
struct Vars{Aphys, Atrans, F, P} <: TwoDNavierStokesVars
    "relative vorticity"
        ζ :: Aphys
    "x-component of velocity"
        u :: Aphys
    "y-component of velocity"
        v :: Aphys
    "Fourier transform of relative vorticity"
       ζh :: Atrans
    "Fourier transform of x-component of velocity"
       uh :: Atrans
    "Fourier transform of y-component of velocity"
       vh :: Atrans
    "Fourier transform of forcing"
       Fh :: F
    "`sol` at previous time-step"
  prevsol :: P
end

const DecayingVars = Vars{<:AbstractArray, <:AbstractArray, Nothing, Nothing}
const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    DecayingVars(dev, grid)

Return the variables `vars` for unforced two-dimensional Navier-Stokes problem on `grid`.
"""
function DecayingVars(grid::AbstractGrid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) ζ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) ζh uh vh
  
  return Vars(ζ, u, v, ζh, uh, vh, nothing, nothing)
end

"""
    ForcedVars(grid)

Return the variables `vars` for forced two-dimensional Navier-Stokes on `grid`.
"""
function ForcedVars(grid::AbstractGrid)
  Dev = typeof(grid.device)
  T = eltype(grid)
  
  @devzeros Dev T (grid.nx, grid.ny) ζ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) ζh uh vh Fh
  
  return Vars(ζ, u, v, ζh, uh, vh, Fh, nothing)
end

"""
    StochasticForcedVars(grid)

Return the variables `vars` for stochastically forced two-dimensional Navier-Stokes on device `grid`.
"""
function StochasticForcedVars(grid::AbstractGrid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) ζ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) ζh uh vh Fh prevsol
  
  return Vars(ζ, u, v, ζh, uh, vh, Fh, prevsol)
end


# -------
# Solvers
# -------

"""
    calcN_advection!(N, sol, t, clock, vars, params, grid)

Calculate the Fourier transform of the advection term, ``- 𝖩(ψ, ζ)`` in conservative form, 
i.e., ``- ∂_x[(∂_y ψ)ζ] - ∂_y[(∂_x ψ)ζ]`` and store it in `N`:

```math
N = - \\widehat{𝖩(ψ, ζ)} = - i k_x \\widehat{u ζ} - i k_y \\widehat{v ζ} .
```
"""
function calcN_advection!(N, sol, t, clock, vars, params, grid)
  @. vars.uh =   im * grid.l  * grid.invKrsq * sol
  @. vars.vh = - im * grid.kr * grid.invKrsq * sol
  @. vars.ζh = sol

  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)
  ldiv!(vars.ζ, grid.rfftplan, vars.ζh)
  
  uζ = vars.u                  # use vars.u as scratch variable
  @. uζ *= vars.ζ              # u*ζ
  vζ = vars.v                  # use vars.v as scratch variable
  @. vζ *= vars.ζ              # v*ζ
  
  uζh = vars.uh                # use vars.uh as scratch variable
  mul!(uζh, grid.rfftplan, uζ) # \hat{u*ζ}
  vζh = vars.vh                # use vars.vh as scratch variable
  mul!(vζh, grid.rfftplan, vζ) # \hat{v*ζ}

  @. N = - im * grid.kr * uζh - im * grid.l * vζh
  
  return nothing
end

"""
    calcN!(N, sol, t, clock, vars, params, grid)

Calculate the nonlinear term, that is the advection term and the forcing,

```math
N = - \\widehat{𝖩(ψ, ζ)} + F̂ .
```
"""
function calcN!(N, sol, t, clock, vars, params, grid)
  dealias!(sol, grid)
  
  calcN_advection!(N, sol, t, clock, vars, params, grid)
  
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

"""
    addforcing!(N, sol, t, clock, vars, params, grid)

When the problem includes forcing, calculate the forcing term ``F̂`` and add it to the
nonlinear term ``N``.
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
  
  @. vars.ζh = sol
  @. vars.uh =   im * grid.l  * grid.invKrsq * sol
  @. vars.vh = - im * grid.kr * grid.invKrsq * sol
  
  ldiv!(vars.ζ, grid.rfftplan, deepcopy(vars.ζh)) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh)) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh)) # deepcopy() since inverse real-fft destroys its input
  
  return nothing
end

"""
    set_ζ!(prob, ζ)

Set the solution `sol` as the transform of `ζ` and then update variables in `prob.vars`.
"""
function set_ζ!(prob, ζ)
  mul!(prob.sol, prob.grid.rfftplan, ζ)
  
  CUDA.@allowscalar prob.sol[1, 1] = 0 # enforce zero domain average
  
  updatevars!(prob)
  
  return nothing
end

"""
    energy(prob)

Return the domain-averaged kinetic energy. Since ``u² + v² = |{\\bf ∇} ψ|²``, the domain-averaged 
kinetic energy is

```math
\\int \\frac1{2} |{\\bf ∇} ψ|² \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} \\frac1{2} |𝐤|² |ψ̂|² ,
```

where ``ψ`` is the streamfunction.
"""
@inline function energy(prob)
  sol, vars, grid = prob.sol, prob.vars, prob.grid
  energyh = vars.uh # use vars.uh as scratch variable
  
  @. energyh = 1 / 2 * grid.invKrsq * abs2(sol)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energyh, grid)
end

"""
    enstrophy(prob)

Returns the problem's (`prob`) domain-averaged enstrophy,

```math
\\int \\frac1{2} ζ² \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} \\frac1{2} |ζ̂|² ,
```

where ``ζ`` is the relative vorticity.
"""
@inline enstrophy(prob) = 1 / (2 * prob.grid.Lx * prob.grid.Ly) * parsevalsum(abs2.(prob.sol), prob.grid)

"""
    energy_dissipation(prob, ξ, νξ)

Return the domain-averaged energy dissipation rate done by the viscous term,

```math
- ξ (-1)^{n_ξ+1} \\int ψ ∇^{2n_ξ} ζ \\frac{𝖽x 𝖽y}{L_x L_y} = - ξ \\sum_{𝐤} |𝐤|^{2(n_ξ-1)} |ζ̂|² ,
```
where ``ξ`` and ``nξ`` could be either the (hyper)-viscosity coefficient ``ν`` and its order 
``n_ν``, or the hypo-viscocity coefficient ``μ`` and its order ``n_μ``.
"""
@inline function energy_dissipation(prob, ξ, nξ)
  sol, vars, grid = prob.sol, prob.vars, prob.grid
  energy_dissipationh = vars.uh # use vars.uh as scratch variable
  
  @. energy_dissipationh = - ξ * grid.Krsq^(nξ - 1) * abs2(sol)
  CUDA.@allowscalar energy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dissipationh, grid)
end

"""
    energy_dissipation_hyperviscosity(prob)

Return the problem's (`prob`) domain-averaged energy dissipation rate done by the ``ν`` (hyper)-viscosity.
"""
energy_dissipation_hyperviscosity(prob) = energy_dissipation(prob, prob.params.ν, prob.params.nν)

"""
    energy_dissipation_hypoviscosity(prob)

Return the problem's (`prob`) domain-averaged energy dissipation rate done by the ``μ`` (hypo)-viscosity.
"""
energy_dissipation_hypoviscosity(prob) = energy_dissipation(prob, prob.params.μ, prob.params.nμ)

"""
    enstrophy_dissipation(prob, ξ, νξ)

Return the problem's (`prob`) domain-averaged enstrophy dissipation rate done by the viscous term,

```math
ξ (-1)^{n_ξ+1} \\int ζ ∇^{2n_ξ} ζ \\frac{𝖽x 𝖽y}{L_x L_y} = - ξ \\sum_{𝐤} |𝐤|^{2n_ξ} |ζ̂|² ,
```

where ``ξ`` and ``nξ`` could be either the (hyper)-viscosity coefficient ``ν`` and its order 
``n_ν``, or the hypo-viscocity coefficient ``μ`` and its order ``n_μ``.
"""
@inline function enstrophy_dissipation(prob, ξ, nξ)
  sol, vars, grid = prob.sol, prob.vars, prob.grid
  enstrophy_dissipationh = vars.uh # use vars.uh as scratch variable
  
  @. enstrophy_dissipationh = - ξ * grid.Krsq^nξ * abs2(sol)
  CUDA.@allowscalar enstrophy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dissipationh, grid)
end

"""
    enstrophy_dissipation_hyperviscosity(prob)

Return the problem's (`prob`) domain-averaged enstrophy dissipation rate done by the ``ν`` (hyper)-viscosity.
"""
enstrophy_dissipation_hyperviscosity(prob) = enstrophy_dissipation(prob, prob.params.ν, prob.params.nν)

"""
    enstrophy_dissipation_hypoviscosity(prob)

Return the problem's (`prob`) domain-averaged enstrophy dissipation rate done by the ``μ`` (hypo)-viscosity.
"""
enstrophy_dissipation_hypoviscosity(prob) = enstrophy_dissipation(prob, prob.params.μ, prob.params.nμ)

"""
    energy_work(prob)

Return the problem's (`prob`) domain-averaged rate of work of energy by the forcing ``F``,

```math
- \\int ψ F \\frac{𝖽x 𝖽y}{L_x L_y} = - \\sum_{𝐤} ψ̂ F̂^* .
```

where ``ψ`` is the stream flow.
"""
@inline energy_work(prob) = energy_work(prob.sol, prob.vars, prob.grid)

@inline function energy_work(sol, vars::ForcedVars, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable
  
  @. energy_workh = grid.invKrsq * sol * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_workh, grid)
end

@inline function energy_work(sol, vars::StochasticForcedVars, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable
  
  @. energy_workh = grid.invKrsq * (vars.prevsol + sol) / 2 * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_workh, grid)
end

"""
    enstrophy_work(prob)

Return the problem's (`prob`) domain-averaged rate of work of enstrophy by the forcing ``F``,

```math
\\int ζ F \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} ζ̂ F̂^* ,
```

where ``ζ`` is the relative vorticity.
"""
@inline enstrophy_work(prob) = enstrophy_work(prob.sol, prob.vars, prob.grid)

@inline function enstrophy_work(sol, vars::ForcedVars, grid)
  enstrophy_workh = vars.uh # use vars.uh as scratch variable
  
  @. enstrophy_workh = sol * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_workh, grid)
end

@inline function enstrophy_work(sol, vars::StochasticForcedVars, grid)
  enstrophy_workh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_workh = (vars.prevsol + sol) / 2 * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_workh, grid)
end

end # module
