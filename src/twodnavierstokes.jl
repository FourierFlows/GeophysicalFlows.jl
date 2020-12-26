module TwoDNavierStokes

export
  Problem,
  set_ζ!,
  updatevars!,

  energy,
  energy_dissipation,
  energy_work,
  energy_drag,
  enstrophy,
  enstrophy_dissipation,
  enstrophy_work,
  enstrophy_drag

using
  CUDA,
  Reexport

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum

nothingfunction(args...) = nothing

"""
    Problem(dev::Device; parameters...)

Construct a 2D Navier-Stokes problem.
"""
function Problem(dev::Device=CPU();
  # Numerical parameters
          nx = 256,
          Lx = 2π,
          ny = nx,
          Ly = Lx,
          dt = 0.01,
  # Drag and/or hyper-/hypo-viscosity
           ν = 0,
          nν = 1,
           μ = 0,
          nμ = 0,
  # Timestepper and equation options
     stepper = "RK4",
       calcF = nothingfunction,
  stochastic = false,
           T = Float64)

  grid = TwoDGrid(dev, nx, Lx, ny, Ly; T=T)

  params = Params{T}(ν, nν, μ, nμ, calcF)

  vars = calcF == nothingfunction ? Vars(dev, grid) : (stochastic ? StochasticForcedVars(dev, grid) : ForcedVars(dev, grid))

  equation = Equation(params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
end


# ----------
# Parameters
# ----------

"""
    Params{T}(ν, nν, μ, nμ, calcF!)

Returns the params for two-dimensional Navier-Stokes.
"""
struct Params{T} <: AbstractParams
    "small-scale (hyper)-viscosity coefficient"
       ν :: T
    "(hyper)-viscosity order, `nν ≥ 1`"
      nν :: Int
    "large-scale (hypo)-viscosity coefficient"
       μ :: T
    "(hypo)-viscosity order, `nν ≤ 0`"
      nμ :: Int
    "function that calculates the forcing F̂"
  calcF! :: Function  # function that calculates the forcing F̂
end

Params(ν, nν) = Params(ν, nν, typeof(ν)(0), 0, nothingfunction)


# ---------
# Equations
# ---------

"""
    Equation(params, grid)

Returns the `equation` for two-dimensional Navier-Stokes with `params` and `grid`. The linear
opeartor ``L`` includes (hyper)-viscosity of order ``n_ν`` with coefficient ``ν`` and 
hypo-viscocity of order ``n_μ`` with coefficient ``μ``. Plain old viscocity corresponds to 
``n_ν=1`` while ``n_μ=0`` corresponds to linear drag.

```math
L = - ν |𝐤|^{2 n_ν} - μ |𝐤|^{2 n_μ} .
```

The nonlinear term is computed via function `calcN!()`.
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
    Vars{Aphys, Atrans, F, P}(ζ, u, v, ζh, uh, vh, Fh, prevsol)

Returns the vars for two-dimensional Navier-Stokes.
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

const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    Vars(dev, grid)

Returns the `vars` for unforced two-dimensional Navier-Stokes problem on device `dev` and 
with `grid`.
"""
function Vars(::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) ζ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) ζh uh vh
  return Vars(ζ, u, v, ζh, uh, vh, nothing, nothing)
end

"""
    ForcedVars(dev, grid)

Returns the vars for forced two-dimensional Navier-Stokes on device `dev` and with `grid`.
"""
function ForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) ζ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) ζh uh vh Fh
  return Vars(ζ, u, v, ζh, uh, vh, Fh, nothing)
end

"""
    StochasticForcedVars(dev, grid)

Returns the vars for stochastically forced two-dimensional Navier-Stokes on device `dev` and 
with `grid`.
"""
function StochasticForcedVars(dev::Dev, grid::AbstractGrid) where Dev
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

Calculates the Fourier transform of the advection term, ``- 𝖩(ψ, ζ)`` in conservative 
form, i.e., ``- ∂_y[(∂_x ψ)ζ] + ∂_x[(∂_y ψ)ζ]`` and stores it in `N`:

```math
N(ζ̂) = - \\widehat{𝖩(ψ, ζ)} = - i k_x \\widehat{u ζ} - i k_y \\widehat{v ζ} .
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

Calculates the nonlinear term, that is the advection term and the forcing,

```math
N(ζ̂) = - \\widehat{𝖩(ψ, ζ)} + F̂ ,
```

by calling `calcN_advection!` and `addforcing!`.
"""
function calcN!(N, sol, t, clock, vars, params, grid)
  calcN_advection!(N, sol, t, clock, vars, params, grid)
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

"""
    addforcing!(N, sol, t, clock, vars, params, grid)

When the problem includes forcing, calculate the forcing term ``F̂`` and add it to the 
nonlinear term ``N``.
"""
addforcing!(N, sol, t, clock, vars::Vars, params, grid) = nothing

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

Update variables in `vars` with solution in `sol`.
"""
function updatevars!(prob)
  vars, grid, sol = prob.vars, prob.grid, prob.sol
  
  @. vars.ζh = sol
  @. vars.uh =   im * grid.l  * grid.invKrsq * sol
  @. vars.vh = - im * grid.kr * grid.invKrsq * sol
  
  ldiv!(vars.ζ, grid.rfftplan, deepcopy(vars.ζh))
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))
  ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh))
  
  return nothing
end

"""
    set_ζ!(prob, ζ)

Set the solution `sol` as the transform of `ζ` and then update variables in `vars`.
"""
function set_ζ!(prob, ζ)
  mul!(prob.sol, prob.grid.rfftplan, ζ)
  CUDA.@allowscalar prob.sol[1, 1] = 0 # zero domain average
  
  updatevars!(prob)
  
  return nothing
end

# = \\sum_{𝐤} \\frac1{2} |𝐤|^2 |ψ̂|^2 .

"""
    energy(prob)

Returns the domain-averaged kinetic energy,
```math
\\int \\frac1{2} (u² + v²) \\frac{𝖽x 𝖽y}{L_x L_y} = \\int \\frac1{2} |{\\bf ∇} ψ|² \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} \\frac1{2} |𝐤|² |ψ̂|² .
```
"""
@inline function energy(prob)
  sol, vars, grid = prob.sol, prob.vars, prob.grid
  energyh = vars.uh # use vars.uh as scratch variable
  
  @. energyh = 1 / 2 * grid.invKrsq * abs2(sol)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energyh, grid)
end

"""
    enstrophy(prob)

Returns the domain-averaged enstrophy,
```math
\\int \\frac1{2} ζ² \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} \\frac1{2} |ζ̂|² .
```
"""
@inline function enstrophy(prob)
  sol, grid = prob.sol, prob.grid
  return 1 / (2 * grid.Lx * grid.Ly) * parsevalsum(abs2.(sol), grid)
end

"""
    energy_dissipation(prob)

Returns the domain-averaged energy dissipation rate done by the ``ν`` viscous term,
```math
- ν (-1)^{n_ν+1} \\int ψ ∇^{2 n_ν}ζ \\frac{𝖽x 𝖽y}{L_x L_y} = - ν \\sum_{𝐤} |𝐤|^{2(n_ν-1)} |ζ̂|² .
```
"""
@inline function energy_dissipation(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  energy_dissipationh = vars.uh # use vars.uh as scratch variable
  
  @. energy_dissipationh = - params.ν * grid.Krsq^(params.nν - 1) * abs2(sol)
  CUDA.@allowscalar energy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dissipationh, grid)
end

"""
    enstrophy_dissipation(prob)

Returns the domain-averaged enstrophy dissipation rate done by the ``ν`` viscous term,
```math
ν (-1)^{n_ν+1} \\int ζ ∇^{2 n_ν}ζ \\frac{𝖽x 𝖽y}{L_x L_y} = - ν \\sum_{𝐤} |𝐤|^{2n_ν} |ζ̂|² .
```
"""
@inline function enstrophy_dissipation(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  enstrophy_dissipationh = vars.uh # use vars.uh as scratch variable
  
  @. enstrophy_dissipationh = - params.ν * grid.Krsq^params.nν * abs2(sol)
  CUDA.@allowscalar enstrophy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dissipationh, grid)
end

"""
    energy_drag(prob)

Returns the extraction of domain-averaged energy done by the ``μ`` viscous term,
```math
- μ (-1)^{n_μ+1} \\int ψ ∇^{2 n_μ}ζ \\frac{𝖽x 𝖽y}{L_x L_y} = - ν \\sum_{𝐤} |𝐤|^{2(n_μ-1)} |ζ̂|² .
```
"""
@inline function energy_drag(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  
  energy_dragh = vars.uh # use vars.uh as scratch variable
  
  @. energy_dragh = - params.μ * grid.Krsq^(params.nμ - 1) * abs2(sol)
  CUDA.@allowscalar energy_dragh[1, 1] = 0
  
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dragh, grid)
end

"""
    enstrophy_drag(prob)

Returns the extraction of domain-averaged enstrophy by the ``μ`` viscous term,
```math
μ (-1)^{n_μ+1} \\int ζ ∇^{2 n_μ}ζ \\frac{𝖽x 𝖽y}{L_x L_y} = - μ \\sum_{𝐤} |𝐤|^{2n_μ} |ζ̂|² .
```
"""
@inline function enstrophy_drag(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid

  enstrophy_dragh = vars.uh # use vars.uh as scratch variable
  
  @. enstrophy_dragh = - params.μ * grid.Krsq^params.nμ * abs2(sol)
  CUDA.@allowscalar enstrophy_dragh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dragh, grid)
end

"""
    energy_work(prob)
    energy_work(sol, vars, grid)

Returns the domain-averaged rate of work of energy by the forcing ``F``,
```math
- \\int ψ F \\frac{𝖽x 𝖽y}{L_x L_y} = - \\sum_{𝐤} ψ̂ F̂^* .
```
"""
@inline function energy_work(sol, vars::ForcedVars, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable
  
  @. energy_workh = grid.invKrsq * sol * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_workh, grid)
end

@inline function energy_work(sol, vars::StochasticForcedVars, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable
  
  @. energy_workh = grid.invKrsq * (vars.prevsol + sol) / 2 * conj(vars.Fh) # Stratonovich
  # @. energy_workh = grid.invKrsq * vars.prevsol * conj(vars.Fh)           # Ito
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_workh, grid)
end

@inline energy_work(prob) = energy_work(prob.sol, prob.vars, prob.grid)

"""
    enstrophy_work(prob)
    enstrophy_work(sol, vars, grid)

Returns the domain-averaged rate of work of enstrophy by the forcing ``F``,
```math
\\int ζ F \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} ζ̂ F̂^* .
```
"""
@inline function enstrophy_work(sol, vars::ForcedVars, grid)
  enstrophy_workh = vars.uh # use vars.uh as scratch variable
  
  @. enstrophy_workh = sol * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_workh, grid)
end

@inline function enstrophy_work(sol, vars::StochasticForcedVars, grid)
  enstrophy_workh = vars.uh # use vars.uh as scratch variable
  
  @. enstrophy_workh = (vars.prevsol + sol) / 2 * conj(vars.Fh) # Stratonovich
  # @. enstrophy_workh = grid.invKrsq * vars.prevsol * conj(vars.Fh)           # Ito
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_workh, grid)
end

@inline enstrophy_work(prob) = enstrophy_work(prob.sol, prob.vars, prob.grid)

end # module
