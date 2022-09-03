module SurfaceQG

export
  Problem,
  set_b!,
  updatevars!,

  kinetic_energy,
  buoyancy_variance,
  buoyancy_dissipation,
  buoyancy_work

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
                     Lx = 2π,
                     ny = nx,
                     Ly = Lx,
                      ν = 0,
                     nν = 1,
                     dt = 0.01,
                stepper = "RK4",
                  calcF = nothingfunction,
             stochastic = false,
       aliased_fraction = 1/3,
                      T = Float64)

Construct a surface quasi-geostrophic problem on device `dev`.

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
  # Hyper-viscosity parameters
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

  params = Params{T}(ν, nν, calcF)

  vars = calcF == nothingfunction ? DecayingVars(grid) : (stochastic ? StochasticForcedVars(grid) : ForcedVars(grid))

  equation = Equation(params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params)
end


# ----------
# Parameters
# ----------

"""
    Params{T}(ν, nν, calcF!)

A struct containing the parameters for Surface QG dynamics. Included are:

$(TYPEDFIELDS)
"""
struct Params{T} <: AbstractParams
    "buoyancy (hyper)-viscosity coefficient"
       ν :: T
    "buoyancy (hyper)-viscosity order"
      nν :: Int
    "function that calculates the Fourier transform of the forcing, ``F̂``"
  calcF! :: Function
end

Params(ν, nν) = Params(ν, nν, nothingfunction)


# ---------
# Equations
# ---------

"""
    Equation(params, grid)

Return the `equation` for surface QG dynamics with `params` and `grid`. The linear 
opeartor ``L`` includes (hyper)-viscosity of order ``n_ν`` with coefficient ``ν``,

```math
L = - ν |𝐤|^{2 n_ν} .
```

Plain old viscocity corresponds to ``n_ν=1``.

The nonlinear term is computed via function `calcN!()`.
"""
function Equation(params::Params, grid::AbstractGrid)
  L = @. - params.ν * grid.Krsq^params.nν
  CUDA.@allowscalar L[1, 1] = 0
  
  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

abstract type SurfaceQGVars <: AbstractVars end

"""
    Vars{Aphys, Atrans, F, P}(b, u, v, bh, uh, vh, Fh, prevsol)

The variables for surface QG problem:

$(FIELDS)
"""
struct Vars{Aphys, Atrans, F, P} <: SurfaceQGVars
    "buoyancy"
        b :: Aphys
    "``x``-component of velocity"
        u :: Aphys
    "``y``-component of velocity"
        v :: Aphys
    "Fourier transform of buoyancy"
       bh :: Atrans
    "Fourier transform of ``x``-component of velocity"
       uh :: Atrans
    "Fourier transform of ``y``-component of velocity"
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
    DecayingVars(grid)

Return the variables for unforced surface QG dynamics on `grid`.
"""
function DecayingVars(grid::AbstractGrid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) b u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) bh uh vh

  return Vars(b, u, v, bh, uh, vh, nothing, nothing)
end

"""
    ForcedVars(grid)

Return the variables for forced surface QG dynamics on `grid`.
"""
function ForcedVars(grid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) b u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) bh uh vh Fh
  
  return Vars(b, u, v, bh, uh, vh, Fh, nothing)
end

"""
    StochasticForcedVars(grid)

Return the variables for stochastically forced surface QG dynamics on `grid`.
"""
function StochasticForcedVars(grid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) b u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) bh uh vh Fh prevsol
  
  return Vars(b, u, v, bh, uh, vh, Fh, prevsol)
end


# -------
# Solvers
# -------

"""
    calcN_advection!(N, sol, t, clock, vars, params, grid)

Calculate the Fourier transform of the advection term, ``- 𝖩(ψ, b)`` in conservative 
form, i.e., ``- ∂_x[(∂_y ψ)b] - ∂_y[(∂_x ψ)b]`` and store it in `N`:

```math
N = - \\widehat{𝖩(ψ, b)} = - i k_x \\widehat{u b} - i k_y \\widehat{v b} .
```
"""
function calcN_advection!(N, sol, t, clock, vars, params, grid)
  @. vars.bh = sol
  @. vars.uh =   im * grid.l  * sqrt(grid.invKrsq) * sol
  @. vars.vh = - im * grid.kr * sqrt(grid.invKrsq) * sol

  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)
  ldiv!(vars.b, grid.rfftplan, vars.bh)
  
  ub, ubh = vars.u, vars.uh         # use vars.u, vars.uh as scratch variables
  vb, vbh = vars.v, vars.vh         # use vars.v, vars.vh as scratch variables
  
  @. ub *= vars.b                   # u*b
  @. vb *= vars.b                   # v*b

  mul!(ubh, grid.rfftplan, ub)      # \hat{u*b}
  mul!(vbh, grid.rfftplan, vb)      # \hat{v*b}

  @. N = - im * grid.kr * ubh - im * grid.l * vbh
  
  return nothing
end

"""
    calcN!(N, sol, t, clock, vars, params, grid)

Calculate the nonlinear term, that is the advection term and the forcing,

```math
N = - \\widehat{𝖩(ψ, b)} + F̂ .
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
  
  dealias!(sol, grid)
  
  @. vars.bh = sol
  @. vars.uh =   im * grid.l  * sqrt(grid.invKrsq) * sol
  @. vars.vh = - im * grid.kr * sqrt(grid.invKrsq) * sol
  
  ldiv!(vars.b, grid.rfftplan, deepcopy(vars.bh))
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))
  ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh))
  
  return nothing
end

"""
    set_b!(prob, b)

Set the solution `sol` as the transform of `b` and update all variables.
"""
function set_b!(prob, b)
  mul!(prob.sol, prob.grid.rfftplan, b)
  CUDA.@allowscalar prob.sol[1, 1] = 0 # zero domain average
  
  updatevars!(prob)
  
  return nothing
end

"""
    kinetic_energy(prob)

Return the domain-averaged surface kinetic energy. Since ``u² + v² = |{\\bf ∇} ψ|²``, we get
```math
\\int \\frac1{2} |{\\bf ∇} ψ|² \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} \\frac1{2} |𝐤|² |ψ̂|² .
```
In SQG, this is identical to half the domain-averaged surface buoyancy variance.
"""
@inline function kinetic_energy(prob)
  sol, vars, grid = prob.sol, prob.vars, prob.grid

  ψh = vars.uh                     # use vars.uh as scratch variable
  kinetic_energyh = vars.bh        # use vars.bh as scratch variable
  
  @. ψh = sqrt(grid.invKrsq) * sol
  @. kinetic_energyh = 1 / 2 * grid.Krsq * abs2(ψh)
  
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(kinetic_energyh, grid)
end

"""
    buoyancy_variance(prob)

Return the buoyancy variance,
```math
\\int b² \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} |b̂|² .
```
In SQG, this is identical to the velocity variance (i.e., twice the domain-averaged kinetic 
energy).
"""
@inline function buoyancy_variance(prob)
  sol, grid = prob.sol, prob.grid
  
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(abs2.(sol), grid)
end

"""
    buoyancy_dissipation(prob)

Return the domain-averaged dissipation rate of surface buoyancy variance due
to small scale (hyper)-viscosity,
```math
2 ν (-1)^{n_ν} \\int b ∇^{2n_ν} b \\frac{𝖽x 𝖽y}{L_x L_y} = - 2 ν \\sum_{𝐤} |𝐤|^{2n_ν} |b̂|² ,
```
where ``ν`` the (hyper)-viscosity coefficient ``ν`` and ``nν`` the (hyper)-viscosity order.
In SQG, this is identical to twice the rate of kinetic energy dissipation
"""
@inline function buoyancy_dissipation(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  buoyancy_dissipationh = vars.uh         # use vars.uh as scratch variable
  
  @. buoyancy_dissipationh = 2 * params.ν * grid.Krsq^params.nν * abs2(sol)
  CUDA.@allowscalar buoyancy_dissipationh[1, 1] = 0
  
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(buoyancy_dissipationh, grid)
end

"""
    buoyancy_work(prob)
    buoyancy_work(sol, vars, grid)

Return the domain-averaged rate of work of buoyancy variance by the forcing,
```math
\\int 2 b F \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} 2 b̂ F̂^* .
```
"""
@inline function buoyancy_work(sol, vars::ForcedVars, grid)
  buoyancy_workh = vars.uh         # use vars.uh as scratch variable

  @. buoyancy_workh =  2 * sol * conj(vars.Fh)    # 2*b̂*conj(f̂)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(buoyancy_workh, grid)
end

@inline function buoyancy_work(sol, vars::StochasticForcedVars, grid)
  buoyancy_workh = vars.uh         # use vars.uh as scratch variable
  
  @. buoyancy_workh =  (vars.prevsol + sol) * conj(vars.Fh) # Stratonovich
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(buoyancy_workh, grid)
end

@inline buoyancy_work(prob) = buoyancy_work(prob.sol, prob.vars, prob.grid)

end # module
