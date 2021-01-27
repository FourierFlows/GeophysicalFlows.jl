module SurfaceQG

export
  Problem,
  set_b!,
  updatevars!,

  kinetic_energy,
  buoyancy_variance,
  buoyancy_dissipation,
  buoyancy_work,
  buoyancy_advection,
  kinetic_energy_advection

  using
    CUDA,
    Reexport

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum

nothingfunction(args...) = nothing

"""
    Problem(; parameters...)

Construct a Surface QG turbulence problem.
"""
function Problem(dev::Device=CPU();
  # Numerical parameters
          nx = 256,
          Lx = 2π,
          ny = nx,
          Ly = Lx,
          dt = 0.01,
  # Hyper-viscosity parameters
           ν = 0,
          nν = 1,
  # Timestepper and equation options
     stepper = "RK4",
       calcF = nothingfunction,
  stochastic = false,
           T = Float64)

  grid = TwoDGrid(dev, nx, Lx, ny, Ly; T=T)

  params = Params{T}(ν, nν, calcF)

  vars = calcF == nothingfunction ? Vars(dev, grid) : (stochastic ? StochasticForcedVars(dev, grid) : ForcedVars(dev, grid))

  equation = Equation(params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
end


# ----------
# Parameters
# ----------

"""
    Params(ν, nν, calcF!)

Returns the params for Surface QG turbulence.
"""
struct Params{T} <: AbstractParams
       ν :: T         # Buoyancy viscosity coefficient
      nν :: Int       # Buoyancy hyperviscous order
  calcF! :: Function  # Calculates the Fourier transform of the buoyancy forcing `Fh`
end

Params(ν, nν) = Params(ν, nν, nothingfunction)


# ---------
# Equations
# ---------

"""
    Equation(params, grid)

Return the equation for Surface QG turbulence with `params` and `grid`. The linear operator 
``L`` includes (hyper)-viscosity of order ``n_ν`` with coefficient ``ν``,

```math
L = - ν |𝐤|^{2 n_ν} .
```

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

struct Vars{Aphys, Atrans, F, P} <: SurfaceQGVars
    "buoyancy"
        b :: Aphys
    "x-component of velocity"
        u :: Aphys
    "y-component of velocity"
        v :: Aphys
    "Fourier transform of buoyancy"
       bh :: Atrans
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

Return the vars for unforced surface QG turbulence on device dev and with `grid`.
"""
function Vars(::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) b u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) bh uh vh
  
  return Vars(b, u, v, bh, uh, vh, nothing, nothing)
end

"""
    ForcedVars(dev, grid)

Return the vars for forced surface QG turbulence on device `dev` and with `grid`.
"""
function ForcedVars(dev::Dev, grid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) b u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) bh uh vh Fh
  
  return Vars(b, u, v, bh, uh, vh, Fh, nothing)
end

"""
    StochasticForcedVars(dev, grid)

Return the `vars` for stochastically forced surface QG turbulence on device `dev` and with `grid`.
"""
function StochasticForcedVars(dev::Dev, grid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) b u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) bh uh vh Fh prevsol
  
  return Vars(b, u, v, bh, uh, vh, Fh, prevsol)
end


# -------
# Solvers
# -------

"""
    calcN_advection(N, sol, t, clock, vars, params, grid)

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
  
  @. ub *= vars.b # u*b
  @. vb *= vars.b # v*b

  mul!(ubh, grid.rfftplan, ub) # \hat{u*b}
  mul!(vbh, grid.rfftplan, vb) # \hat{v*b}

  @. N = - im * grid.kr * ubh - im * grid.l * vbh
  
  return nothing
end

function calcN!(N, sol, t, clock, vars, params, grid)
  calcN_advection!(N, sol, t, clock, vars, params, grid)
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

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

Set the solution `sol` as the transform of b and update all variables.
"""
function set_b!(prob, b)
  mul!(prob.sol, prob.grid.rfftplan, b)
  CUDA.@allowscalar prob.sol[1, 1] = 0 # zero domain average
  
  updatevars!(prob)
  
  return nothing
end

"""
    kinetic_energy(prob)

Returns the domain-averaged surface kinetic energy. In SQG, this is
identical to half the domain-averaged surface buoyancy variance.
"""
@inline function kinetic_energy(prob)
  sol, vars, grid = prob.sol, prob.vars, prob.grid
  
  @. vars.uh =   im * grid.l  * sqrt(grid.invKrsq) * sol
  @. vars.vh = - im * grid.kr * sqrt(grid.invKrsq) * sol
  
  kinetic_energyh = vars.bh         # use vars.bh as scratch variable

  @. kinetic_energyh = 0.5 * (abs2(vars.uh) + abs2(vars.vh)) # ½(|û|²+|v̂|²)
  
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(kinetic_energyh, grid)
end

"""
    buoyancy_variance(prob)

Returns the domain-averaged buoyancy variance. In SQG flows this is identical to
the domain-averaged velocity variance (twice the kinetic energy)
"""
@inline function buoyancy_variance(prob)
  sol, grid = prob.sol, prob.grid
  
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(abs2.(sol), grid)
end

"""
    buoyancy_dissipation(prob)

Returns the domain-averaged dissipation rate of surface buoyancy variance due
to small scale diffusion/viscosity. nν must be >= 1.

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

Returns the domain-averaged rate of work of buoyancy variance by the forcing Fh.
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
