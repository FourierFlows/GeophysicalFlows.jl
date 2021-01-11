module ShallowWater

export
  Problem,
  updatevars!,

  energy

using
  CUDA,
  Reexport

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum

nothingfunction(args...) = nothing

"""
    Problem(; parameters...)

Construct a 2D shallow water problem.
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

Returns the params for two-dimensional turbulence.
"""
struct Params{T} <: AbstractParams
       ν :: T         # Viscosity
      nν :: Int       # (Hyper)-viscous order
  calcF! :: Function  # Function that calculates the forcing `Fh`
end


# ---------
# Equations
# ---------

"""
    Equation(params, grid)

Returns the equation for two-dimensional turbulence with `params` and `grid`.
"""
function Equation(params::Params, grid::AbstractGrid)
  D = @. - params.ν * grid.Krsq^params.nν
  CUDA.@allowscalar D[1, 1] = 0
  
  L = zeros(dev, T, (grid.nkr, grid.nl, 3))
  
  L[:, :, 1] .= D # for qu equation
  L[:, :, 2] .= D # for qv equation
  L[:, :, 3] .= D # for h equation
  
  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

abstract type ShallowWaterVars <: AbstractVars end

struct Vars{Aphys, Atrans, S, F, P} <: ShallowWaterVars
       qu :: Aphys
       qv :: Aphys
        u :: Aphys
        v :: Aphys
        h :: Aphys
      quh :: Atrans
      qvh :: Atrans
       hh :: Atrans
    state :: S
       Fh :: F
  prevsol :: P
end

const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    Vars(dev, grid)

Returns the `vars` for unforced two-dimensional turbulence on device `dev` and with `grid`.
"""
function Vars(::Dev, grid::TwoDGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) qu qv u v h
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) quh qvh hh
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, 3) state
  return Vars(qu, qv, u, v, h, uh, vh, hh, state, nothing, nothing)
end

"""
    ForcedVars(dev, grid)

Returns the vars for forced two-dimensional turbulence on device `dev` and with `grid`.
"""
function ForcedVars(dev::Dev, grid::TwoDGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) qu qv u v h
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) quh qvh hh Fh
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, 3) state
  return Vars(qu, qv, u, v, h, uh, vh, hh, state, Fh, nothing)
end

"""
    StochasticForcedVars(dev, grid)

Returns the vars for stochastically forced two-dimensional turbulence on device `dev` and with `grid`.
"""
function StochasticForcedVars(dev::Dev, grid::TwoDGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) qu qv u v h
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) quh qvh hh Fh prevsol
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, 3) state
  return Vars(qu, qv, u, v, h, uh, vh, hh, state, Fh, prevsol)
end


# -------
# Solvers
# -------

"""
    calcN_advection(N, sol, t, clock, vars, params, grid)

Calculates the advection term.
"""
function calcN_advection!(N, sol, t, clock, vars, params, grid)
  state = vars.state
  state = (qu = view(sol, :, :, 1), qv = view(sol, :, :, 2), h = view(sol, :, :, 3))
  
  @. N = 0 * sol
  
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
  
  @. vars.quh = sol[:, :, 1]
  @. vars.qvh = sol[:, :, 2]
  @. vars.hh  = sol[:, :, 3]
  
  ldiv!(vars.qu, grid.rfftplan, deepcopy(sol[:, :, 1])) # use deepcopy() because irfft destroys its input
  ldiv!(vars.qv, grid.rfftplan, deepcopy(sol[:, :, 2])) # use deepcopy() because irfft destroys its input
  ldiv!(vars.h, grid.rfftplan, deepcopy(sol[:, :, 3])) # use deepcopy() because irfft destroys its input

  @. vars.u = vars.qu / vars.h
  @. vars.v = vars.qv / vars.h
  
  return nothing
end

end # module
