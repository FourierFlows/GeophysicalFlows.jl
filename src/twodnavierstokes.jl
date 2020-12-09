module TwoDNavierStokes

export
  Problem,
  set_zeta!,
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
    Problem(; parameters...)

Construct a 2D turbulence problem.
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
    Params(ν, nν, μ, nμ, calcF!)

Returns the params for two-dimensional turbulence.
"""
struct Params{T} <: AbstractParams
       ν :: T         # Vorticity viscosity
      nν :: Int       # Vorticity hyperviscous order
       μ :: T         # Bottom drag or hypoviscosity
      nμ :: Int       # Order of hypodrag
  calcF! :: Function  # Function that calculates the forcing F
end

Params(ν, nν) = Params(ν, nν, typeof(ν)(0), 0, nothingfunction)


# ---------
# Equations
# ---------

"""
    Equation(params, grid)

Returns the equation for two-dimensional turbulence with `params` and `grid`.
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

struct Vars{Aphys, Atrans, F, P} <: TwoDNavierStokesVars
     zeta :: Aphys
        u :: Aphys
        v :: Aphys
    zetah :: Atrans
       uh :: Atrans
       vh :: Atrans
       Fh :: F
  prevsol :: P
end

const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    Vars(dev, grid)

Returns the `vars` for unforced two-dimensional turbulence on device `dev` and with `grid`.
"""
function Vars(::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) zeta u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) zetah uh vh
  return Vars(zeta, u, v, zetah, uh, vh, nothing, nothing)
end

"""
    ForcedVars(dev, grid)

Returns the vars for forced two-dimensional turbulence on device `dev` and with `grid`.
"""
function ForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) zeta u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) zetah uh vh Fh
  return Vars(zeta, u, v, zetah, uh, vh, Fh, nothing)
end

"""
    StochasticForcedVars(dev, grid)

Returns the vars for stochastically forced two-dimensional turbulence on device `dev` and with `grid`.
"""
function StochasticForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) zeta u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) zetah uh vh Fh prevsol
  return Vars(zeta, u, v, zetah, uh, vh, Fh, prevsol)
end


# -------
# Solvers
# -------

"""
    calcN_advection(N, sol, t, clock, vars, params, grid)

Calculates the advection term.
"""
function calcN_advection!(N, sol, t, clock, vars, params, grid)
  @. vars.uh =   im * grid.l  * grid.invKrsq * sol
  @. vars.vh = - im * grid.kr * grid.invKrsq * sol
  @. vars.zetah = sol

  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)
  ldiv!(vars.zeta, grid.rfftplan, vars.zetah)
  
  uζ = vars.u        # use vars.u as scratch variable
  @. uζ *= vars.zeta # u*zeta
  vζ = vars.v        # use vars.v as scratch variable
  @. vζ *= vars.zeta # v*zeta
  
  uζh = vars.uh
  mul!(uζh, grid.rfftplan, uζ) # \hat{u*zeta}
  vζh = vars.vh 
  mul!(vζh, grid.rfftplan, vζ) # \hat{v*zeta}

  @. N = - im * grid.kr * uζh - im * grid.l * vζh
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
  @. vars.zetah = sol
  @. vars.uh =   im * grid.l  * grid.invKrsq * sol
  @. vars.vh = - im * grid.kr * grid.invKrsq * sol
  ldiv!(vars.zeta, grid.rfftplan, deepcopy(vars.zetah))
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))
  ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh))
  return nothing
end

"""
    set_zeta!(prob, zeta)

Set the solution `sol` as the transform of `zeta` and update variables.
"""
function set_zeta!(prob, zeta)
  mul!(prob.sol, prob.grid.rfftplan, zeta)
  CUDA.@allowscalar prob.sol[1, 1] = 0 # zero domain average
  
  updatevars!(prob)
  
  return nothing
end

"""
    energy(prob)

Returns the domain-averaged kinetic energy, ∫ ½(u²+v²)dxdy / (Lx Ly), for the solution in `sol`.
"""
@inline function energy(prob)
  sol, vars, grid = prob.sol, prob.vars, prob.grid
  energyh = vars.uh # use vars.uh as scratch variable
  
  @. energyh = 1 / 2 * grid.invKrsq * abs2(sol)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energyh, grid)
end

"""
    enstrophy(prob)

Returns the domain-averaged enstrophy, ∫ ½ ζ² dxdy / (Lx Ly), for the solution in `sol`.
"""
@inline function enstrophy(prob)
  sol, grid = prob.sol, prob.grid
  return 1 / (2 * grid.Lx * grid.Ly) * parsevalsum(abs2.(sol), grid)
end

"""
    energy_dissipation(prob)

Returns the domain-averaged energy dissipation rate. `nν` must be >= 1.
"""
@inline function energy_dissipation(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  energy_dissipationh = vars.uh # use vars.uh as scratch variable
  
  @. energy_dissipationh = params.ν * grid.Krsq^(params.nν - 1) * abs2(sol)
  CUDA.@allowscalar energy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dissipationh, grid)
end

"""
    enstrophy_dissipation(prob)

Returns the domain-averaged enstrophy dissipation rate. `nν` must be >= 1.
"""
@inline function enstrophy_dissipation(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  enstrophy_dissipationh = vars.uh # use vars.uh as scratch variable
  
  @. enstrophy_dissipationh = params.ν * grid.Krsq^params.nν * abs2(sol)
  CUDA.@allowscalar enstrophy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dissipationh, grid)
end

"""
    energy_work(prob)
    energy_work(sol, vars, grid)

Returns the domain-averaged rate of work of energy by the forcing `Fh`.
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

Returns the domain-averaged rate of work of enstrophy by the forcing `Fh`.
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

"""
    energy_drag(prob)

Returns the extraction of domain-averaged energy by drag/hypodrag μ.
"""
@inline function energy_drag(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  
  energy_dragh = vars.uh # use vars.uh as scratch variable
  
  @. energy_dragh = params.μ * grid.Krsq^(params.nμ - 1) * abs2(sol)
  CUDA.@allowscalar energy_dragh[1, 1] = 0
  
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dragh, grid)
end

"""
    enstrophy_drag(prob)

Returns the extraction of domain-averaged enstrophy by drag/hypodrag μ.
"""
@inline function enstrophy_drag(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid

  enstrophy_dragh = vars.uh # use vars.uh as scratch variable
  
  @. enstrophy_dragh = params.μ * grid.Krsq^params.nμ * abs2(sol)
  CUDA.@allowscalar enstrophy_dragh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dragh, grid)
end

end # module
