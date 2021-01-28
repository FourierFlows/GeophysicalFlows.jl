module BarotropicQGQL

export
  Problem,
  set_zeta!,
  updatevars!,

  energy,
  enstrophy,
  dissipation,
  work,
  drag

using
  FFTW,
  CUDA,
  Reexport

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum, parsevalsum2
import FFTW: rfft

abstract type BarotropicQGQLVars <: AbstractVars end

nothingfunction(args...) = nothing

"""
    Problem(dev=CPU(); parameters...)

Construct a BarotropicQGQL turbulence problem on device `dev`.
"""
function Problem(dev::Device=CPU();
  # Numerical parameters
            nx = 256,
            Lx = 2π,
            ny = nx,
            Ly = Lx,
            dt = 0.01,
  # Physical parameters
             β = 0.0,
           eta = nothing,
  # Drag and/or hyper-/hypo-viscosity
             ν = 0.0,
            nν = 1,
             μ = 0.0,
  # Timestepper and equation options
       stepper = "RK4",
         calcF = nothingfunction,
    stochastic = false,
             T = Float64)

  # the grid
  grid = TwoDGrid(dev, nx, Lx, ny, Ly; T=T)
  x, y = gridpoints(grid)

  # topographic PV
  eta === nothing && ( eta = zeros(dev, T, (nx, ny)) )

  params = !(typeof(eta)<:ArrayType(dev)) ?
           Params(grid, β, eta, μ, ν, nν, calcF) :
           Params(β, eta, rfft(eta), μ, ν, nν, calcF)

  vars = calcF == nothingfunction ? DecayingVars(dev, grid) : stochastic ? StochasticForcedVars(dev, grid) : ForcedVars(dev, grid)

  equation = BarotropicQGQL.Equation(params, grid)
  
  FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
end


# ----------
# Parameters
# ----------

"""
    Params

A struct that contains all parameter values for a two-dimensional barotropic QG QL problem.
"""
struct Params{T, Aphys, Atrans} <: AbstractParams
       β :: T          # Planetary vorticity y-gradient
     eta :: Aphys      # Topographic PV
    etah :: Atrans     # FFT of Topographic PV
       μ :: T          # Linear drag
       ν :: T          # Viscosity coefficient
      nν :: Int        # Hyperviscous order (nν=1 is plain old viscosity)
  calcF! :: Function   # Function that calculates the forcing on QGPV q
end

"""
    Params(grid, β, eta::Function, μ, ν, nν, calcF)

Constructor for `params` that accepts a generating function for the topographic PV, `eta`.
"""
function Params(grid::AbstractGrid{T, A}, β, eta::Function, μ, ν, nν, calcF) where {T, A}
  eta_on_grid = FourierFlows.on_grid(eta, grid)
  etah_on_grid = rfft(eta_on_grid)
  
  return Params(β, A(eta_on_grid), A(etah_on_grid), μ, ν, nν, calcF)
end


# ---------
# Equations
# ---------

"""
    Equation(params, grid)

Return the equation for two-dimensional barotropic QG QL problem with parameters `params` and 
on `grid`. Linear operator ``L`` includes bottom drag ``μ``, (hyper)-viscosity of order ``n_ν``
with coefficient ``ν`` and the ``β`` term:
```math
L = - μ - ν |𝐤|^{2 n_ν} + i β k_x / |𝐤|² .
```
Nonlinear term is computed via `calcN!` function.
"""
function Equation(params::Params, grid::AbstractGrid)
  L = @. - params.μ - params.ν * grid.Krsq^params.nν + im * params.β * grid.kr * grid.invKrsq
  CUDA.@allowscalar L[1, 1] = 0
  
  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

"""
    Vars
A struct that contains all variables for a two-dimensional barotropic QG QL problem.
"""
mutable struct Vars{Aphys, Atrans, F, P} <: BarotropicQGQLVars
        u :: Aphys
        v :: Aphys
        U :: Aphys
    uzeta :: Aphys
    vzeta :: Aphys
     zeta :: Aphys
     Zeta :: Aphys
      psi :: Aphys
      Psi :: Aphys
       Nz :: Atrans
       NZ :: Atrans
       uh :: Atrans
       vh :: Atrans
       Uh :: Atrans
    zetah :: Atrans
    Zetah :: Atrans
     psih :: Atrans
     Psih :: Atrans
       Fh :: F
  prevsol :: P     
end

const DecayingVars = Vars{<:AbstractArray, <:AbstractArray, Nothing, Nothing}
const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    DecayingVars(dev, grid)

Return the vars for unforced two-dimensional barotropic QG problem with `grid`.
"""
function DecayingVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) u v U uzeta vzeta zeta Zeta psi Psi
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) N NZ uh vh Uh zetah Zetah psih Psih
  
  return Vars(u, v, U, uzeta, vzeta, zeta, Zeta, psi, Psi, N, NZ, uh, vh, Uh, zetah, Zetah, psih, Psih, nothing, nothing)
end

"""
    ForcedVars(dev, grid)

Return the `vars` for forced two-dimensional barotropic QG QL problem on device `dev` and `grid`.
"""
function ForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) u v U uzeta vzeta zeta Zeta psi Psi
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) N NZ uh vh Uh zetah Zetah psih Psih Fh
  
  return Vars(u, v, U, uzeta, vzeta, zeta, Zeta, psi, Psi, N, NZ, uh, vh, Uh, zetah, Zetah, psih, Psih, Fh, nothing)
end

"""
    StochasticForcedVars(dev, grid)

Return the `vars` for stochastically forced two-dimensional barotropic QG QL problem on 
device `dev` and `grid`.
"""
function StochasticForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) u v U uzeta vzeta zeta Zeta psi Psi
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) N NZ uh vh Uh zetah Zetah psih Psih Fh prevsol
  
  return Vars(u, v, U, uzeta, vzeta, zeta, Zeta, psi, Psi, N, NZ, uh, vh, Uh, zetah, Zetah, psih, Psih, Fh, prevsol)
end


# -------
# Solvers
# -------

function calcN_advection!(N, sol, t, clock, vars, params, grid)
  @. vars.zetah = sol
  CUDA.@allowscalar @. vars.zetah[1, :] = 0
  @. vars.Zetah = sol
  CUDA.@allowscalar @. vars.Zetah[2:end, :] = 0

  @. vars.uh =  im * grid.l  * grid.invKrsq * vars.zetah
  @. vars.vh = -im * grid.kr * grid.invKrsq * vars.zetah
  @. vars.Uh =  im * grid.l  * grid.invKrsq * vars.Zetah

  ldiv!(vars.zeta, grid.rfftplan, vars.zetah)
  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)

  ldiv!(vars.Zeta, grid.rfftplan, vars.Zetah)
  ldiv!(vars.U, grid.rfftplan, vars.Uh)

  @. vars.uzeta = vars.u * vars.zeta           # u*ζ
  @. vars.vzeta = vars.v * vars.zeta           # v*ζ

  mul!(vars.uh, grid.rfftplan, vars.uzeta)     # \hat{u*q}
  @. vars.NZ = -im * grid.kr * vars.uh         # -∂[u*q]/∂x
  mul!(vars.vh, grid.rfftplan, vars.vzeta)     # \hat{v*q}
  @. vars.NZ += - im * grid.l * vars.vh        # -∂[v*q]/∂y
  CUDA.@allowscalar @. vars.NZ[2:end, :] = 0

  @. vars.U = vars.U * vars.zeta                # U*ζ
  @. vars.u = vars.u * vars.Zeta                # u*Ζ
  @. vars.v = vars.v * vars.Zeta                # v*Ζ

  mul!(vars.uh, grid.rfftplan, vars.U + vars.u) # \hat{U*ζ + u*Ζ}
  @. vars.Nz = -im * grid.kr*vars.uh            # -∂[U*ζ + u*Ζ]/∂x
  mul!(vars.vh, grid.rfftplan, vars.v)          # \hat{v*Z}
  @. vars.Nz += - im * grid.l*vars.vh           # -∂[v*Z]/∂y
  CUDA.@allowscalar @. vars.Nz[1, :] = 0

  @. N = vars.NZ + vars.Nz
  
  return nothing
end

function calcN!(N, sol, t, clock, vars, params, grid)
  calcN_advection!(N, sol, t, clock, vars, params, grid)
  addforcing!(N, sol, t, clock, vars, params, grid)
  
  return nothing
end

addforcing!(N, sol, t, cl, vars::Vars, params, grid) = nothing

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
    updatevars!(sol, vars, params, grid)
    updatevars!(prob)

Update the `vars` of a problem `prob` that has `grid` and `params` with the solution in `sol`.
"""
function updatevars!(sol, vars, params, grid)
  CUDA.@allowscalar sol[1, 1] = 0
  @. vars.zetah = sol
  CUDA.@allowscalar @. vars.zetah[1, :] = 0
  @. vars.Zetah = sol
  CUDA.@allowscalar @. vars.Zetah[2:end, :] = 0

  @. vars.Psih = -vars.Zetah * grid.invKrsq
  @. vars.psih = -vars.zetah * grid.invKrsq
  @. vars.uh = -im * grid.l  * vars.psih
  @. vars.vh =  im * grid.kr * vars.psih
  @. vars.Uh =  im * grid.l  * vars.Zetah * grid.invKrsq

  ldiv!(vars.zeta, grid.rfftplan, deepcopy(vars.zetah))
  ldiv!(vars.Zeta, grid.rfftplan, deepcopy(vars.Zetah))
  ldiv!(vars.psi, grid.rfftplan, vars.psih)
  ldiv!(vars.Psi, grid.rfftplan, vars.Psih)
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))
  ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh))
  ldiv!(vars.U, grid.rfftplan, deepcopy(vars.Uh))

  return nothing
end

updatevars!(prob) = updatevars!(prob.sol, prob.vars, prob.params, prob.grid)

"""
    set_zeta!(prob, zeta)
    set_zeta!(sol, vars, grid, zeta)

Set the solution `sol` as the transform of `zeta` and update variables `vars` on the `grid`.
"""
function set_zeta!(sol, vars, params, grid, zeta)
  mul!(vars.zetah, grid.rfftplan, zeta)
  CUDA.@allowscalar vars.zetah[1, 1] = 0.0
  @. sol = vars.zetah

  updatevars!(sol, vars, params, grid)
  return nothing
end

set_zeta!(prob, zeta) = set_zeta!(prob.sol, prob.vars, prob.params, prob.grid, zeta)

"""
    energy(sol, grid)
    energy(prob)

Return the domain-averaged kinetic energy of `sol`.
"""
@inline energy(sol, grid::AbstractGrid) =
  0.5 * (parsevalsum2(grid.kr .* grid.invKrsq .* sol, grid)
        + parsevalsum2(grid.l .* grid.invKrsq .* sol, grid)) / (grid.Lx * grid.Ly)

energy(prob) = energy(prob.sol, prob.grid)

"""
    enstrophy(sol, grid, vars)
    enstrophy(prob)

Return the domain-averaged enstrophy of `sol`.
"""
function enstrophy(sol, grid::AbstractGrid, vars::AbstractVars)
  @. vars.uh = sol
  CUDA.@allowscalar vars.uh[1, 1] = 0
  
  return 0.5 * parsevalsum2(vars.uh, grid) / (grid.Lx * grid.Ly)
end
enstrophy(prob) = enstrophy(prob.sol, prob.grid, prob.vars)


"""
    dissipation(prob)
    dissipation(sol, v, p, g)

Return the domain-averaged dissipation rate. `nν` must be >= 1.
"""
@inline function dissipation(sol, vars, params, grid)
  @. vars.uh = grid.Krsq^(params.nν - 1) * abs2(sol)
  CUDA.@allowscalar vars.uh[1, 1] = 0
  
  return params.ν / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline dissipation(prob) = dissipation(prob.sol, prob.vars, prob.params, prob.grid)

"""
    work(prob)
    work(sol, vars, params, grid)

Return the domain-averaged rate of work of energy by the forcing, `params.Fh`.
"""
@inline function work(sol, vars::ForcedVars, grid)
  @. vars.uh = grid.invKrsq * sol * conj(vars.Fh)
  
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline function work(sol, vars::StochasticForcedVars, grid)
  @. vars.uh = grid.invKrsq * (vars.prevsol + sol) / 2 * conj(vars.Fh)
  
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline work(prob) = work(prob.sol, prob.vars, prob.grid)

"""
    drag(prob)
    drag(sol, vars, params, grid)

Return the extraction of domain-averaged energy by drag `μ`.
"""
@inline function drag(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  @. vars.uh = grid.invKrsq * abs2(sol)
  CUDA.@allowscalar vars.uh[1, 1] = 0
  
  return params.μ / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

end # module
