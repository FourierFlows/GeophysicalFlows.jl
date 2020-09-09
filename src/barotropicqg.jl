module BarotropicQG

export
  Problem,
  set_zeta!,
  updatevars!,

  energy,
  enstrophy,
  meanenergy,
  meanenstrophy,
  dissipation,
  work,
  drag,
  drag_ens,
  work_ens,
  dissipation_ens

using
  CUDA,
  Reexport

@reexport using FourierFlows

using FFTW: rfft
using LinearAlgebra: mul!, ldiv!
using FourierFlows: getfieldspecs, parsevalsum, parsevalsum2

nothingfunction(args...) = nothing

"""
    Problem(; parameters...)

Construct a BarotropicQG turbulence problem.
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
      calcFU = nothingfunction,
      calcFq = nothingfunction,
  stochastic = false,
           T = Float64)

  # the grid
  grid = TwoDGrid(dev, nx, Lx, ny, Ly; T=T)
  x, y = gridpoints(grid)

  # topographic PV
  eta === nothing && ( eta = zeros(dev, T, (nx, ny)) )

  params = !(typeof(eta)<:ArrayType(dev)) ? Params(grid, β, eta, μ, ν, nν, calcFU, calcFq) : Params(β, eta, rfft(eta), μ, ν, nν, calcFU, calcFq)

  vars = (calcFq == nothingfunction && calcFU == nothingfunction) ? Vars(dev, grid) : (stochastic ? StochasticForcedVars(dev, grid) : ForcedVars(dev, grid))

  equation = Equation(params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
end


# ----------
# Parameters
# ----------

"""
    Params(g::TwoDGrid, β, FU, eta, μ, ν, nν, calcFU, calcFq)

Returns the params for an unforced two-dimensional barotropic QG problem.
"""
struct Params{T, Aphys, Atrans} <: AbstractParams
        β :: T            # Planetary vorticity y-gradient
      eta :: Aphys        # Topographic PV
     etah :: Atrans       # FFT of Topographic PV
        μ :: T            # Linear drag
        ν :: T            # Viscosity coefficient
       nν :: Int          # Hyperviscous order (nν=1 is plain old viscosity)
   calcFU :: Function     # Function that calculates the forcing F(t) on
                          # domain-averaged zonal flow U(t)
  calcFq! :: Function     # Function that calculates the forcing on QGPV q
end

"""
    Params(g::TwoDGrid, β, eta::Function, μ, ν, nν, calcFU, calcFq)

Constructor for Params that accepts a generating function for the topographic PV.
"""
function Params(grid::AbstractGrid{T, A}, β, eta::Function, μ, ν, nν::Int, calcFU, calcFq) where {T, A}
  etagrid = A([eta(grid.x[i], grid.y[j]) for i=1:grid.nx, j=1:grid.ny])
     etah = rfft(etagrid)
  return Params(β, etagrid, etah, μ, ν, nν, calcFU, calcFq)
end


# ---------
# Equations
# ---------

"""
    Equation(params, grid)

Returns the equation for two-dimensional barotropic QG problem with `params` and `grid`.
"""
function Equation(params::Params, grid::AbstractGrid)
  L = @. - params.μ - params.ν * grid.Krsq^params.nν + im * params.β * grid.kr * grid.invKrsq
  CUDA.@allowscalar L[1, 1] = 0
  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

abstract type BarotropicQGVars <: AbstractVars end

struct Vars{Ascalar, Aphys, Atrans, F, P} <: BarotropicQGVars
        U :: Ascalar
        q :: Aphys
     zeta :: Aphys
      psi :: Aphys
        u :: Aphys
        v :: Aphys
       qh :: Atrans
    zetah :: Atrans
     psih :: Atrans
       uh :: Atrans
       vh :: Atrans
      Fqh :: F
  prevsol :: P
end

const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    Vars(dev, grid)

Returns the vars for unforced two-dimensional barotropic QG problem on device `dev` and with `grid`
"""
function Vars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  U = ArrayType(dev, T, 0)(undef, ); CUDA.@allowscalar U[] = 0
  @devzeros Dev T (grid.nx, grid.ny) q u v psi zeta
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh uh vh psih zetah
  Vars(U, q, zeta, psi, u, v, qh, zetah, psih, uh, vh, nothing, nothing)
end

"""
    ForcedVars(dev, grid)

Returns the vars for forced two-dimensional barotropic QG problem on device dev and with `grid`.
"""
function ForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  U = ArrayType(dev, T, 0)(undef, ); CUDA.@allowscalar U[] = 0
  @devzeros Dev T (grid.nx, grid.ny) q u v psi zeta
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh uh vh psih zetah Fqh
  return Vars(U, q, zeta, psi, u, v, qh, zetah, psih, uh, vh, Fqh, nothing)
end

"""
    StochasticForcedVars(dev, grid)

Returns the vars for stochastically forced two-dimensional barotropic QG problem on device dev and with `grid`.
"""
function StochasticForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  U = ArrayType(dev, T, 0)(undef, ); CUDA.@allowscalar U[] = 0
  @devzeros Dev T (grid.nx, grid.ny) q u v psi zeta
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh uh vh psih zetah Fqh prevsol
  return Vars(U, q, zeta, psi, u, v, qh, zetah, psih, uh, vh, Fqh, prevsol)
end


# -------
# Solvers
# -------

function calcN_advection!(N, sol, t, clock, vars, params, grid)
  # Note that U = sol[1, 1]. For all other elements ζ = sol
  CUDA.@allowscalar vars.U[] = sol[1, 1].re
  @. vars.zetah = sol
  CUDA.@allowscalar vars.zetah[1, 1] = 0

  @. vars.uh =  im * grid.l  * grid.invKrsq * vars.zetah
  @. vars.vh = -im * grid.kr * grid.invKrsq * vars.zetah

  ldiv!(vars.zeta, grid.rfftplan, vars.zetah)
  ldiv!(vars.u, grid.rfftplan, vars.uh)
  vars.psih .= vars.vh # FFTW's irfft destroys its input; vars.vh is needed for N[1, 1]
  ldiv!(vars.v, grid.rfftplan, vars.psih)

  @. vars.q = vars.zeta + params.eta
  CUDA.@allowscalar @. vars.u = (vars.U[] + vars.u) * vars.q # (U+u)*q
  @. vars.v = vars.v * vars.q # v*q

  mul!(vars.uh, grid.rfftplan, vars.u)      # \hat{(u+U)*q}

  # Nonlinear advection term for q (part 1)
  @. N = -im * grid.kr * vars.uh            # -∂[(U+u)q]/∂x
  mul!(vars.uh, grid.rfftplan, vars.v)      # \hat{v*q}
  @. N += - im * grid.l * vars.uh           # -∂[vq]/∂y
  return nothing
end

function calcN!(N, sol, t, clock, vars, params, grid)
  calcN_advection!(N, sol, t, clock, vars, params, grid)
  addforcing!(N, sol, t, clock, vars, params, grid)
  if params.calcFU != nothingfunction
    # 'Nonlinear' term for U with topographic correlation.
    # Note: < v*eta > =   sum(conj(vh)*eta) / (nx²*ny²) if  fft is used
    # while < v*eta > = 2*sum(conj(vh)*eta) / (nx²*ny²) if rfft is used
    CUDA.@allowscalar N[1, 1] = params.calcFU(t) + 2 * sum(conj(vars.vh) .* params.etah).re / (grid.nx^2 * grid.ny^2)
  end
  return nothing
end

addforcing!(N, sol, t, clock, vars::Vars, params, grid) = nothing

function addforcing!(N, sol, t, clock, vars::ForcedVars, params, grid)
  params.calcFq!(vars.Fqh, sol, t, clock, vars, params, grid)
  @. N += vars.Fqh
  return nothing
end

function addforcing!(N, sol, t, clock, vars::StochasticForcedVars, params, grid)
  if t == clock.t # not a substep
    @. vars.prevsol = sol # sol at previous time-step is needed to compute budgets for stochastic forcing
    params.calcFq!(vars.Fqh, sol, t, clock, vars, params, grid)
  end
  @. N += vars.Fqh
  return nothing
end


# ----------------
# Helper functions
# ----------------

"""
    updatevars!(sol, vars, params, grid)

Update the variables in `vars` with the solution in `sol`.
"""
function updatevars!(sol, vars, params, grid)
  CUDA.@allowscalar vars.U[] = sol[1, 1].re
  @. vars.zetah = sol
  CUDA.@allowscalar vars.zetah[1, 1] = 0.0

  @. vars.psih = - vars.zetah * grid.invKrsq
  @. vars.uh = - im * grid.l  * vars.psih
  @. vars.vh =   im * grid.kr * vars.psih

  ldiv!(vars.zeta, grid.rfftplan, deepcopy(vars.zetah))
  ldiv!(vars.psi, grid.rfftplan, deepcopy(vars.psih))
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))
  ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh))

  @. vars.q = vars.zeta + params.eta
  return nothing
end

updatevars!(prob) = updatevars!(prob.sol, prob.vars, prob.params, prob.grid)

"""
    set_zeta!(prob, zeta)
    set_zeta!(sol, vars, params, grid)

Set the solution `sol` as the transform of zeta and update variables `vars`
on the `grid`.
"""
function set_zeta!(sol, vars::Vars, params, grid, zeta)
  mul!(vars.zetah, grid.rfftplan, zeta)
  CUDA.@allowscalar vars.zetah[1, 1] = 0.0
  @. sol = vars.zetah

  updatevars!(sol, vars, params, grid)
  return nothing
end

function set_zeta!(sol, vars::Union{ForcedVars, StochasticForcedVars}, params, grid, zeta)
  CUDA.@allowscalar vars.U[] = deepcopy(sol[1, 1])
  mul!(vars.zetah, grid.rfftplan, zeta)
  CUDA.@allowscalar vars.zetah[1, 1] = 0.0
  @. sol = vars.zetah
  CUDA.@allowscalar sol[1, 1] = vars.U[]

  updatevars!(sol, vars, params, grid)
  return nothing
end

set_zeta!(prob, zeta) = set_zeta!(prob.sol, prob.vars, prob.params, prob.grid, zeta)

"""
    set_U!(prob, U)
    set_U!(sol, v, g, U)

Set the (kx, ky)=(0, 0) part of solution sol as the domain-average zonal flow U.
"""
function set_U!(sol, vars, params, grid, U::Float64)
  CUDA.@allowscalar sol[1, 1] = U
  updatevars!(sol, vars, params, grid)
  return nothing
end

set_U!(prob, U::Float64) = set_U!(prob.sol, prob.vars, prob.params, prob.grid, U)


"""
    energy(sol, grid)
    energy(prob)

Returns the domain-averaged kinetic energy of solution `sol`.
"""
energy(sol, grid::AbstractGrid) = 0.5 * ( parsevalsum2(grid.kr .* grid.invKrsq .* sol, grid) + parsevalsum2(grid.l .* grid.invKrsq .* sol, grid) ) / (grid.Lx * grid.Ly)
energy(prob) = energy(prob.sol, prob.grid)

"""
    enstrophy(sol, grid, vars)
    enstrophy(prob)

Returns the domain-averaged enstrophy of solution `sol`.
"""
function enstrophy(sol, grid::AbstractGrid, vars::AbstractVars)
  @. vars.uh = sol
  CUDA.@allowscalar vars.uh[1, 1] = 0
  return 0.5*parsevalsum2(vars.uh, grid) / (grid.Lx * grid.Ly)
end
enstrophy(prob) = enstrophy(prob.sol, prob.grid, prob.vars)

"""
    meanenergy(prob)

Returns the energy of the domain-averaged U.
"""
meanenergy(prob) = CUDA.@allowscalar real(0.5 * prob.sol[1, 1].^2)

"""
    meanenstrophy(prob)

Returns the enstrophy of the domain-averaged U.
"""
meanenstrophy(prob) = CUDA.@allowscalar real(prob.params.β * prob.sol[1, 1])

"""
    dissipation_ens(prob)

Returns the domain-averaged dissipation rate of enstrophy. nν must be >= 1.
"""
@inline function dissipation_ens(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  @. vars.uh = grid.Krsq^params.nν * abs2(sol)
  vars.uh[1, 1] = 0
  return params.ν / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

"""
    dissipation(prob)
    dissipation(sol, vars, params, grid)

Returns the domain-averaged dissipation rate. nν must be >= 1.
"""
@inline function dissipation(sol, vars, params, grid)
  @. vars.uh = grid.Krsq^(params.nν-1) * abs2(sol)
  CUDA.@allowscalar vars.uh[1, 1] = 0
  return params.ν / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline dissipation(prob) = dissipation(prob.sol, prob.vars, prob.params, prob.grid)

"""
    work(prob)
    work(sol, vars, grid)

Returns the domain-averaged rate of work of energy by the forcing Fqh.
"""
@inline function work(sol, vars::ForcedVars, grid)
  @. vars.uh = grid.invKrsq * sol * conj(vars.Fqh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline function work(sol, vars::StochasticForcedVars, grid)
  @. vars.uh = grid.invKrsq * (vars.prevsol + sol)/2 * conj(vars.Fqh) # Stratonovich
  # @. vars.uh = grid.invKrsq * vars.prevsol * conj(vars.Fqh)             # Ito
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline work(prob) = work(prob.sol, prob.vars, prob.grid)

"""
    work_ens(prob)
    work_ens(sol, v, grid)

Returns the domain-averaged rate of work of enstrophy by the forcing Fh.
"""
@inline function work_ens(sol, vars::ForcedVars, grid)
  @. vars.uh = sol * conj(vars.Fqh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline function work_ens(sol, vars::StochasticForcedVars, grid)
  @. vars.uh = (vars.prevsol + sol) / 2 * conj(vars.Fqh) # Stratonovich
  # @. vars.uh = grid.invKrsq * vars.prevsol * conj(vars.Fh)           # Ito
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline work_ens(prob) = work_ens(prob.sol, prob.vars, prob.grid)

"""
    drag(prob)
    drag(sol, vars, params, grid)

Returns the extraction of domain-averaged energy by drag μ.
"""
@inline function drag(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  @. vars.uh = grid.invKrsq * abs2(sol)
  CUDA.@allowscalar vars.uh[1, 1] = 0
  return params.μ / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

"""
    drag_ens(prob)

Returns the extraction of domain-averaged enstrophy by drag/hypodrag μ.
"""
@inline function drag_ens(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  @. vars.uh = grid.Krsq^0.0 * abs2(sol)
  vars.uh[1, 1] = 0
  return params.μ / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

end # module
