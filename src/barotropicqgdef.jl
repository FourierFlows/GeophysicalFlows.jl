module BarotropicQG_def

export
  Problem,
  set_zeta!,
  updatevars!,
  ke_energy,
  pe_energy,
  energy,
  enstrophy,
  enstrophy_lia,
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
using FourierFlows: parsevalsum, parsevalsum2

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
        kdef = 0.0,
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

  params = !(typeof(eta)<:ArrayType(dev)) ? Params(grid, β, kdef, eta, μ, ν, nν, calcFU, calcFq) : Params(β, kdef, eta, rfft(eta), μ, ν, nν, calcFU, calcFq)

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
     kdef :: T            # deformation wavenumber
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
function Params(grid::AbstractGrid{T, A}, β, kdef, eta ::Function, μ, ν, nν::Int, calcFU, calcFq) where {T, A}
  etagrid = A([eta(grid.x[i], grid.y[j]) for i=1:grid.nx, j=1:grid.ny])
     etah = rfft(etagrid)
  return Params(β,kdef, etagrid, etah, μ, ν, nν, calcFU, calcFq)
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

## add a new variable for Lapl psi - kdef^2 * psi ?
function streamfunction!(psih,zetah,grid,params)
  @. psih = - zetah / (grid.Krsq + params.kdef^2)
  CUDA.@allowscalar psih[1,1] = 0.0
  return nothing
end

function calcN_advection!(N, sol, t, clock, vars, params, grid)
  # Note that U = sol[1, 1]. For all other elements ζ = sol
  CUDA.@allowscalar vars.U[] = sol[1, 1].re
  @. vars.zetah = sol
  CUDA.@allowscalar vars.zetah[1, 1] = 0
  streamfunction!(vars.psih,vars.zetah, grid, params)
  @. vars.uh =  - im * grid.l  * vars.psih
  @. vars.vh = im * grid.kr  * vars.psih

  ldiv!(vars.zeta, grid.rfftplan, vars.zetah)
  ldiv!(vars.u, grid.rfftplan, vars.uh)
  vars.psih .= vars.vh # FFTW's irfft destroys its input; vars.vh is needed for N[1, 1]
  ldiv!(vars.v, grid.rfftplan, vars.psih)

  @. vars.q = vars.zeta + params.eta
  uq = vars.u                                            # use vars.u as scratch variable
  CUDA.@allowscalar @. uq = (vars.U[] + vars.u) * vars.q # (U+u)*q
  vq = vars.v                                            # use vars.v as scratch variable
  @. vq *= vars.q                                        # v*q

  uqh = vars.uh                                          # use vars.uh as scratch variable
  mul!(uqh, grid.rfftplan, uq)                           # \hat{(u+U)*q}

  # Nonlinear advection term for q (part 1)
  @. N = -im * grid.kr * uqh                             # -∂[(U+u)q]/∂x

  vqh = vars.uh                                          # use vars.uh as scratch variable
  mul!(vqh, grid.rfftplan, vq)                           # \hat{v*q}

  # Nonlinear advection term for q (part 2)
  @. N += - im * grid.l * vqh                            # -∂[vq]/∂y
  return nothing
end

function calcN!(N, sol, t, clock, vars, params, grid)
  calcN_advection!(N, sol, t, clock, vars, params, grid)
  addforcing!(N, sol, t, clock, vars, params, grid)
  if params.calcFU != nothingfunction
    # 'Nonlinear' term for U with topographic correlation.
    # Note: ⟨v*η⟩ =     sum(conj(v̂)*η̂) / (nx²ny²) if  fft is used,
    # while ⟨v*η⟩ = 2 * sum(conj(v̂)*η̂) / (nx²ny²) if rfft is used
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

  streamfunction!(vars.psih,vars.zetah, grid, params)
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
    energy(prob)
    energy(sol, grid)
Returns the domain-averaged kinetic energy of solution `sol`.
"""
energy(sol, grid::AbstractGrid) = 0.5 * ( parsevalsum2(grid.kr .* grid.invKrsq .* sol, grid) + parsevalsum2(grid.l .* grid.invKrsq .* sol, grid) ) / (grid.Lx * grid.Ly)
energy(prob) = energy(prob.sol, prob.grid)


function ke_energy(sol, grid, vars, params)
    streamfunction!(vars.psih,sol,grid,params)

  return parsevalsum(grid.Krsq .* abs2.(vars.psih), grid) / (grid.Lx * grid.Ly)
end
ke_energy(prob) = ke_energy(prob.sol, prob.grid, prob.vars, prob.params)

function pe_energy(sol, grid, vars, params)
    streamfunction!(vars.psih,sol,grid,params)

  return params.kdef*params.kdef*parsevalsum2(vars.psih, grid) / (grid.Lx * grid.Ly)
end
pe_energy(prob) = pe_energy(prob.sol, prob.grid, prob.vars, prob.params)

"""
    enstrophy(prob)
    enstrophy(sol, grid, vars)
Returns the domain-averaged enstrophy of solution `sol`.
"""
function enstrophy(sol, grid, vars)
  @. vars.uh = sol
  CUDA.@allowscalar vars.uh[1, 1] = 0
  return 0.5*parsevalsum2(vars.uh, grid) / (grid.Lx * grid.Ly)
end
enstrophy(prob) = enstrophy(prob.sol, prob.grid, prob.vars)


function enstrophy_lia(sol, grid, vars, params)
  streamfunction!(vars.psih,sol,grid,params)
  return parsevalsum(grid.Krsq.*grid.Krsq .* abs2.(vars.psih), grid) / (grid.Lx * grid.Ly)
end
enstrophy_lia(prob) = enstrophy_lia(prob.sol, prob.grid, prob.vars,prob.params)

"""
    meanenergy(prob)
Returns the energy of the domain-averaged U.
"""
meanenergy(prob) = CUDA.@allowscalar real(0.5 * prob.sol[1, 1]^2)

"""
    meanenstrophy(prob)
Returns the enstrophy of the domain-averaged U.
"""
meanenstrophy(prob) = CUDA.@allowscalar real(prob.params.β * prob.sol[1, 1])

"""
    energy_dissipation(prob)
    energy_dissipation(sol, vars, params, grid)
Returns the domain-averaged energy dissipation rate. nν must be >= 1.
"""
@inline function energy_dissipation(sol, vars, params, grid)
  energy_dissipationh = vars.uh # use vars.uh as scratch variable

  @. energy_dissipationh = params.ν * grid.Krsq^(params.nν-1) * abs2(sol)
  CUDA.@allowscalar energy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dissipationh, grid)
end

@inline energy_dissipation(prob) = energy_dissipation(prob.sol, prob.vars, prob.params, prob.grid)

"""
    enstrophy_dissipation(prob)
    enstrophy_dissipation(sol, vars, params, grid)
Returns the domain-averaged enstrophy dissipation rate. nν must be >= 1.
"""
@inline function enstrophy_dissipation(sol, vars, params, grid)
  enstrophy_dissipationh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_dissipationh = params.ν * grid.Krsq^params.nν * abs2(sol)
  CUDA.@allowscalar enstrophy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dissipationh, grid)
end

@inline enstrophy_dissipation(prob) = enstrophy_dissipation(prob.sol, prob.vars, prob.params, prob.grid)

"""
    energy_work(prob)
    energy_work(sol, vars, grid)
Returns the domain-averaged rate of work of energy by the forcing `Fqh`.
"""
@inline function energy_work(sol, vars::ForcedVars, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable

  @. energy_workh = grid.invKrsq * sol * conj(vars.Fqh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_workh, grid)
end

@inline function energy_work(sol, vars::StochasticForcedVars, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable

  @. energy_workh = grid.invKrsq * (vars.prevsol + sol)/2 * conj(vars.Fqh) # Stratonovich
  # @. energy_workh = grid.invKrsq * vars.prevsol * conj(vars.Fqh)             # Ito
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline energy_work(prob) = energy_work(prob.sol, prob.vars, prob.grid)

"""
    enstrophy_work(prob)
    enstrophy_work(sol, vars, grid)
Returns the domain-averaged rate of work of enstrophy by the forcing `Fqh`.
"""
@inline function enstrophy_work(sol, vars::ForcedVars, grid)
  enstrophy_workh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_workh = sol * conj(vars.Fqh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_workh, grid)
end

@inline function enstrophy_work(sol, vars::StochasticForcedVars, grid)
  enstrophy_workh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_workh = (vars.prevsol + sol) / 2 * conj(vars.Fqh) # Stratonovich
  # @. enstrophy_workh = grid.invKrsq * vars.prevsol * conj(vars.Fh)           # Ito
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_workh, grid)
end

@inline enstrophy_work(prob) = enstrophy_work(prob.sol, prob.vars, prob.grid)

"""
    energy_drag(prob)
Returns the extraction of domain-averaged energy by drag μ.
"""
@inline function energy_drag(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  energy_dragh = vars.uh # use vars.uh as scratch variable

  @. energy_dragh = params.μ * grid.invKrsq * abs2(sol)
  CUDA.@allowscalar energy_dragh[1, 1] = 0
  return  1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dragh, grid)
end

"""
    enstrophy_drag(prob)
Returns the extraction of domain-averaged enstrophy by drag/hypodrag μ.
"""
@inline function enstrophy_drag(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  enstrophy_dragh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_dragh = params.μ * abs2(sol)
  CUDA.@allowscalar enstrophy_dragh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dragh, grid)
end

end # module
