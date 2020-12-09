module SingleLayerQG

export
  Problem,
  set_zeta!,
  updatevars!,

<<<<<<< HEAD:src/singlelayerqg.jl
  k_energy,
  p_energy,
=======
  kinetic_energy,
  potential_energy,
>>>>>>> def_radius:src/barotropicqg.jl
  energy_dissipation,
  energy_work,
  energy_drag,
  enstrophy,
  reduced_enstrophy,
  enstrophy_dissipation,
  enstrophy_work,
  enstrophy_drag

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
       calcF = nothingfunction,
  stochastic = false,
           T = Float64)

  # the grid
  grid = TwoDGrid(dev, nx, Lx, ny, Ly; T=T)
  x, y = gridpoints(grid)

  # topographic PV
  eta === nothing && ( eta = zeros(dev, T, (nx, ny)) )

  params = !(typeof(eta)<:ArrayType(dev)) ? Params(grid, β, kdef, eta, μ, ν, nν, calcF) : Params(β, kdef, eta, rfft(eta), μ, ν, nν, calcF)

  vars = calcF == nothingfunction ? Vars(dev, grid) : (stochastic ? StochasticForcedVars(dev, grid) : ForcedVars(dev, grid))

  equation = Equation(params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
end


# ----------
# Parameters
# ----------

"""
    Params(g::TwoDGrid, β, eta, μ, ν, nν, calcF)

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
   calcF! :: Function     # Function that calculates the forcing on QGPV q
end

"""
    Params(g::TwoDGrid, β, eta::Function, μ, ν, nν, calcF)

Constructor for Params that accepts a generating function for the topographic PV.
"""
function Params(grid::AbstractGrid{T, A}, β, kdef, eta::Function, μ, ν, nν::Int, calcF) where {T, A}
  etagrid = A([eta(grid.x[i], grid.y[j]) for i=1:grid.nx, j=1:grid.ny])
  etah = rfft(etagrid)
  return Params(β, kdef, etagrid, etah, μ, ν, nν, calcF)
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

struct Vars{Aphys, Atrans, F, P} <: BarotropicQGVars
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
       Fh :: F
  prevsol :: P
end

const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    Vars(dev, grid)

Returns the vars for unforced two-dimensional barotropic QG problem on device `dev` and with `grid`
"""
function Vars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) q u v psi zeta
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh uh vh psih zetah
  Vars(q, zeta, psi, u, v, qh, zetah, psih, uh, vh, nothing, nothing)
end

"""
    ForcedVars(dev, grid)

Returns the vars for forced two-dimensional barotropic QG problem on device dev and with `grid`.
"""
function ForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) q u v psi zeta
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh uh vh psih zetah Fh
  return Vars(q, zeta, psi, u, v, qh, zetah, psih, uh, vh, Fh, nothing)
end

"""
    StochasticForcedVars(dev, grid)

Returns the vars for stochastically forced two-dimensional barotropic QG problem on device dev and with `grid`.
"""
function StochasticForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) q u v psi zeta
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh uh vh psih zetah Fh prevsol
  return Vars(q, zeta, psi, u, v, qh, zetah, psih, uh, vh, Fh, prevsol)
end


# -------
# Solvers
# -------

function calcN_advection!(N, sol, t, clock, vars, params, grid)
  @. vars.zetah = sol
<<<<<<< HEAD:src/singlelayerqg.jl
  @. vars.psih  = - vars.zetah / (grid.Krsq .+ params.kdef^2)
  CUDA.@allowscalar vars.psih[1, 1] = 0
=======
  @. vars.psih  = - vars.zetah / (grid.Krsq + params.kdef^2)
  if params.kdef == 0.0
      CUDA.@allowscalar vars.psih[1, 1] = 0
  end
>>>>>>> def_radius:src/barotropicqg.jl
  @. vars.uh    = -im * grid.l  * vars.psih
  @. vars.vh    =  im * grid.kr * vars.psih

  ldiv!(vars.zeta, grid.rfftplan, vars.zetah)
  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)

  @. vars.q = vars.zeta + params.eta
  uq = vars.u                                            # use vars.u as scratch variable
  @. uq *= vars.q                                        # u*q
  vq = vars.v                                            # use vars.v as scratch variable
  @. vq *= vars.q                                        # v*q

  uqh = vars.uh                                          # use vars.uh as scratch variable
  mul!(uqh, grid.rfftplan, uq)                           # \hat{u*q}
  vqh = vars.vh                                          # use vars.vh as scratch variable
  mul!(vqh, grid.rfftplan, vq)                           # \hat{v*q}

  @. N = -im * grid.kr * uqh - im * grid.l * vqh         # -∂(u*q)/∂x -∂(v*q)/∂y
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
    updatevars!(sol, vars, params, grid)

Update the variables in `vars` with the solution in `sol`.
"""
function updatevars!(sol, vars, params, grid)
  @. vars.zetah = sol
<<<<<<< HEAD:src/singlelayerqg.jl
  @. vars.psih  = - vars.zetah / (grid.Krsq .+ params.kdef^2)
  CUDA.@allowscalar vars.psih[1, 1] = 0
=======
  @. vars.psih  = - vars.zetah / (grid.Krsq + params.kdef^2)
  if params.kdef == 0.0
      CUDA.@allowscalar vars.psih[1, 1] = 0
  end
>>>>>>> def_radius:src/barotropicqg.jl
  @. vars.uh    = -im * grid.l  * vars.psih
  @. vars.vh    =  im * grid.kr * vars.psih

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
function set_zeta!(sol, vars, params, grid, zeta)
  mul!(vars.zetah, grid.rfftplan, zeta)

  @. sol = vars.zetah

  updatevars!(sol, vars, params, grid)

  return nothing
end

set_zeta!(prob, zeta) = set_zeta!(prob.sol, prob.vars, prob.params, prob.grid, zeta)


"""
<<<<<<< HEAD:src/singlelayerqg.jl
    k_energy(prob)
    p_energy(prob)
    k_energy(vars, grid)
    p_energy(vars, grid, params)
=======
    kinetic_energy(prob)
    potential_energy(prob)
    kinetic_energy(vars, grid)
    potential_energy(vars, grid, params)
>>>>>>> def_radius:src/barotropicqg.jl

Returns the domain-averaged kinetic energy of solution `sol`: ∫ ½ (u²+v²) dxdy / (Lx Ly) = ∑ ½ k² |ψ̂|² / (Lx Ly).
Returns the domain-averaged potential energy of solution `sol`: ½ kdef² ∫ ψ² dxdy / (Lx Ly) = ½ kdef² ∑ |ψ̂|² / (Lx Ly).

"""
<<<<<<< HEAD:src/singlelayerqg.jl
function k_energy(sol, grid, vars, params)
    @. vars.uh = sqrt.(grid.Krsq) .* sol ./(grid.Krsq .+ params.kdef^2) ## uh is a dummy variable
    CUDA.@allowscalar vars.uh[1, 1] = 0
    return parsevalsum2(vars.uh , grid) / (2 * grid.Lx * grid.Ly)
end

function p_energy(sol, grid, vars, params)
    @. vars.uh = sol ./(grid.Krsq .+ params.kdef^2) ## uh is a dummy variable
    CUDA.@allowscalar vars.uh[1, 1] = 0
    return params.kdef^2*parsevalsum2(vars.uh, grid) / (2 * grid.Lx * grid.Ly)
end

k_energy(prob) = k_energy(prob.sol, prob.grid, prob.vars, prob.params)
p_energy(prob) = p_energy(prob.sol, prob.grid, prob.vars, prob.params)
=======
function kinetic_energy(sol, grid, vars, params)
    @. vars.uh = sqrt.(grid.Krsq) * sol /(grid.Krsq + params.kdef^2) ## uh is a dummy variable
    if params.kdef == 0.0
        CUDA.@allowscalar vars.uh[1, 1] = 0
    end
    return parsevalsum2(vars.uh , grid) / (2 * grid.Lx * grid.Ly)
end

function potential_energy(sol, grid, vars, params)
    @. vars.uh = sol /(grid.Krsq + params.kdef^2) ## uh is a dummy variable
    if params.kdef == 0.0
        CUDA.@allowscalar vars.uh[1, 1] = 0
    end
    return params.kdef^2*parsevalsum2(vars.uh, grid) / (2 * grid.Lx * grid.Ly)
end

kinetic_energy(prob) = kinetic_energy(prob.sol, prob.grid, prob.vars, prob.params)
potential_energy(prob) = potential_energy(prob.sol, prob.grid, prob.vars, prob.params)
>>>>>>> def_radius:src/barotropicqg.jl

"""
    enstrophy(prob)
    enstrophy(sol, grid, vars)
    reduced_enstrophy(prob)
    reduced_enstrophy(sol, grid, vars)

Returns the domain-averaged enstrophy ½ ∫ q² dxdy / (Lx Ly), with q = ζ + η and sol = ζ.
Returns the domain-averaged reduced enstrophy ½ ∫(ζ² + 2ζη) dxdy / (Lx Ly) .

"""
function enstrophy(sol, grid, vars, params)
  @. vars.zetah = sol
  return 0.5*parsevalsum2(vars.zetah + params.etah, grid) / (grid.Lx * grid.Ly)
end
enstrophy(prob) = enstrophy(prob.sol, prob.grid, prob.vars, prob.params)

function reduced_enstrophy(sol, grid, vars, params)
  @. vars.zetah = sol
  return 0.5*parsevalsum(abs2.(vars.zetah) .+ 2* vars.zetah .* params.etah, grid) / (grid.Lx * grid.Ly)
end
reduced_enstrophy(prob) = reduced_enstrophy(prob.sol, prob.grid, prob.vars, prob.params)

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

Returns the domain-averaged rate of work of energy by the forcing `Fh`.
"""
@inline function energy_work(sol, vars::ForcedVars, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable

  @. energy_workh = grid.invKrsq * sol * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_workh, grid)
end

@inline function energy_work(sol, vars::StochasticForcedVars, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable

  @. energy_workh = grid.invKrsq * (vars.prevsol + sol)/2 * conj(vars.Fh) # Stratonovich
  # @. energy_workh = grid.invKrsq * vars.prevsol * conj(vars.Fh)             # Ito
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
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
