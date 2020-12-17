module SingleLayerQG

export
  Problem,
  set_q!,
  updatevars!,

  energy,
  kinetic_energy,
  potential_energy,
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

using FFTW: rfft
using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum, parsevalsum2

nothingfunction(args...) = nothing

"""
    Problem(; parameters...)

Construct a SingleLayerQG problem.
"""

function Problem(dev::Device=CPU();
  # Numerical parameters
                  nx = 256,
                  ny = nx,
                  Lx = 2π,
                  Ly = Lx,
                  dt = 0.01,
  # Physical parameters
                   β = 0.0,
  deformation_radius = Inf,
                 eta = nothing,
  # Drag and (hyper-)viscosity
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

  params = deformation_radius == Inf ? BarotropicQGParams(grid, T(β), eta, T(μ), T(ν), nν, calcF) : EquivalentBarotropicQGParams(grid, T(β), T(deformation_radius), eta, T(μ), T(ν), nν, calcF)

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
abstract type SingleLayerQGParams <: AbstractParams end

struct Params{T, Aphys, Atrans, ℓ} <: SingleLayerQGParams
                   β :: T            # Planetary vorticity y-gradient
  deformation_radius :: ℓ            # deformation radius
                 eta :: Aphys        # Topographic PV
                etah :: Atrans       # FFT of Topographic PV
                   μ :: T            # Linear drag
                   ν :: T            # Viscosity coefficient
                  nν :: Int          # Hyperviscous order (nν=1 is plain old viscosity)
              calcF! :: Function     # Function that calculates the forcing on QGPV q
end

const BarotropicQGParams = Params{<:AbstractFloat, <:AbstractArray, <:AbstractArray, Nothing}
const EquivalentBarotropicQGParams = Params{<:AbstractFloat, <:AbstractArray, <:AbstractArray, <:AbstractFloat}

get_topographicPV_grid_values(eta::Function, grid::AbstractGrid{T, A}) where {T, A} = A([eta(grid.x[i], grid.y[j]) for i=1:grid.nx, j=1:grid.ny])

"""
    EquivalentBarotropicQGParams(grid::TwoDGrid, β, deformation_radius, eta, μ, ν, nν::Int, calcF

Constructor for EquivalentBarotropicQGParams (finite Rossby radius of deformation).
"""
function EquivalentBarotropicQGParams(grid::AbstractGrid{T, A}, β, deformation_radius, eta, μ, ν, nν::Int, calcF) where {T, A}
  eta_grid = typeof(eta) <: AbstractArray ? eta : get_topographicPV_grid_values(eta, grid)
  eta_gridh = rfft(eta_grid)
  return Params(β, deformation_radius, eta_grid, eta_gridh, μ, ν, nν, calcF)
end

"""
    BarotropicQGParams(grid::TwoDGrid, β, eta, μ, ν, nν::Int, calcF

Constructor for BarotropicQGParams (infinite Rossby radius of deformation).
"""
BarotropicQGParams(grid::AbstractGrid{T, A}, β, eta, μ, ν, nν::Int, calcF) where {T, A} =
    EquivalentBarotropicQGParams(grid, β, nothing, eta, μ, ν, nν, calcF)
    

# ---------
# Equations
# ---------

"""
    Equation(params, grid)

Returns the equation for two-dimensional barotropic QG problem with `params` and `grid`.
"""
function Equation(params::BarotropicQGParams, grid::AbstractGrid)
  L = @. - params.μ - params.ν * grid.Krsq^params.nν + im * params.β * grid.kr * grid.invKrsq
  CUDA.@allowscalar L[1, 1] = 0
  return FourierFlows.Equation(L, calcN!, grid)
end

function Equation(params::EquivalentBarotropicQGParams, grid::AbstractGrid)
  L = @. - params.μ - params.ν * grid.Krsq^params.nν + im * params.β * grid.kr / (grid.Krsq + 1 / params.deformation_radius^2)
  CUDA.@allowscalar L[1, 1] = 0
  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

abstract type SingleLayerQGVars <: AbstractVars end

struct Vars{Aphys, Atrans, F, P} <: SingleLayerQGVars
        q :: Aphys
        ψ :: Aphys
        u :: Aphys
        v :: Aphys
       qh :: Atrans
       ψh :: Atrans
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
  @devzeros Dev T (grid.nx, grid.ny) q u v ψ
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh uh vh ψh
  Vars(q, ψ, u, v, qh, ψh, uh, vh, nothing, nothing)
end

"""
    ForcedVars(dev, grid)

Returns the vars for forced two-dimensional barotropic QG problem on device dev and with `grid`.
"""
function ForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) q u v ψ
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh uh vh ψh Fh
  return Vars(q, ψ, u, v, qh, ψh, uh, vh, Fh, nothing)
end

"""
    StochasticForcedVars(dev, grid)

Returns the vars for stochastically forced two-dimensional barotropic QG problem on device dev and with `grid`.
"""
function StochasticForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) q u v ψ
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh uh vh ψh Fh prevsol
  return Vars(q, ψ, u, v, qh, ψh, uh, vh, Fh, prevsol)
end


# -------
# Solvers
# -------

function calcN_advection!(N, sol, t, clock, vars, params, grid)
  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  ldiv!(vars.q, grid.rfftplan, vars.qh)
  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)

  uq_plus_η = vars.u                                            # use vars.u as scratch variable
  @. uq_plus_η *= vars.q + params.eta                           # u*(q+η)
  vq_plus_η = vars.v                                            # use vars.v as scratch variable
  @. vq_plus_η *= vars.q + params.eta                           # v*(q+η)

  uq_plus_ηh = vars.uh                                          # use vars.uh as scratch variable
  mul!(uq_plus_ηh, grid.rfftplan, uq_plus_η)                    # \hat{u*(q+η)}
  vq_plus_ηh = vars.vh                                          # use vars.vh as scratch variable
  mul!(vq_plus_ηh, grid.rfftplan, vq_plus_η)                    # \hat{v*(q+η)}

  @. N = -im * grid.kr * uq_plus_ηh - im * grid.l * vq_plus_ηh  # -∂[u*(q+η)]/∂x -∂[v*(q+η)]/∂y
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

"""
    streamfunctionfrompv!(ψh, qh, params, grid)

Invert the Fourier transform of PV to obtain the Fourier transform of the streamfunction `ψh`.
"""
function streamfunctionfrompv!(ψh, qh, params::BarotropicQGParams, grid)
  @. ψh =  - grid.invKrsq * qh
end

function streamfunctionfrompv!(ψh, qh, params::EquivalentBarotropicQGParams, grid)
  @. ψh  = - qh / (grid.Krsq + 1 / params.deformation_radius^2)
end


# ----------------
# Helper functions
# ----------------

"""
    updatevars!(sol, vars, params, grid)

Update the variables in `vars` with the solution in `sol`.
"""
function updatevars!(sol, vars, params, grid)
  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  ldiv!(vars.q, grid.rfftplan, deepcopy(vars.qh))
  ldiv!(vars.ψ, grid.rfftplan, deepcopy(vars.ψh))
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))
  ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh))

  return nothing
end

updatevars!(prob) = updatevars!(prob.sol, prob.vars, prob.params, prob.grid)

"""
    set_q!(prob, q)

Set the solution of problem, `prob.sol` as the transform of ``q`` and update variables `prob.vars`.
"""
function set_q!(prob, q)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  
  mul!(vars.qh, grid.rfftplan, q)
  @. sol = vars.qh

  updatevars!(sol, vars, params, grid)

  return nothing
end

"""
    kinetic_energy(prob)
    kinetic_energy(sol, grid, vars, params)

Returns the domain-averaged kinetic energy of the fluid, 
```math
\\textrm{KE} = \\int \\frac1{2} |\\boldsymbol{\\nabla} \\psi|^2 \\frac{\\mathrm{d}^2 \\boldsymbol{x}}{L_x L_y} = \\sum_{\\boldsymbol{k}} \\frac1{2} |\\boldsymbol{k}|^2 |\\hat{\\psi}|^2 \\ .
```
"""
function kinetic_energy(sol, vars, params, grid)
  streamfunctionfrompv!(vars.ψh, sol, params, grid)
  @. vars.uh = sqrt.(grid.Krsq) * vars.ψh      # vars.uh is a dummy variable

  return parsevalsum2(vars.uh , grid) / (2 * grid.Lx * grid.Ly)
end

kinetic_energy(prob) = kinetic_energy(prob.sol, prob.vars, prob.params, prob.grid)

"""
    potential_energy(prob)
    potential_energy(sol, grid, vars, params)

Returns the domain-averaged potential energy of the fluid, 
```math
\\textrm{PE} = \\int \\frac1{2} \\frac{\\psi^2}{\\ell^2} \\frac{\\mathrm{d}^2 \\boldsymbol{x}}{L_x L_y} = \\sum_{\\boldsymbol{k}} \\frac1{2} \\frac{|\\hat{\\psi}|^2}{\\ell^2} \\ .
```
"""
function potential_energy(sol, vars, params::EquivalentBarotropicQGParams, grid)
  streamfunctionfrompv!(vars.ψh, sol, params, grid)
  return 1 / params.deformation_radius^2 * parsevalsum2(vars.ψh, grid) / (2 * grid.Lx * grid.Ly)
end

@inline potential_energy(sol, vars, params::BarotropicQGParams, grid) = 0

@inline potential_energy(prob) = potential_energy(prob.sol, prob.vars, prob.params, prob.grid)

"""
    energy(prob)
    energy(sol, grid, vars, params)

Returns the domain-averaged total energy of the fluid, that is, the kinetic energy for a
pure barotropic flow or the sum of kinetic and potential energies for an equivalent barotropic
flow.
"""
@inline energy(prob) = energy(prob.sol, prob.vars, prob.params, prob.grid)

@inline energy(sol, vars, params::BarotropicQGParams, grid) = kinetic_energy(sol, vars, params, grid)

@inline energy(sol, vars, params::EquivalentBarotropicQGParams, grid) = kinetic_energy(sol, vars, params, grid) + potential_energy(sol, vars, params, grid)

"""
    enstrophy(prob)
    enstrophy(sol, vars, params, grid)

Returns the domain-averaged enstrophy
```math
\\int \\frac1{2} (q + \\eta)^2 \\frac{\\mathrm{d}^2 \\boldsymbol{x}}{L_x L_y} = \\sum_{\\boldsymbol{k}} \\frac1{2} |\\hat{q} + \\hat{\\eta}|^2 \\ .
```
"""
function enstrophy(sol, vars, params, grid)
  @. vars.qh = sol
  return parsevalsum2(vars.qh + params.etah, grid) / (2 * grid.Lx * grid.Ly)
end

@inline enstrophy(prob) = enstrophy(prob.sol, prob.vars, prob.params, prob.grid)

"""
    energy_dissipation(prob)
    energy_dissipation(sol, vars, params, grid)

Returns the domain-averaged energy dissipation rate. nν must be >= 1.
"""
@inline function energy_dissipation(sol, vars, params::BarotropicQGParams, grid)
  energy_dissipationh = vars.uh # use vars.uh as scratch variable

  @. energy_dissipationh = params.ν * grid.Krsq^(params.nν-1) * abs2(sol)
  CUDA.@allowscalar energy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dissipationh, grid)
end

energy_dissipation(sol, vars, params::EquivalentBarotropicQGParams, grid) = error("not implemented for finite deformation radius")

@inline energy_dissipation(prob) = energy_dissipation(prob.sol, prob.vars, prob.params, prob.grid)

"""
    enstrophy_dissipation(prob)
    enstrophy_dissipation(sol, vars, params, grid)

Returns the domain-averaged enstrophy dissipation rate. nν must be >= 1.
"""
@inline function enstrophy_dissipation(sol, vars, params::BarotropicQGParams, grid)
  enstrophy_dissipationh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_dissipationh = params.ν * grid.Krsq^params.nν * abs2(sol)
  CUDA.@allowscalar enstrophy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dissipationh, grid)
end

@inline enstrophy_dissipation(sol, vars, params::EquivalentBarotropicQGParams, grid) = error("not implemented for finite deformation radius")

@inline enstrophy_dissipation(prob) = enstrophy_dissipation(prob.sol, prob.vars, prob.params, prob.grid)

"""
    energy_work(prob)
    energy_work(sol, vars, params, grid)

Returns the domain-averaged rate of work of energy by the forcing `Fh`.
"""
@inline function energy_work(sol, vars::ForcedVars, params::BarotropicQGParams, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable

  @. energy_workh = grid.invKrsq * sol * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_workh, grid)
end

@inline function energy_work(sol, vars::StochasticForcedVars, params::BarotropicQGParams, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable

  @. energy_workh = grid.invKrsq * (vars.prevsol + sol)/2 * conj(vars.Fh) # Stratonovich
  # @. energy_workh = grid.invKrsq * vars.prevsol * conj(vars.Fh)             # Ito
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline energy_work(sol, vars, params::EquivalentBarotropicQGParams, grid) = error("not implemented for finite deformation radius")

@inline energy_work(prob) = energy_work(prob.sol, prob.vars, prob.params, prob.grid)

"""
    enstrophy_work(prob)
    enstrophy_work(sol, vars, params, grid)

Returns the domain-averaged rate of work of enstrophy by the forcing `Fh`.
"""
@inline function enstrophy_work(sol, vars::ForcedVars, params::BarotropicQGParams, grid)
  enstrophy_workh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_workh = sol * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_workh, grid)
end

@inline function enstrophy_work(sol, vars::StochasticForcedVars, params::BarotropicQGParams, grid)
  enstrophy_workh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_workh = (vars.prevsol + sol) / 2 * conj(vars.Fh) # Stratonovich
  # @. enstrophy_workh = grid.invKrsq * vars.prevsol * conj(vars.Fh)           # Ito
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_workh, grid)
end

@inline enstrophy_work(sol, vars, params::EquivalentBarotropicQGParams, grid) = error("not implemented for finite deformation radius")

@inline enstrophy_work(prob) = enstrophy_work(prob.sol, prob.vars, prob.params, prob.grid)

"""
    energy_drag(prob)
    energy_drag(sol, vars, params, grid)

Returns the extraction of domain-averaged energy by drag μ.
"""
@inline function energy_drag(sol, vars, params::BarotropicQGParams, grid)
  energy_dragh = vars.uh # use vars.uh as scratch variable

  @. energy_dragh = params.μ * grid.invKrsq * abs2(sol)
  CUDA.@allowscalar energy_dragh[1, 1] = 0
  return  1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dragh, grid)
end

@inline energy_drag(sol, vars, params::EquivalentBarotropicQGParams, grid) = error("not implemented for finite deformation radius")

@inline energy_drag(prob) = energy_drag(prob.sol, prob.vars, prob.params, prob.grid)

"""
    enstrophy_drag(prob)
    enstrophy_drag(sol, vars, params, grid)

Returns the extraction of domain-averaged enstrophy by drag/hypodrag μ.
"""
@inline function enstrophy_drag(sol, vars, params::BarotropicQGParams, grid)
  enstrophy_dragh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_dragh = params.μ * abs2(sol)
  CUDA.@allowscalar enstrophy_dragh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dragh, grid)
end

@inline enstrophy_drag(sol, vars, params::EquivalentBarotropicQGParams, grid) = error("not implemented for finite deformation radius")

@inline enstrophy_drag(prob) = enstrophy_drag(prob.sol, prob.vars, prob.params, prob.grid)

end # module
