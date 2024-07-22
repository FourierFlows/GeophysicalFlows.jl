module SingleLayerQG

export
  Problem,
  streamfunctionfrompv!,
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
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum, parsevalsum2

nothingfunction(args...) = nothing

"""
    Problem(dev::Device = CPU();
                     nx = 256,
                     ny = nx,
                     Lx = 2π,
                     Ly = Lx,
                      β = 0.0,
     deformation_radius = Inf,
                      U = 0.0,
                    eta = nothing,
                      ν = 0.0,
                     nν = 1,
                      μ = 0.0,
                     dt = 0.01,
                stepper = "RK4",
                  calcF = nothingfunction,
             stochastic = false,
       aliased_fraction = 1/3,
                      T = Float64)

Construct a single-layer quasi-geostrophic problem on device `dev`.

Arguments
=========
  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.

Keyword arguments
=================
  - `nx`: Number of grid points in ``x``-domain.
  - `ny`: Number of grid points in ``y``-domain.
  - `Lx`: Extent of the ``x``-domain.
  - `Ly`: Extent of the ``y``-domain.
  - `β`: Planetary vorticity ``y``-gradient.
  - `deformation_radius`: Rossby radius of deformation; set `Inf` for purely barotropic.
  - `U`: Background flow in the ``x``-direction.
  - `eta`: Topographic potential vorticity.
  - `ν`: Small-scale (hyper)-viscosity coefficient.
  - `nν`: (Hyper)-viscosity order, `nν```≥ 1``.
  - `μ`: Linear drag coefficient.
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
  # Physical parameters
                   β = 0.0,
  deformation_radius = Inf,
                   U = 0.0,
                 eta = nothing,
  # Drag and (hyper-)viscosity
                   ν = 0.0,
                  nν = 1,
                   μ = 0.0,
  # Timestepper and equation options
                  dt = 0.01,
             stepper = "RK4",
               calcF = nothingfunction,
          stochastic = false,
  # Float type and dealiasing
    aliased_fraction = 1/3,
                   T = Float64)

  # the grid
  grid = TwoDGrid(dev; nx, Lx, ny, Ly, aliased_fraction, T)
  x, y = gridpoints(grid)

  # topographic PV
  eta === nothing && (eta = zeros(dev, T, (nx, ny)))

  if U isa Number
	U = T(U)
  else
	U = device_array(dev)(reshape(U,(1, grid.ny)))
  end

  params = deformation_radius == Inf ? BarotropicQGParams(grid, T(β), U, eta, T(μ), T(ν), nν, calcF) : EquivalentBarotropicQGParams(grid, T(β), T(deformation_radius), U, eta, T(μ), T(ν), nν, calcF)

  vars = calcF == nothingfunction ? DecayingVars(grid) : (stochastic ? StochasticForcedVars(grid) : ForcedVars(grid))

  equation = Equation(params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params)
end


# ----------
# Parameters
# ----------

abstract type SingleLayerQGParams <: AbstractParams end

"""
    struct Params{T, Aphys, Atrans, ℓ} <: SingleLayerQGParams

The parameters for the `SingleLayerQG` problem.

$(TYPEDFIELDS)
"""
struct Params{T, Aphys, Atrans, ℓ} <: SingleLayerQGParams
    "planetary vorticity ``y``-gradient"
                   β :: T
    "Rossby radius of deformation"
  deformation_radius :: ℓ
    "Background flow in x direction"
                   U :: Union{T, Aphys}
    "topographic potential vorticity"
                 eta :: Aphys
    "Fourier transform of topographic potential vorticity"
                etah :: Atrans
    "linear drag coefficient"
                   μ :: T
    "small-scale (hyper)-viscosity coefficient"
                   ν :: T
    "(hyper)-viscosity order, `nν```≥ 1``"
                  nν :: Int
    "function that calculates the Fourier transform of the forcing, ``F̂``"
              calcF! :: Function
end

const BarotropicQGParams = Params{<:AbstractFloat, <:AbstractArray, <:AbstractArray, Nothing}
const EquivalentBarotropicQGParams = Params{<:AbstractFloat, <:AbstractArray, <:AbstractArray, <:AbstractFloat}

"""
    EquivalentBarotropicQGParams(grid, β, deformation_radius, U, eta, μ, ν, nν, calcF)

Return the parameters for an Equivalent Barotropic QG problem (i.e., with finite Rossby radius of deformation).
"""
function EquivalentBarotropicQGParams(grid::AbstractGrid{T, A}, β, deformation_radius, U, eta, μ, ν, nν::Int, calcF) where {T, A}
  eta_on_grid = typeof(eta) <: AbstractArray ? A(eta) : FourierFlows.on_grid(eta, grid)
  etah_on_grid = rfft(eta_on_grid)

  return Params(β, deformation_radius, U, eta_on_grid, etah_on_grid, μ, ν, nν, calcF)
end

"""
    BarotropicQGParams(grid, β, U, eta, μ, ν, nν, calcF)

Return the parameters for a Barotropic QG problem (i.e., with infinite Rossby radius of deformation).
"""
BarotropicQGParams(grid::AbstractGrid, β, U, eta, μ, ν, nν::Int, calcF) =
    EquivalentBarotropicQGParams(grid, β, nothing, U, eta, μ, ν, nν, calcF)


# ---------
# Equations
# ---------

"""
    Equation(params::BarotropicQGParams, grid)

Return the equation for a barotropic QG problem with `params` and `grid`. Linear operator 
``L`` includes bottom drag ``μ``, (hyper)-viscosity of order ``n_ν`` with coefficient ``ν``, 
the ``β`` term, and a constant background flow ``U``:

```math
L = - μ - ν |𝐤|^{2 n_ν} + i β k_x / |𝐤|² - i U k_x .
```

The nonlinear term is computed via `calcN!` function.
"""
function Equation(params::BarotropicQGParams, grid::AbstractGrid)
  
  if params.U isa Number
	U₀ = params.U
  else
	U₀ = 0.0
  end

  L = @. - params.μ - params.ν * grid.Krsq^params.nν + im * params.β * grid.kr * grid.invKrsq - im * U₀ * grid.kr
  CUDA.@allowscalar L[1, 1] = 0
  
  return FourierFlows.Equation(L, calcN!, grid)
end

"""
    Equation(params::EquivalentBarotropicQGParams, grid)

Return the equation for an equivalent-barotropic QG problem with `params` and `grid`. 
Linear operator ``L`` includes bottom drag ``μ``, (hyper)-viscosity of order ``n_ν`` with 
coefficient ``ν``, the ``β`` term and a constant background flow ``U``:

```math
L = -μ - ν |𝐤|^{2 n_ν} + i β k_x / (|𝐤|² + 1/ℓ²) - i U k_x .
```

The nonlinear term is computed via `calcN!` function.
"""
function Equation(params::EquivalentBarotropicQGParams, grid::AbstractGrid)

  if params.U isa Number
	U₀ = params.U
  else
	U₀ = 0.0
  end

  L = @. - params.μ - params.ν * grid.Krsq^params.nν + im * params.β * grid.kr / (grid.Krsq + 1 / params.deformation_radius^2) - im * U₀ * grid.kr
  CUDA.@allowscalar L[1, 1] = 0
  
  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

abstract type SingleLayerQGVars <: AbstractVars end

"""
    struct Vars{Aphys, Atrans, F, P} <: SingleLayerQGVars

The variables for SingleLayer QG:

$(FIELDS)
"""
struct Vars{Aphys, Atrans, F, P} <: SingleLayerQGVars
    "relative vorticity (+ vortex stretching)"
        q :: Aphys
    "streamfunction"
        ψ :: Aphys
    "``x``-component of velocity"
        u :: Aphys
    "``y``-component of velocity"
        v :: Aphys
    "Fourier transform of relative vorticity (+ vortex stretching)"
       qh :: Atrans
    "Fourier transform of streamfunction"
       ψh :: Atrans
    "Fourier transform of ``x``-component of velocity"
       uh :: Atrans
    "Fourier transform of ``y``-component of velocity"
       vh :: Atrans
    "Fourier transform of forcing"
       Fh :: F
    "`sol` at previous time-step"
  prevsol :: P
end

const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    DecayingVars(grid)

Return the variables for unforced single-layer QG problem on `grid`.
"""
function DecayingVars(grid::AbstractGrid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) q u v ψ
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh uh vh ψh

  Vars(q, ψ, u, v, qh, ψh, uh, vh, nothing, nothing)
end

"""
    ForcedVars(grid)

Return the variables for forced single-layer QG problem on `grid`.
"""
function ForcedVars(grid::AbstractGrid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) q u v ψ
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh uh vh ψh Fh

  return Vars(q, ψ, u, v, qh, ψh, uh, vh, Fh, nothing)
end

"""
    StochasticForcedVars(grid)

Return the variables for stochastically forced barotropic QG problem on `grid`.
"""
function StochasticForcedVars(grid::AbstractGrid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) q u v ψ
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) qh uh vh ψh Fh prevsol

  return Vars(q, ψ, u, v, qh, ψh, uh, vh, Fh, prevsol)
end


# -------
# Solvers
# -------

"""
    calcN_advection!(N, sol, t, clock, vars, params, grid)

Calculate the Fourier transform of the advection term, ``- 𝖩(ψ+X, q+η-∂U/∂y)`` in conservative 
form, i.e., ``- ∂[(u+U)*(q+η-∂U/∂y)]/∂x - ∂[v*(q+η-∂U/∂y)]/∂y`` and store it in `N`:

```math
N = - \\widehat{𝖩(ψ + X, q + η - ∂U/∂y)} = - i k_x \\widehat{(u+U) (q + η - ∂U/∂y)} - i k_y \\widehat{v (q + η - ∂U/∂y)} .
```
"""
function calcN_advection!(N, sol, t, clock, vars, params, grid)

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  ldiv!(vars.q, grid.rfftplan, vars.qh)
  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)

  if params.U isa Number

  	uq_plus_η = vars.u                                            # use vars.u as scratch variable
  	@. uq_plus_η *= vars.q + params.eta                           # u * (q + η)
  	vq_plus_η = vars.v                                            # use vars.v as scratch variable
  	@. vq_plus_η *= vars.q + params.eta                           # v * (q + η)

  else

	Uy = real.(ifft(im * grid.l .* fft(params.U)))                # PV background (η - ∂U/∂y)

  	uq_plus_η = vars.u .+ params.U                                # use vars.u as scratch variable
  	@. uq_plus_η *= vars.q + params.eta .- Uy                     # (u + U) * (q + η - ∂U/∂y)
  	vq_plus_η = vars.v                                            # use vars.v as scratch variable
  	@. vq_plus_η *= vars.q + params.eta .- Uy                     # v * (q + η - ∂U/∂y)

  end

  uq_plus_ηh = vars.uh                                                # use vars.uh as scratch variable
  mul!(uq_plus_ηh, grid.rfftplan, uq_plus_η)                          # \hat{(u + U) * (q + η - ∂U/∂y)}
  vq_plus_ηh = vars.vh                                                # use vars.vh as scratch variable
  mul!(vq_plus_ηh, grid.rfftplan, vq_plus_η)                          # \hat{v * (q + η - ∂U/∂y)}

  @. N = -im * grid.kr * uq_plus_ηh - im * grid.l * vq_plus_ηh        # - ∂[(u+U)*(q+η-∂U/∂y)]/∂x - ∂[v*(q+η-∂U/∂y)]/∂y

  return nothing
end

"""
    calcN!(N, sol, t, clock, vars, params, grid)

Calculate the nonlinear term, that is the advection term and the forcing,

```math
N = - \\widehat{𝖩(ψ, q + η)} + F̂ .
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

"""
    streamfunctionfrompv!(ψh, qh, params, grid)

Invert the Fourier transform of PV `qh` to obtain the Fourier transform of the streamfunction `ψh`.
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
  dealias!(sol, grid)
  
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

Return the problem's (`prob`) domain-averaged kinetic energy of the fluid. Since
``u² + v² = |{\\bf ∇} ψ|²``, the domain-averaged kinetic energy is 

```math
\\int \\frac1{2} |{\\bf ∇} ψ|² \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} \\frac1{2} |𝐤|² |ψ̂|² .
```
"""
@inline kinetic_energy(prob) = kinetic_energy(prob.sol, prob.vars, prob.params, prob.grid)

function kinetic_energy(sol, vars, params, grid)
  streamfunctionfrompv!(vars.ψh, sol, params, grid)
  @. vars.uh = sqrt.(grid.Krsq) * vars.ψh      # vars.uh is a dummy variable

  return parsevalsum2(vars.uh , grid) / (2 * grid.Lx * grid.Ly)
end

"""
    potential_energy(prob)

Return the problem's (`prob`) domain-averaged potential energy of the fluid,

```math
\\int \\frac1{2} \\frac{ψ²}{ℓ²} \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} \\frac1{2} \\frac{|ψ̂|²}{ℓ²} .
```
"""
@inline potential_energy(prob) = potential_energy(prob.sol, prob.vars, prob.params, prob.grid)

@inline potential_energy(sol, vars, params::BarotropicQGParams, grid) = 0

function potential_energy(sol, vars, params::EquivalentBarotropicQGParams, grid)
  streamfunctionfrompv!(vars.ψh, sol, params, grid)

  return 1 / params.deformation_radius^2 * parsevalsum2(vars.ψh, grid) / (2 * grid.Lx * grid.Ly)
end

"""
    energy(prob)

Return the problem's (`prob`) domain-averaged total energy of the fluid, that is, the kinetic
energy for a pure barotropic flow *or* the sum of kinetic and potential energies for an equivalent
barotropic flow.
"""
@inline energy(prob) = energy(prob.sol, prob.vars, prob.params, prob.grid)

@inline energy(sol, vars, params::BarotropicQGParams, grid) = kinetic_energy(sol, vars, params, grid)

@inline energy(sol, vars, params::EquivalentBarotropicQGParams, grid) = kinetic_energy(sol, vars, params, grid) + potential_energy(sol, vars, params, grid)

"""
    enstrophy(prob)

Return the problem's (`prob`) domain-averaged enstrophy

```math
\\int \\frac1{2} (q + η)² \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} \\frac1{2} |q̂ + η̂|² .
```
"""
function enstrophy(sol, vars, params, grid)
  @. vars.qh = sol
  return parsevalsum2(vars.qh + params.etah, grid) / (2 * grid.Lx * grid.Ly)
end

@inline enstrophy(prob) = enstrophy(prob.sol, prob.vars, prob.params, prob.grid)

"""
    energy_dissipation(prob)

Return the problem's (`prob`) domain-averaged energy dissipation rate.
"""
@inline energy_dissipation(prob) = energy_dissipation(prob.sol, prob.vars, prob.params, prob.grid)

@inline function energy_dissipation(sol, vars, params::BarotropicQGParams, grid)
  energy_dissipationh = vars.uh # use vars.uh as scratch variable

  @. energy_dissipationh = params.ν * grid.Krsq^(params.nν-1) * abs2(sol)
  CUDA.@allowscalar energy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dissipationh, grid)
end

energy_dissipation(sol, vars, params::EquivalentBarotropicQGParams, grid) = error("not implemented for finite deformation radius")

"""
    enstrophy_dissipation(prob)

Return the problem's (`prob`) domain-averaged enstrophy dissipation rate.
"""
@inline enstrophy_dissipation(prob) = enstrophy_dissipation(prob.sol, prob.vars, prob.params, prob.grid)

@inline function enstrophy_dissipation(sol, vars, params::BarotropicQGParams, grid)
  enstrophy_dissipationh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_dissipationh = params.ν * grid.Krsq^params.nν * abs2(sol)
  CUDA.@allowscalar enstrophy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dissipationh, grid)
end

@inline enstrophy_dissipation(sol, vars, params::EquivalentBarotropicQGParams, grid) = error("not implemented for finite deformation radius")

"""
    energy_work(prob)

Return the problem's (`prob`) domain-averaged rate of work of energy by the forcing `F`.
"""
@inline energy_work(prob) = energy_work(prob.sol, prob.vars, prob.params, prob.grid)

@inline function energy_work(sol, vars::ForcedVars, params::BarotropicQGParams, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable

  @. energy_workh = grid.invKrsq * sol * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_workh, grid)
end

@inline function energy_work(sol, vars::StochasticForcedVars, params::BarotropicQGParams, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable

  @. energy_workh = grid.invKrsq * (vars.prevsol + sol)/2 * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline energy_work(sol, vars, params::EquivalentBarotropicQGParams, grid) = error("not implemented for finite deformation radius")

"""
    enstrophy_work(prob)

Return the problem's (`prob`) domain-averaged rate of work of enstrophy by the forcing ``F``.
"""
@inline enstrophy_work(prob) = enstrophy_work(prob.sol, prob.vars, prob.params, prob.grid)

@inline function enstrophy_work(sol, vars::ForcedVars, params::BarotropicQGParams, grid)
  enstrophy_workh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_workh = sol * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_workh, grid)
end

@inline function enstrophy_work(sol, vars::StochasticForcedVars, params::BarotropicQGParams, grid)
  enstrophy_workh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_workh = (vars.prevsol + sol) / 2 * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_workh, grid)
end

@inline enstrophy_work(sol, vars, params::EquivalentBarotropicQGParams, grid) = error("not implemented for finite deformation radius")

"""
    energy_drag(prob)

Return the problem's (`prob`) extraction of domain-averaged energy by drag ``μ``.
"""
@inline energy_drag(prob) = energy_drag(prob.sol, prob.vars, prob.params, prob.grid)

@inline function energy_drag(sol, vars, params::BarotropicQGParams, grid)
  energy_dragh = vars.uh # use vars.uh as scratch variable

  @. energy_dragh = params.μ * grid.invKrsq * abs2(sol)
  CUDA.@allowscalar energy_dragh[1, 1] = 0
  return  1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dragh, grid)
end

@inline energy_drag(sol, vars, params::EquivalentBarotropicQGParams, grid) = error("not implemented for finite deformation radius")

"""
    enstrophy_drag(prob)

Return the problem's (`prob`) extraction of domain-averaged enstrophy by drag/hypodrag ``μ``.
"""
@inline enstrophy_drag(prob) = enstrophy_drag(prob.sol, prob.vars, prob.params, prob.grid)

@inline function enstrophy_drag(sol, vars, params::BarotropicQGParams, grid)
  enstrophy_dragh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_dragh = params.μ * abs2(sol)
  CUDA.@allowscalar enstrophy_dragh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dragh, grid)
end

@inline enstrophy_drag(sol, vars, params::EquivalentBarotropicQGParams, grid) = error("not implemented for finite deformation radius")

end # module
