module TwoDNavierStokesTracer

export
  fwdtransform!,
  invtransform!,
  Problem,
  set_ζ!,
  updatevars!,

  energy,
  energy_dissipation_hyperviscosity,
  energy_dissipation_hypoviscosity,
  energy_work,
  enstrophy,
  enstrophy_dissipation_hyperviscosity,
  enstrophy_dissipation_hypoviscosity,
  enstrophy_work

using
  FFTW,
  CUDA,
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum, plan_flows_rfft

nothingfunction(args...) = nothing

"""
    Problem(ntracers::Int,
              dev::Device=CPU();
                nx = 256,
                ny = nx,
                Lx = 2π,
                Ly = Lx,
                 ν = 0,
                nν = 1,
                 μ = 0,
                nμ = 0,
                 κ = 0,
                nκ = 1,
                dt = 0.01,
           stepper = "RK4",
             calcF = nothingfunction,
        stochastic = false,
  aliased_fraction = 1/3,
                 T = Float64)

Construct a two-dimensional Navier-Stokes with tracers `problem` on device `dev`.

Keyword arguments
=================
  - `ntracers`: (required) Number of tracers.
  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.
  - `nx`: Number of grid points in ``x``-domain.
  - `ny`: Number of grid points in ``y``-domain.
  - `Lx`: Extent of the ``x``-domain.
  - `Ly`: Extent of the ``y``-domain.
  - `ν`: Small-scale (hyper)-viscosity coefficient.
  - `nν`: (Hyper)-viscosity order, `nν```≥ 1``.
  - `μ`: Large-scale (hypo)-viscosity coefficient.
  - `nμ`: (Hypo)-viscosity order, `nμ```≤ 0``.
  - `κ`: Small-scale diffusivity coefficient.
  - `nκ`: Diffusivity order, `nκ```≥ 1``.
  - `dt`: Time-step.
  - `stepper`: Time-stepping method.
  - `calcF`: Function that calculates the Fourier transform of the forcing, ``F̂``.
  - `stochastic`: `true` or `false`; boolean denoting whether `calcF` is temporally stochastic.
  - `aliased_fraction`: the fraction of high-wavenumbers that are zero-ed out by `dealias!()`.
  - `T`: `Float32` or `Float64`; floating point type used for `problem` data.
"""
function Problem(ntracers::Int,       # number of tracers
                dev::Device=CPU();
  # Numerical parameters
                nx = 256,
                ny = nx,
                Lx = 2π,
                Ly = Lx,
  # Drag and/or hyper-/hypo-viscosity
                 ν = 0.0,
                nν = 1,
                 μ = 0.0,
                nμ = 0,
                 κ = 0.0,
                nκ = 0,
  # Timestepper and equation options
                dt = 0.01,
           stepper = "RK4",
             calcF = nothingfunction,
        stochastic = false,
  # Float type and dealiasing
  aliased_fraction = 1/3,
                 T = Float64)

  grid = TwoDGrid(dev, nx, Lx, ny, Ly; aliased_fraction=aliased_fraction, T=T)

  params = Params(ntracers, ν, nν, μ, nμ, κ, nκ, grid; calcF, dev)

  vars = calcF == nothingfunction ? DecayingVars(dev, grid, params) : (stochastic ? StochasticForcedVars(dev, grid, params) : ForcedVars(dev, grid, params))

  equation = Equation(dev, params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
end


# ----------
# Parameters
# ----------

"""
    Params{T}(ν, nν, μ, nμ, κ, nκ calcF!)

A struct containing the parameters for the two-dimensional Navier-Stokes with tracers. Included are:

$(TYPEDFIELDS)
"""
struct Params{T, Trfft} <: AbstractParams
    "number of tracers"
       ntracers :: Int
    "small-scale (hyper)-viscosity coefficient"
       ν :: T
    "(hyper)-viscosity order, `nν```≥ 1``"
      nν :: Int
    "large-scale (hypo)-viscosity coefficient"
       μ :: T
    "(hypo)-viscosity order, `nμ```≤ 0``"
      nμ :: Int
    "tracer diffusivity coefficient"
       κ :: T
   "tracer diffusivity order"
      nκ :: Int
    "function that calculates the Fourier transform of the forcing, ``F̂``"
      calcF! :: Function
    "rfft plan for FFTs (Derived parameter)"
  rfftplan :: Trfft
end

function Params(ntracers, ν, nν, μ, nμ, κ, nκ, grid; calcF=nothingfunction,  effort=FFTW.MEASURE, dev::Device=CPU()) where TU
  T = eltype(grid)
  A = ArrayType(dev)

   ny, nx = grid.ny , grid.nx
  nkr, nl = grid.nkr, grid.nl
   kr, l  = grid.kr , grid.l
  
  rfftplanlayered = plan_flows_rfft(A{T, 3}(undef, grid.nx, grid.ny, ntracers + 1), [1, 2]; flags=effort)

  return Params(ntracers, ν, nν, μ, nμ, κ, nκ, calcF, rfftplanlayered)
end

"""
    fwdtransform!(varh, var, params)
Compute the Fourier transform of `var` and store it in `varh`.
"""
fwdtransform!(varh, var, params::AbstractParams) = mul!(varh, params.rfftplan, var)

"""
    invtransform!(var, varh, params)
Compute the inverse Fourier transform of `varh` and store it in `var`.
"""
invtransform!(var, varh, params::AbstractParams) = ldiv!(var, params.rfftplan, varh)


# ---------
# Equations
# ---------

"Create a variable for number of layers. First layer describes motion, the other layers are tracers."
numberoflayers(params) = params.ntracers + 1

"""
    Equation(dev, params, grid)
    Return the `equation` for two-dimensional dynamics and the `equations`` for tracer evolution under these dynamics,
    using `params` and `grid`. The first layer of the linear operator ``L`` describes the dynamics and 
    includes (hyper)-viscosity of order ``n_ν`` with coefficient ``ν`` and hypo-viscocity of order ``n_μ`` 
    with coefficient ``μ``. The second layer onwards of ``L`` describe the tracer diffusion of order ``n_κ``
    with coefficient ``κ``.
    
    ```math
    L[:, :, 1] = - ν |𝐤|^{2 n_ν} - μ |𝐤|^{2 n_μ} .
    ```
    ```math
    L[:, :, 2:ntracers+1] = - κ |𝐤|^{2 n_κ}.
    ```
  Plain old viscocity corresponds to ``n_ν=1`` while ``n_μ=0`` corresponds to linear drag.
  The nonlinear term is computed via function `calcN!()`.
"""
function Equation(dev, params, grid)
  #L = hyperviscosity(dev, params, grid)
  nlayers = numberoflayers(params)
  T = eltype(grid)
  L = ArrayType(dev){T}(undef, (grid.nkr, grid.nl, nlayers))
  @views @. L[:,:,1] = - params.ν * grid.Krsq^params.nν - params.μ * grid.Krsq^params.nμ
  @views @. L[:,:,2:nlayers] = - params.κ * grid.Krsq^params.nκ
  # Need to add diffusivities for different layers
  @views @. L[1, 1, :] = 0
  
  return FourierFlows.Equation(L, calcN!, grid)
end

# ----
# Vars
# ----

"""
    Vars{Aphys, Atrans, F, P}(ζ, u, v, ζh, uh, vh, Fh, prevsol)

The variables for two-dimensional Navier-Stokes:

$(FIELDS)
"""
struct Vars{Aphys3D, Aphys, Atrans3D, Atrans, F, P} <: AbstractVars
    "relative vorticity ([:, :, 1] layer) and tracers (other layers)"
        ζ :: Aphys3D
    "x-component of velocity"
        u :: Aphys
    "y-component of velocity"
        v :: Aphys
    "Fourier transform of relative vorticity ([:, :, 1] layer) and tracers (other layers)"
       ζh :: Atrans3D
    "Fourier transform of x-component of velocity"
       uh :: Atrans
    "Fourier transform of y-component of velocity"
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
    DecayingVars(dev, grid)

Return the `vars` for unforced two-dimensional Navier-Stokes problem on device `dev` and
with `grid`.
"""
function DecayingVars(::Dev, grid::AbstractGrid, params) where Dev
  nlayers = numberoflayers(params)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny, nlayers) ζ
  @devzeros Dev T (grid.nx, grid.ny) u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, nlayers) ζh
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) uh vh

  return Vars(ζ, u, v, ζh, uh, vh, nothing, nothing)
end

"""
    ForcedVars(dev, grid)

Return the `vars` for forced two-dimensional Navier-Stokes on device `dev` and with `grid`.
"""
function ForcedVars(dev::Dev, grid::AbstractGrid, params) where Dev
  nlayers = numberoflayers(params)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny, nlayers) ζ
  @devzeros Dev T (grid.nx, grid.ny) u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, nlayers) ζh
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) uh vh Fh

  return Vars(ζ, u, v, ζh, uh, vh, Fh, nothing)
end

"""
    StochasticForcedVars(dev, grid)

Return the `vars` for stochastically forced two-dimensional Navier-Stokes on device `dev` and
with `grid`.
"""
function StochasticForcedVars(dev::Dev, grid::AbstractGrid, params) where Dev
  nlayers = numberoflayers(params)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny, nlayers) ζ
  @devzeros Dev T (grid.nx, grid.ny) u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, nlayers) ζh prevsol
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) uh vh Fh

  return Vars(ζ, u, v, ζh, uh, vh, Fh, prevsol)
end


# -------
# Solvers
# -------

"""
    calcN_advection!(N, sol, t, clock, vars, params, grid)

Calculate the Fourier transform of the advection term, ``- 𝖩(ψ, ζ)`` in conservative form,
i.e., ``- ∂_x[(∂_y ψ)ζ] - ∂_y[(∂_x ψ)ζ]`` and store it in `N`:

```math
N = - \\widehat{𝖩(ψ, ζ)} = - i k_x \\widehat{u ζ} - i k_y \\widehat{v ζ} .
```

Note the first layer ``N[:, :, 1]`` of this advection term corresponds to advection of vorticity ``ζ``.
The subsequent layers of ``N[:, :, 2:ntracers+1]`` correspond to advection of the each tracer.
"""
function calcN_advection!(N, sol, t, clock, vars, params, grid)
  nlayers = numberoflayers(params)
  @. vars.ζh = sol

  for j in 1:nlayers
    @. vars.uh =   im * grid.l  * grid.invKrsq * sol[:, :, 1]
    @. vars.vh = - im * grid.kr * grid.invKrsq * sol[:, :, 1]
    ldiv!(vars.u, grid.rfftplan, vars.ζh[:, :, j])
    vars.ζ[:, : ,j] = vars.u
    ldiv!(vars.u, grid.rfftplan, vars.uh)
    ldiv!(vars.v, grid.rfftplan, vars.vh)

    uζ, uζh = vars.u, vars.uh         # use vars.u, vars.uh as scratch variables
    vζ, vζh = vars.v, vars.vh         # use vars.v, vars.vh as scratch variables

    @. uζ *= vars.ζ[:, :, j]                   # u*ζ [note u is 2D array and ζ is 3D array]
    @. vζ *= vars.ζ[:, :, j]                   # v*ζ [note v is 2D array and ζ is 3D array]

    mul!(uζh, grid.rfftplan, uζ) # \hat{u*ζ}
    mul!(vζh, grid.rfftplan, vζ) # \hat{v*ζ}

    @views @. N[:, :, j] = - im * grid.kr * uζh - im * grid.l * vζh
  end

  return nothing
end

"""
    calcN!(N, sol, t, clock, vars, params, grid)

Calculate the nonlinear term, that is the advection term and the forcing,

```math
N = - \\widehat{𝖩(ψ, ζ)} + F̂ .
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
addforcing!(N, sol, t, clock, vars::DecayingVars, params, grid) = nothing

function addforcing!(N, sol, t, clock, vars::ForcedVars, params, grid)
  #if !isnothing(vars.Fh) # Ignore if there is no forcing
    params.calcF!(vars.Fh, sol, t, clock, vars, params, grid)
    @views @. N[:, :, 1] += vars.Fh 
  #end

  return nothing
end
function addforcing!(N, sol, t, clock, vars::StochasticForcedVars, params, grid)
  if t == clock.t # not a substep
    @. vars.prevsol = sol[:, :, 1] # sol at previous time-step is needed to compute budgets for stochastic forcing
    params.calcF!(vars.Fh, sol, t, clock, vars, params, grid)
  end
  @views @. N[:, :, 1] += vars.Fh 

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
  
  @. vars.ζh = sol
  @. vars.uh =   im * grid.l  * grid.invKrsq * sol[:,:,1]
  @. vars.vh = - im * grid.kr * grid.invKrsq * sol[:,:,1]
  
  invtransform!(vars.ζ, deepcopy(vars.ζh), prob.params) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh)) # deepcopy() since inverse real-fft destroys its input
  ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh)) # deepcopy() since inverse real-fft destroys its input

  return nothing
end

"""
    set_ζ_and_tracers!(prob, ζ)

Set the first solution layer `sol[:,:,1]` as the transform of `ζ` and lower layers as transform of `tracers`.
Then update variables in `vars`.
"""
function set_ζ_and_tracers!(prob, ζ, tracers)
  full_ζ = cat(ζ, tracers, dims=3)     # append b and tracers for use in sol
  fwdtransform!(prob.sol, full_ζ, prob.params)
  @views CUDA.@allowscalar prob.sol[1, 1, 1] = 0 # zero domain average

  updatevars!(prob)

  return nothing
end

"""
    energy(prob)

Return the domain-averaged kinetic energy. Since ``u² + v² = |{\\bf ∇} ψ|²``, the domain-averaged
kinetic energy is

```math
\\int \\frac1{2} |{\\bf ∇} ψ|² \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} \\frac1{2} |𝐤|² |ψ̂|² .
```
"""
@inline function energy(prob)
  sol, vars, grid = prob.sol, prob.vars, prob.grid
  energyh = vars.uh # use vars.uh as scratch variable

  @. energyh = 1 / 2 * grid.invKrsq * abs2(sol[:,:,1])
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energyh, grid)
end

"""
    enstrophy(prob)

Returns the domain-averaged enstrophy,

```math
\\int \\frac1{2} ζ² \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} \\frac1{2} |ζ̂|² .
```
"""
@inline function enstrophy(prob)
  sol, grid = prob.sol, prob.grid
  return 1 / (2 * grid.Lx * grid.Ly) * parsevalsum(abs2.(sol[:,:,1]), grid)
end

"""
    energy_dissipation(prob, ξ, nξ)

Return the domain-averaged energy dissipation rate done by the viscous term,

```math
- ξ (-1)^{n_ξ+1} \\int ψ ∇^{2n_ξ} ζ \\frac{𝖽x 𝖽y}{L_x L_y} = - ξ \\sum_{𝐤} |𝐤|^{2(n_ξ-1)} |ζ̂|² ,
```
where ``ξ`` and ``nξ`` could be either the (hyper)-viscosity coefficient ``ν`` and its order
``n_ν``, or the hypo-viscocity coefficient ``μ`` and its order ``n_μ``.
"""
@inline function energy_dissipation(prob, ξ, nξ)
  sol, vars, grid = prob.sol, prob.vars, prob.grid
  energy_dissipationh = vars.uh # use vars.uh as scratch variable

  @. energy_dissipationh = - ξ * grid.Krsq^(nξ - 1) * abs2(sol[:,:,1])
  CUDA.@allowscalar energy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dissipationh, grid)
end

"""
    energy_dissipation_hyperviscosity(prob, ξ, νξ)

Return the domain-averaged energy dissipation rate done by the ``ν`` (hyper)-viscosity.
"""
energy_dissipation_hyperviscosity(prob) = energy_dissipation(prob, prob.params.ν, prob.params.nν)

"""
    energy_dissipation_hypoviscosity(prob, ξ, νξ)

Return the domain-averaged energy dissipation rate done by the ``μ`` (hypo)-viscosity.
"""
energy_dissipation_hypoviscosity(prob) = energy_dissipation(prob, prob.params.μ, prob.params.nμ)

"""
    enstrophy_dissipation(prob, ξ, νξ)

Return the domain-averaged enstrophy dissipation rate done by the viscous term,

```math
ξ (-1)^{n_ξ+1} \\int ζ ∇^{2n_ξ} ζ \\frac{𝖽x 𝖽y}{L_x L_y} = - ξ \\sum_{𝐤} |𝐤|^{2n_ξ} |ζ̂|² ,
```

where ``ξ`` and ``nξ`` could be either the (hyper)-viscosity coefficient ``ν`` and its order
``n_ν``, or the hypo-viscocity coefficient ``μ`` and its order ``n_μ``.
"""
@inline function enstrophy_dissipation(prob, ξ, nξ)
  sol, vars, grid = prob.sol, prob.vars, prob.grid
  enstrophy_dissipationh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_dissipationh = - ξ * grid.Krsq^nξ * abs2(sol[:,:,1])
  CUDA.@allowscalar enstrophy_dissipationh[1, 1] = 0
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dissipationh, grid)
end

"""
    enstrophy_dissipation_hyperviscosity(prob, ξ, νξ)

Return the domain-averaged enstrophy dissipation rate done by the ``ν`` (hyper)-viscosity.
"""
enstrophy_dissipation_hyperviscosity(prob) = enstrophy_dissipation(prob, prob.params.ν, prob.params.nν)

"""
    enstrophy_dissipation_hypoviscosity(prob, ξ, νξ)

Return the domain-averaged enstrophy dissipation rate done by the ``μ`` (hypo)-viscosity.
"""
enstrophy_dissipation_hypoviscosity(prob) = enstrophy_dissipation(prob, prob.params.μ, prob.params.nμ)

"""
    energy_work(prob)
    energy_work(sol, vars, grid)

Return the domain-averaged rate of work of energy by the forcing ``F``,

```math
- \\int ψ F \\frac{𝖽x 𝖽y}{L_x L_y} = - \\sum_{𝐤} ψ̂ F̂^* .
```
"""
@inline function energy_work(sol, vars::ForcedVars, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable

  @. energy_workh = grid.invKrsq * sol[:,:,1] * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_workh, grid)
end

@inline function energy_work(sol, vars::StochasticForcedVars, grid)
  energy_workh = vars.uh # use vars.uh as scratch variable

  @. energy_workh = grid.invKrsq * (vars.prevsol + sol[:,:,1]) / 2 * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_workh, grid)
end

@inline energy_work(prob) = energy_work(prob.sol, prob.vars, prob.grid)

"""
    enstrophy_work(prob)
    enstrophy_work(sol, vars, grid)

Return the domain-averaged rate of work of enstrophy by the forcing ``F``,

```math
\\int ζ F \\frac{𝖽x 𝖽y}{L_x L_y} = \\sum_{𝐤} ζ̂ F̂^* .
```
"""
@inline function enstrophy_work(sol, vars::ForcedVars, grid)
  enstrophy_workh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_workh = sol[:,:,1] * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_workh, grid)
end

@inline function enstrophy_work(sol, vars::StochasticForcedVars, grid)
  enstrophy_workh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_workh = (vars.prevsol + sol[:,:,1]) / 2 * conj(vars.Fh)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_workh, grid)
end

@inline enstrophy_work(prob) = enstrophy_work(prob.sol, prob.vars, prob.grid)

end # module
