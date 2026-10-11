module MultiLayerQG

export
  fwdtransform!,
  invtransform!,
  streamfunctionfrompv!,
  pvfromstreamfunction!,
  updatevars!,

  set_q!,
  set_ψ!,
  energies,
  fluxes

using
  FFTW,
  CUDA,
  LinearAlgebra,
  StaticArrays,
  Reexport,
  DocStringExtensions,
  KernelAbstractions

@reexport using FourierFlows

using FourierFlows: parsevalsum, parsevalsum2, superzeros, plan_flows_rfft, CPU, GPU
using KernelAbstractions.Extras.LoopInfo: @unroll

nothingfunction(args...) = nothing

"""
    Problem(nlayers :: Int,
                        dev = CPU();
                         nx = 128,
                         ny = nx,
                         Lx = 2π,
                         Ly = Lx,
                         f₀ = 1.0,
                          β = 0.0,
                          U = zeros(nlayers),
                          H = 1/nlayers * ones(nlayers),
                          b = -(1 .+ 1/nlayers * (0:nlayers-1)),
                        eta = nothing,
    topographic_pv_gradient = (0, 0),
                          μ = 0,
                          ν = 0,
                         nν = 1,
                         dt = 0.01,
                    stepper = "RK4",
                     calcFq = nothingfunction,
                 stochastic = false,
                     linear = false,
           aliased_fraction = 1/3,
                          T = Float64)

Construct a multi-layer quasi-geostrophic problem with `nlayers` fluid layers on device `dev`.

Arguments
=========
- `nlayers`: (required) Number of fluid layers.
- `dev`: (required) `CPU()` (default) or `GPU()`; computer architecture used to time-step `problem`.

Keyword arguments
=================
  - `nx`: Number of grid points in ``x``-domain.
  - `ny`: Number of grid points in ``y``-domain.
  - `Lx`: Extent of the ``x``-domain.
  - `Ly`: Extent of the ``y``-domain.
  - `f₀`: Constant planetary vorticity.
  - `β`: Planetary vorticity ``y``-gradient.
  - `U`: Imposed background constant zonal flow ``U(y)`` in each fluid layer.
  - `H`: Rest height of each fluid layer.
  - `b`: Boussinesq buoyancy of each fluid layer.
  - `eta`: Periodic component of the topographic potential vorticity.
  - `topographic_pv_gradient`: The ``(x, y)`` components of the topographic PV large-scale gradient.
  - `μ`: Linear bottom drag coefficient.
  - `ν`: Small-scale (hyper)-viscosity coefficient.
  - `nν`: (Hyper)-viscosity order, `nν```≥ 1``.
  - `dt`: Time-step.
  - `stepper`: Time-stepping method.
  - `calcF`: Function that calculates the Fourier transform of the forcing, ``F̂``.
  - `stochastic`: `true` or `false` (default); boolean denoting whether `calcF` is temporally stochastic.
  - `linear`: `true` or `false` (default); boolean denoting whether the linearized equations of motions are used.
  - `aliased_fraction`: the fraction of high wavenumbers that are zero-ed out by `dealias!()`.
  - `T`: `Float32` or `Float64` (default); floating point type used for `problem` data.
"""
function Problem(nlayers::Int,                                     # number of fluid layers
                          dev = CPU();
              # Numerical parameters
                           nx = 128,
                           ny = nx,
                           Lx = 2π,
                           Ly = Lx,
              # Physical parameters
                           f₀ = 1.0,                               # Coriolis parameter
                            β = 0.0,                               # y-gradient of Coriolis parameter
                            U = zeros(nlayers),                    # imposed zonal flow U(y) in each layer
                            H = 1/nlayers * ones(nlayers),         # rest height of each layer
                            b = -(1 .+ 1/nlayers * (0:nlayers-1)), # Boussinesq buoyancy of each layer
                          eta = nothing,                           # periodic component of the topographic PV
      topographic_pv_gradient = (0, 0),                            # tuple with the ``(x, y)`` components of topographic PV large-scale gradient
              # Bottom Drag and/or (hyper)-viscosity
                            μ = 0,
                            ν = 0,
                           nν = 1,
              # Timestepper and equation options
                           dt = 0.01,
                      stepper = "RK4",
                       calcFq = nothingfunction,
                   stochastic = false,
                       linear = false,
              # Float type and dealiasing
             aliased_fraction = 1/3,
                            T = Float64)

  if nlayers == 1
    @warn """MultiLayerQG module does work for single-layer configuration but may not be as
    optimized. We suggest using SingleLayerQG module for single-layer QG simulation unless
    you have reasons to use MultiLayerQG in a single-layer configuration, e.g., you want to
    compare solutions with varying number of fluid layers."""
  end

  # topographic PV
  eta === nothing && (eta = zeros(dev, T, (nx, ny)))

  grid = TwoDGrid(dev; nx, Lx, ny, Ly, aliased_fraction, T)

  params = Params(nlayers, f₀, β, b, H, U, eta, topographic_pv_gradient, μ, ν, nν, grid; calcFq)

  vars = calcFq == nothingfunction ? DecayingVars(grid, params) : (stochastic ? StochasticForcedVars(grid, params) : ForcedVars(grid, params))

  equation = linear ? LinearEquation(params, grid) : Equation(params, grid)

  FourierFlows.Problem(equation, stepper, dt, grid, vars, params)
end

"""
    struct Params{T, Aphys3D, Aphys2D, Atrans4D, Trfft} <: AbstractParams

The parameters for the `MultiLayerQG` problem.

$(TYPEDFIELDS)
"""
struct Params{T, Aphys3D, Aphys2D, Atrans4D, Trfft} <: AbstractParams
  # prescribed params
    "number of fluid layers"
   nlayers :: Int
    "constant planetary vorticity"
        f₀ :: T
    "planetary vorticity ``y``-gradient"
         β :: T
    "tuple with Boussinesq buoyancy of each fluid layer"
         b :: Tuple
    "tuple with rest height of each fluid layer"
         H :: Tuple
    "array with imposed constant zonal flow ``U(y)`` in each fluid layer"
         U :: Aphys3D
    "array containing the topographic PV"
       eta :: Aphys2D
    "tuple containing the ``(x, y)`` components of topographic PV large-scale gradient"
    topographic_pv_gradient :: Tuple{T, T}
    "linear bottom drag coefficient"
         μ :: T
    "small-scale (hyper)-viscosity coefficient"
         ν :: T
    "(hyper)-viscosity order, `nν```≥ 1``"
        nν :: Int
    "function that calculates the Fourier transform of the forcing, ``F̂``"
   calcFq! :: Function

  # derived params
    "tuple with the reduced gravity constants for each fluid interface"
        g′ :: Tuple
    "array containing ``x``-gradient of PV due to eta in each fluid layer"
        Qx :: Aphys3D
    "array containing ``y``-gradient of PV due to ``β``, ``U``, and topographic PV in each fluid layer"
        Qy :: Aphys3D
    "array containing coefficients for getting PV from streamfunction"
         S :: Atrans4D
    "array containing coefficients for inverting PV to streamfunction"
       S⁻¹ :: Atrans4D
    "rfft plan for FFTs"
  rfftplan :: Trfft
end

"""
    struct SingleLayerParams{T, Aphys3D, Aphys2D, Trfft} <: AbstractParams

The parameters for a single-layer `MultiLayerQG` problem.

$(TYPEDFIELDS)
"""
struct SingleLayerParams{T, Aphys3D, Aphys2D, Trfft} <: AbstractParams
  # prescribed params
    "planetary vorticity ``y``-gradient"
         β :: T
    "array with imposed constant zonal flow ``U(y)``"
         U :: Aphys3D
     "array containing the periodic component of the topographic PV"
       eta :: Aphys2D
    "tuple containing the ``(x, y)`` components of topographic PV large-scale gradient"
    topographic_pv_gradient :: Tuple{T, T}
    "linear drag coefficient"
         μ :: T
    "small-scale (hyper)-viscosity coefficient"
         ν :: T
    "(hyper)-viscosity order, `nν```≥ 1``"
        nν :: Int
    "function that calculates the Fourier transform of the forcing, ``F̂``"
   calcFq! :: Function

  # derived params
    "array containing ``x``-gradient of PV due to topographic PV"
        Qx :: Aphys3D
    "array containing ``y``-gradient of PV due to ``β``, ``U``, and topographic PV"
        Qy :: Aphys3D
    "rfft plan for FFTs"
  rfftplan :: Trfft
end

function convert_U_to_U3D(dev, nlayers, grid, U::AbstractArray{TU, 1}) where TU
  T = eltype(grid)

  if length(U) == nlayers
    U_2D = zeros(dev, T, (1, nlayers))
    U_2D[:] = U
    U_2D = repeat(U_2D, outer=(grid.ny, 1))
  else
    U_2D = zeros(dev, T, (grid.ny, 1))
    U_2D[:] = U
  end

  U_3D = zeros(dev, T, (1, grid.ny, nlayers))
  @views U_3D[1, :, :] = U_2D

  return U_3D
end

function convert_U_to_U3D(dev, nlayers, grid, U::AbstractArray{TU, 2}) where TU
  T = eltype(grid)
  U_3D = zeros(dev, T, (1, grid.ny, nlayers))
  @views U_3D[1, :, :] = U

  return U_3D
end

function convert_U_to_U3D(dev, nlayers, grid, U::Number)
  T = eltype(grid)
  A = device_array(dev)
  U_3D = reshape(repeat([T(U)], outer=(grid.ny, 1)), (1, grid.ny, nlayers))

  return A(U_3D)
end

function Params(nlayers::Int, f₀, β, b, H, U, eta, topographic_pv_gradient, μ, ν, nν, grid::TwoDGrid;
                calcFq=nothingfunction, effort=FFTW.MEASURE)
  dev = grid.device
  T = eltype(grid)
  A = device_array(dev)

   ny, nx = grid.ny , grid.nx
  nkr, nl = grid.nkr, grid.nl
   kr, l  = grid.kr , grid.l

    U = convert_U_to_U3D(dev, nlayers, grid, U)

  Uyy = real.(ifft(-l.^2 .* fft(U)))
  Uyy = CUDA.@allowscalar repeat(Uyy, outer=(nx, 1, 1))

  # Calculate the periodic components of the topographic PV gradients
  etah = rfft(A(eta))
  etax = irfft(im * kr .* etah, nx)   # ∂η/∂x
  etay = irfft(im * l  .* etah, nx)   # ∂η/∂y

  # Add large-scale topographic PV gradient
  topographic_pv_gradient = T.(topographic_pv_gradient)
  @. etax += topographic_pv_gradient[1]
  @. etay += topographic_pv_gradient[2]

  Qx = zeros(dev, T, (nx, ny, nlayers))
  @views @. Qx[:, :, nlayers] += etax

  Qy = zeros(dev, T, (nx, ny, nlayers))
  Qy = T(β) .- Uyy  # T(β) ensures that Qy remains same type as U
  @views @. Qy[:, :, nlayers] += etay

  rfftplanlayered = plan_flows_rfft(A{T, 3}(undef, grid.nx, grid.ny, nlayers), [1, 2]; flags=effort)

  if nlayers==1
    return SingleLayerParams(T(β), U, eta, topographic_pv_gradient, T(μ), T(ν), nν, calcFq, Qx, Qy, rfftplanlayered)

  else # if nlayers≥2

    b = reshape(T.(b), (1,  1, nlayers))
    H = Tuple(T.(H))

    g′ = b[1:nlayers-1] - b[2:nlayers] # reduced gravity at each interface

    Fm = @. T(f₀^2 / (g′ * H[2:nlayers]))
    Fp = @. T(f₀^2 / (g′ * H[1:nlayers-1]))

    typeofSkl = SArray{Tuple{nlayers, nlayers}, T, 2, nlayers^2} # StaticArrays of type T and dims = (nlayers, nlayers)

    S = Array{typeofSkl, 2}(undef, (nkr, nl))    # Array of StaticArrays
    calcS!(S, Fp, Fm, nlayers, grid)

    S⁻¹ = Array{typeofSkl, 2}(undef, (nkr, nl))  # Array of StaticArrays
    calcS⁻¹!(S⁻¹, Fp, Fm, nlayers, grid)

    S, S⁻¹ = A(S), A(S⁻¹) # convert to appropriate ArrayType

    @views Qy[:, :, 1] = @. Qy[:, :, 1] - Fp[1] * (U[:, :, 2] - U[:, :, 1])
    for j = 2:nlayers-1
      @views Qy[:, :, j] = @. Qy[:, :, j] - Fp[j] * (U[:, :, j+1] - U[:, :, j]) - Fm[j-1] * (U[:, :, j-1] - U[:, :, j])
    end
    @views Qy[:, :, nlayers] = @. Qy[:, :, nlayers] - Fm[nlayers-1] * (U[:, :, nlayers-1] - U[:, :, nlayers])

    return Params(nlayers, T(f₀), T(β), Tuple(T.(b)), T.(H), U, eta, topographic_pv_gradient, T(μ), T(ν), nν, calcFq, Tuple(T.(g′)), Qx, Qy, S, S⁻¹, rfftplanlayered)
  end
end

numberoflayers(params) = params.nlayers
numberoflayers(::SingleLayerParams) = 1

# ---------
# Equations
# ---------

"""
    hyperviscosity(params, grid)

Return the linear operator `L` that corresponds to (hyper)-viscosity of order ``n_ν`` with
coefficient ``ν`` for ``n`` fluid layers.
```math
L_j = - ν |𝐤|^{2 n_ν}, \\ j = 1, ..., n .
```
"""
function hyperviscosity(params, grid)
  dev = grid.device
  T = eltype(grid)

  L = device_array(dev){T}(undef, (grid.nkr, grid.nl, numberoflayers(params)))
  @. L = - params.ν * grid.Krsq^params.nν
  @views @. L[1, 1, :] = 0

  return L
end

"""
    LinearEquation(params, grid)

Return the equation for a multi-layer quasi-geostrophic problem with `params` and `grid`.
The linear operator ``L`` includes only (hyper)-viscosity and is computed via
`hyperviscosity(params, grid)`.

The nonlinear term is computed via [`calcNlinear!`](@ref).
"""
function LinearEquation(params, grid)
  L = hyperviscosity(params, grid)

  return FourierFlows.Equation(L, calcNlinear!, grid)
end

"""
    Equation(params, grid)

Return the equation for a multi-layer quasi-geostrophic problem with `params` and `grid`.
The linear operator ``L`` includes only (hyper)-viscosity and is computed via
`hyperviscosity(params, grid)`.

The nonlinear term is computed via [`calcN!`](@ref GeophysicalFlows.MultiLayerQG.calcN!).
"""
function Equation(params, grid)
  L = hyperviscosity(params, grid)

  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

"""
    struct Vars{Aphys, Atrans, F, P} <: AbstractVars

The variables for multi-layer QG problem.

$(FIELDS)
"""
struct Vars{Aphys, Atrans, F, P} <: AbstractVars
    "relative vorticity + vortex stretching"
        q :: Aphys
    "streamfunction"
        ψ :: Aphys
    "``x``-component of velocity"
        u :: Aphys
    "``y``-component of velocity"
        v :: Aphys
    "Fourier transform of relative vorticity + vortex stretching"
       qh :: Atrans
    "Fourier transform of streamfunction"
       ψh :: Atrans
    "Fourier transform of ``x``-component of velocity"
       uh :: Atrans
    "Fourier transform of ``y``-component of velocity"
       vh :: Atrans
    "Fourier transform of forcing"
      Fqh :: F
    "`sol` at previous time-step"
  prevsol :: P
end

const DecayingVars = Vars{<:AbstractArray, <:AbstractArray, Nothing, Nothing}
const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    DecayingVars(grid, params)

Return the variables for an unforced multi-layer QG problem with `grid` and `params`.
"""
function DecayingVars(grid, params)
  Dev = typeof(grid.device)
  T = eltype(grid)
  nlayers = numberoflayers(params)

  @devzeros Dev T (grid.nx, grid.ny, nlayers) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, nlayers) qh ψh uh vh

  return Vars(q, ψ, u, v, qh, ψh, uh, vh, nothing, nothing)
end

"""
    ForcedVars(grid, params)

Return the variables for a forced multi-layer QG problem with `grid` and `params`.
"""
function ForcedVars(grid, params)
  Dev = typeof(grid.device)
  T = eltype(grid)
  nlayers = numberoflayers(params)

  @devzeros Dev T (grid.nx, grid.ny, nlayers) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, nlayers) qh ψh uh vh Fqh

  return Vars(q, ψ, u, v, qh, ψh, uh, vh, Fqh, nothing)
end

"""
    StochasticForcedVars(grid, params)

Return the variables for a forced multi-layer QG problem with `grid` and `params`.
"""
function StochasticForcedVars(grid, params)
  Dev = typeof(grid.device)
  T = eltype(grid)
  nlayers = numberoflayers(params)

  @devzeros Dev T (grid.nx, grid.ny, nlayers) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, nlayers) qh ψh uh vh Fqh prevsol

  return Vars(q, ψ, u, v, qh, ψh, uh, vh, Fqh, prevsol)
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

"""
    pv_streamfunction_kernel!(y, M, x, ::Val{N}) where N

Kernel for the PV to streamfunction conversion steps. The kernel performs the
matrix multiplication

```math
y = M x
```

for every wavenumber, where ``y`` and ``x`` are column-vectors of length `nlayers`.
This can be used to perform `qh = params.S * ψh` or `ψh = params.S⁻¹ qh`.

StaticVectors are used to efficiently perform the matrix-vector multiplication.
"""
@kernel function pv_streamfunction_kernel!(y, M, x, ::Val{N}) where N
  i, j = @index(Global, NTuple)

  x_tuple = ntuple(Val(N)) do n
    @inbounds x[i, j, n]
  end

  T = eltype(x)
  x_sv = SVector{N, T}(x_tuple)
  y_sv = @inbounds M[i, j] * x_sv

  ntuple(Val(N)) do n
    @inbounds y[i, j, n] = y_sv[n]
  end
end

"""
    pvfromstreamfunction!(qh, ψh, params, grid)

Obtain the Fourier transform of the PV from the streamfunction `ψh` in each layer using
`qh = params.S * ψh`.

The matrix multiplications are done via launching a kernel. We use a work layout over
which the kernel is launched.
"""
function pvfromstreamfunction!(qh, ψh, params, grid)
  # Larger workgroups are generally more efficient. For more generality, we could put an
  # if statement that incurs different behavior when either nkl or nl are less than 8.
  workgroup = 8, 8

  # The worksize determines how many times the kernel is run
  worksize = grid.nkr, grid.nl

  # Instantiates the kernel for relevant backend device
  backend = KernelAbstractions.get_backend(qh)
  kernel! = pv_streamfunction_kernel!(backend, workgroup, worksize)

  # Launch the kernel
  S, nlayers = params.S, params.nlayers
  kernel!(qh, S, ψh, Val(nlayers))

  # Ensure that no other operations occur until the kernel has finished
  KernelAbstractions.synchronize(backend)

  return nothing
end

"""
    pvfromstreamfunction!(qh, ψh, params::SingleLayerParams, grid)

Obtain the Fourier transform of the PV from the streamfunction `ψh` for the special
case of a single fluid layer configuration. In this case, ``q̂ = - (k_x² + k_y²) ψ̂``.
"""
function pvfromstreamfunction!(qh, ψh, params::SingleLayerParams, grid)
  @. qh = -grid.Krsq * ψh

  return nothing
end

"""
    streamfunctionfrompv!(ψh, qh, params, grid)

Invert the PV to obtain the Fourier transform of the streamfunction `ψh` in each layer from
`qh` using `ψh = params.S⁻¹ * qh`.

The matrix multiplications are done via launching a kernel. We use a work layout over
which the kernel is launched.
"""
function streamfunctionfrompv!(ψh, qh, params, grid)
  # Larger workgroups are generally more efficient. For more generality, we could put an
  # if statement that incurs different behavior when either nkl or nl are less than 8.
  workgroup = 8, 8

  # The worksize determines how many times the kernel is run
  worksize = grid.nkr, grid.nl

  # Instantiates the kernel for relevant backend device
  backend = KernelAbstractions.get_backend(ψh)
  kernel! = pv_streamfunction_kernel!(backend, workgroup, worksize)

  # Launch the kernel
  S⁻¹, nlayers = params.S⁻¹, params.nlayers
  kernel!(ψh, S⁻¹, qh, Val(nlayers))

  # Ensure that no other operations occur until the kernel has finished
  KernelAbstractions.synchronize(backend)

  return nothing
end

"""
    streamfunctionfrompv!(ψh, qh, params::SingleLayerParams, grid)

Invert the PV to obtain the Fourier transform of the streamfunction `ψh` for the special
case of a single fluid layer configuration. In this case, ``ψ̂ = - (k_x² + k_y²)⁻¹ q̂``.
"""
function streamfunctionfrompv!(ψh, qh, params::SingleLayerParams, grid)
  @. ψh = -grid.invKrsq * qh

  return nothing
end

"""
    calcS!(S, Fp, Fm, nlayers, grid)

Construct the array ``𝕊``, which consists of `nlayer` x `nlayer` static arrays ``𝕊_𝐤`` that
relate the ``q̂_j``'s and ``ψ̂_j``'s for every wavenumber: ``q̂_𝐤 = 𝕊_𝐤 ψ̂_𝐤``.
"""
function calcS!(S, Fp, Fm, nlayers, grid)
  F = Matrix(Tridiagonal(Fm, -([Fp; 0] + [0; Fm]), Fp))

  for n=1:grid.nl, m=1:grid.nkr
    k² = CUDA.@allowscalar grid.Krsq[m, n]
    Skl = SMatrix{nlayers, nlayers}(- k² * I + F)
    S[m, n] = Skl
  end

  return nothing
end

"""
    calcS⁻¹!(S, Fp, Fm, nlayers, grid)

Construct the array ``𝕊⁻¹``, which consists of `nlayer` x `nlayer` static arrays ``(𝕊_𝐤)⁻¹``
that relate the ``q̂_j``'s and ``ψ̂_j``'s for every wavenumber: ``ψ̂_𝐤 = (𝕊_𝐤)⁻¹ q̂_𝐤``.
"""
function calcS⁻¹!(S⁻¹, Fp, Fm, nlayers, grid)
  F = Matrix(Tridiagonal(Fm, -([Fp; 0] + [0; Fm]), Fp))

  for n=1:grid.nl, m=1:grid.nkr
    k² = CUDA.@allowscalar grid.Krsq[m, n] == 0 ? 1 : grid.Krsq[m, n]
    Skl = - k² * I + F
    S⁻¹[m, n] = SMatrix{nlayers, nlayers}(I / Skl)
  end

  T = eltype(grid)
  S⁻¹[1, 1] = SMatrix{nlayers, nlayers}(zeros(T, (nlayers, nlayers)))

  return nothing
end


# -------
# Solvers
# -------

"""
    calcN!(N, sol, t, clock, vars, params, grid)

Compute the nonlinear term, that is the advection term, the bottom drag, and the forcing:

```math
N_j = - \\widehat{𝖩(ψ_j, q_j)} - \\widehat{U_j ∂_x Q_j} - \\widehat{U_j ∂_x q_j}
 + \\widehat{(∂_y ψ_j)(∂_x Q_j)} - \\widehat{(∂_x ψ_j)(∂_y Q_j)} + δ_{j, n} μ |𝐤|^2 ψ̂_n + F̂_j .
```
"""
function calcN!(N, sol, t, clock, vars, params, grid)
  nlayers = numberoflayers(params)

  dealias!(sol, grid)

  calcN_advection!(N, sol, vars, params, grid)

  @views @. N[:, :, nlayers] += params.μ * grid.Krsq * vars.ψh[:, :, nlayers]   # bottom linear drag

  addforcing!(N, sol, t, clock, vars, params, grid)

  return nothing
end

"""
    calcNlinear!(N, sol, t, clock, vars, params, grid)

Compute the nonlinear term of the linearized equations:

```math
N_j = - \\widehat{U_j ∂_x Q_j} - \\widehat{U_j ∂_x q_j} + \\widehat{(∂_y ψ_j)(∂_x Q_j)}
- \\widehat{(∂_x ψ_j)(∂_y Q_j)} + δ_{j, n} μ |𝐤|^2 ψ̂_n + F̂_j .
```
"""
function calcNlinear!(N, sol, t, clock, vars, params, grid)
  nlayers = numberoflayers(params)

  calcN_linearadvection!(N, sol, vars, params, grid)
  @views @. N[:, :, nlayers] += params.μ * grid.Krsq * vars.ψh[:, :, nlayers]   # bottom linear drag
  addforcing!(N, sol, t, clock, vars, params, grid)

  return nothing
end

"""
    calcN_advection!(N, sol, vars, params, grid)

Compute the advection term and store it in `N`:

```math
N_j = - \\widehat{𝖩(ψ_j, q_j)} - \\widehat{U_j ∂_x Q_j} - \\widehat{U_j ∂_x q_j}
 + \\widehat{(∂_y ψ_j)(∂_x Q_j)} - \\widehat{(∂_x ψ_j)(∂_y Q_j)} .
```
"""
function calcN_advection!(N, sol, vars, params, grid)
  @. vars.qh = sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  invtransform!(vars.u, vars.uh, params)
  @. vars.u += params.U                    # add the imposed zonal flow U

  uQx, uQxh = vars.q, vars.uh              # use vars.q and vars.uh as scratch variables
  @. uQx = vars.u * params.Qx              # (U+u)*∂Q/∂x
  fwdtransform!(uQxh, uQx, params)
  @. N = - uQxh                            # -\hat{(U+u)*∂Q/∂x}

  invtransform!(vars.v, vars.vh, params)

  vQy, vQyh = vars.q, vars.vh              # use vars.q and vars.vh as scratch variables
  @. vQy = vars.v * params.Qy              # v*∂Q/∂y
  fwdtransform!(vQyh, vQy, params)
  @. N -= vQyh                             # -\hat{v*∂Q/∂y}

  invtransform!(vars.q, vars.qh, params)

  uq , vq  = vars.u , vars.v               # use vars.u and vars.v as scratch variables
  uqh, vqh = vars.uh, vars.vh              # use vars.uh and vars.vh as scratch variables
  @. uq *= vars.q                          # (U+u)*q
  @. vq *= vars.q                          # v*q

  fwdtransform!(uqh, uq, params)
  fwdtransform!(vqh, vq, params)

  @. N -= im * grid.kr * uqh + im * grid.l * vqh    # -\hat{∂[(U+u)q]/∂x} - \hat{∂[vq]/∂y}

  return nothing
end


"""
    calcN_linearadvection!(N, sol, vars, params, grid)

Compute the advection term of the linearized equations and store it in `N`:

```math
N_j = - \\widehat{U_j ∂_x Q_j} - \\widehat{U_j ∂_x q_j}
 + \\widehat{(∂_y ψ_j)(∂_x Q_j)} - \\widehat{(∂_x ψ_j)(∂_y Q_j)} .
```
"""
function calcN_linearadvection!(N, sol, vars, params, grid)
  @. vars.qh = sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  invtransform!(vars.u, vars.uh, params)

  @. vars.u += params.U                    # add the imposed zonal flow U
  uQx, uQxh = vars.q, vars.uh              # use vars.q and vars.uh as scratch variables
  @. uQx  = vars.u * params.Qx             # (U+u)*∂Q/∂x
  fwdtransform!(uQxh, uQx, params)
  @. N = - uQxh                            # -\hat{(U+u)*∂Q/∂x}

  invtransform!(vars.v, vars.vh, params)

  vQy, vQyh = vars.q, vars.vh              # use vars.q and vars.vh as scratch variables

  @. vQy = vars.v * params.Qy              # v*∂Q/∂y
  fwdtransform!(vQyh, vQy, params)
  @. N -= vQyh                             # -\hat{v*∂Q/∂y}

  invtransform!(vars.q, vars.qh, params)

  @. vars.u  = params.U
  Uq , Uqh  = vars.u , vars.uh             # use vars.u and vars.uh as scratch variables
  @. Uq *= vars.q                          # U*q

  fwdtransform!(Uqh, Uq, params)

  @. N -= im * grid.kr * Uqh               # -\hat{∂[U*q]/∂x}

  return nothing
end


"""
    addforcing!(N, sol, t, clock, vars, params, grid)

When the problem includes forcing, calculate the forcing term ``F̂`` for each layer and add
it to the nonlinear term ``N``.
"""
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
    updatevars!(vars, params, grid, sol)
    updatevars!(prob)

Update all problem variables using `sol`.
"""
function updatevars!(vars, params, grid, sol)
  dealias!(sol, grid)

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  invtransform!(vars.q, deepcopy(vars.qh), params)
  invtransform!(vars.ψ, deepcopy(vars.ψh), params)
  invtransform!(vars.u, deepcopy(vars.uh), params)
  invtransform!(vars.v, deepcopy(vars.vh), params)

  return nothing
end

updatevars!(prob) = updatevars!(prob.vars, prob.params, prob.grid, prob.sol)


"""
    set_q!(sol, params, vars, grid, q)
    set_q!(prob, q)

Set the solution `prob.sol` as the transform of `q` and update variables.
"""
function set_q!(sol, params, vars, grid, q)
  A = typeof(vars.q)
  fwdtransform!(vars.qh, A(q), params)
  @. vars.qh[1, 1, :] = 0
  @. sol = vars.qh
  updatevars!(vars, params, grid, sol)

  return nothing
end

function set_q!(sol, params::SingleLayerParams, vars, grid, q::AbstractArray{T, 2}) where T
  A = typeof(vars.q[:, :, 1])
  q_3D = vars.q
  @views q_3D[:, :, 1] = A(q)
  set_q!(sol, params, vars, grid, q_3D)

  return nothing
end

set_q!(prob, q) = set_q!(prob.sol, prob.params, prob.vars, prob.grid, q)


"""
    set_ψ!(params, vars, grid, sol, ψ)
    set_ψ!(prob, ψ)

Set the solution `prob.sol` to the transform `qh` that corresponds to streamfunction `ψ`
and update variables.
"""
function set_ψ!(sol, params, vars, grid, ψ)
  A = typeof(vars.q)
  fwdtransform!(vars.ψh, A(ψ), params)
  pvfromstreamfunction!(vars.qh, vars.ψh, params, grid)
  invtransform!(vars.q, vars.qh, params)

  set_q!(sol, params, vars, grid, vars.q)

  return nothing
end

function set_ψ!(sol, params::SingleLayerParams, vars, grid, ψ::AbstractArray{T, 2}) where T
  A = typeof(vars.ψ[:, :, 1])
  ψ_3D = vars.ψ
  @views ψ_3D[:, :, 1] = A(ψ)

  set_ψ!(sol, params, vars, grid, ψ_3D)

  return nothing
end

set_ψ!(prob, ψ) = set_ψ!(prob.sol, prob.params, prob.vars, prob.grid, ψ)


"""
    energies(vars, params, grid, sol)
    energies(prob)

Return the kinetic energy of each fluid layer KE``_1, ...,`` KE``_{n}``, and the
potential energy of each fluid interface PE``_{3/2}, ...,`` PE``_{n-1/2}``, where ``n``
is the number of layers in the fluid. (When ``n=1``, only the kinetic energy is returned.)

The kinetic energy at the ``j``-th fluid layer is

```math
𝖪𝖤_j = \\frac{H_j}{H} \\int \\frac1{2} |{\\bf ∇} ψ_j|^2 \\frac{𝖽x 𝖽y}{L_x L_y} = \\frac1{2} \\frac{H_j}{H} \\sum_{𝐤} |𝐤|² |ψ̂_j|², \\ j = 1, ..., n ,
```

while the potential energy that corresponds to the interface ``j+1/2`` (i.e., the interface
between the ``j``-th and ``(j+1)``-th fluid layer) is

```math
𝖯𝖤_{j+1/2} = \\int \\frac1{2} \\frac{f₀^2}{g'_{j+1/2} H} (ψ_j - ψ_{j+1})^2 \\frac{𝖽x 𝖽y}{L_x L_y} = \\frac1{2} \\frac{f₀^2}{g'_{j+1/2} H} \\sum_{𝐤} |ψ̂_j - ψ̂_{j+1}|², \\ j = 1, ..., n-1 .
```
"""
function energies(vars, params, grid, sol)
  nlayers = numberoflayers(params)
  KE, PE = zeros(nlayers), zeros(nlayers-1)

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  abs²∇𝐮h = vars.uh        # use vars.uh as scratch variable
  @. abs²∇𝐮h = grid.Krsq * abs2(vars.ψh)

  V = grid.Lx * grid.Ly * sum(params.H)  # total volume of the fluid

  for j = 1:nlayers
    view(KE, j) .= 1 / (2 * V) * parsevalsum(view(abs²∇𝐮h, :, :, j), grid) * params.H[j]
  end

  for j = 1:nlayers-1
    view(PE, j) .= 1 / (2 * V) * params.f₀^2 ./ params.g′[j] .* parsevalsum(abs2.(view(vars.ψh, :, :, j) .- view(vars.ψh, :, :, j+1)), grid)
  end

  return KE, PE
end

function energies(vars, params::SingleLayerParams, grid, sol)
  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  abs²∇𝐮h = vars.uh        # use vars.uh as scratch variable
  @. abs²∇𝐮h = grid.Krsq * abs2(vars.ψh)

  return 1 / (2 * grid.Lx * grid.Ly) * parsevalsum(abs²∇𝐮h, grid)
end

energies(prob) = energies(prob.vars, prob.params, prob.grid, prob.sol)

"""
    fluxes(vars, params, grid, sol)
    fluxes(prob)

Return the lateral eddy fluxes within each fluid layer, lateralfluxes``_1,...,``lateralfluxes``_n``
and also the vertical eddy fluxes at each fluid interface,
verticalfluxes``_{3/2},...,``verticalfluxes``_{n-1/2}``, where ``n`` is the total number of layers in the fluid.
(For a single fluid layer, i.e., when ``n=1``, only the lateral fluxes are returned.)

The lateral eddy fluxes within the ``j``-th fluid layer are

```math
\\textrm{lateralfluxes}_j = \\frac{H_j}{H} \\int U_j v_j ∂_y u_j
\\frac{𝖽x 𝖽y}{L_x L_y} , \\  j = 1, ..., n ,
```

while the vertical eddy fluxes at the ``j+1/2``-th fluid interface (i.e., interface between
the ``j``-th and ``(j+1)``-th fluid layer) are

```math
\\textrm{verticalfluxes}_{j+1/2} = \\int \\frac{f₀²}{g'_{j+1/2} H} (U_j - U_{j+1}) \\,
v_{j+1} ψ_{j} \\frac{𝖽x 𝖽y}{L_x L_y} , \\ j = 1, ..., n-1.
```
"""
function fluxes(vars, params, grid, sol)

  nlayers = numberoflayers(params)

  lateralfluxes, verticalfluxes = zeros(nlayers), zeros(nlayers-1)

  updatevars!(vars, params, grid, sol)

  ∂u∂yh = vars.uh           # use vars.uh as scratch variable
  ∂u∂y  = vars.u            # use vars.u  as scratch variable

  @. ∂u∂yh = im * grid.l * vars.uh
  invtransform!(∂u∂y, ∂u∂yh, params)

  V = grid.Lx * grid.Ly * sum(params.H)  # total volume of the fluid

  lateralfluxes = params.H .* (sum(@. params.U * vars.v * ∂u∂y; dims=(1, 2)))[1, 1, :]
  lateralfluxes *= grid.dx * grid.dy / V

  for j = 1:nlayers-1
    Uⱼ, Uⱼ₊₁ = view(params.U, :, :, j), view(params.U, :, :, j+1)
    ψⱼ = view(vars.ψ, :, :, j)
    vⱼ₊₁ = view(vars.v, :, :, j+1)
    verticalfluxes[j] = sum(@. params.f₀^2 / params.g′[j] * (Uⱼ - Uⱼ₊₁) * vⱼ₊₁ * ψⱼ)
  end
  verticalfluxes *= grid.dx * grid.dy / V

  return lateralfluxes, verticalfluxes
end

function fluxes(vars, params::SingleLayerParams, grid, sol)
  updatevars!(vars, params, grid, sol)

  ∂u∂yh = vars.uh           # use vars.uh as scratch variable
  ∂u∂y  = vars.u            # use vars.u  as scratch variable

  @. ∂u∂yh = im * grid.l * vars.uh
  invtransform!(∂u∂y, ∂u∂yh, params)

  lateralfluxes = sum(@. params.U * vars.v * ∂u∂y)
  lateralfluxes *= grid.dx * grid.dy / (grid.Lx * grid.Ly)

  return lateralfluxes
end

fluxes(prob) = fluxes(prob.vars, prob.params, prob.grid, prob.sol)

end # module
