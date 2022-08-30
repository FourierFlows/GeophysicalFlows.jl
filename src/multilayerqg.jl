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
  DocStringExtensions

@reexport using FourierFlows

using FourierFlows: parsevalsum, parsevalsum2, superzeros, plan_flows_rfft

nothingfunction(args...) = nothing

"""
    Problem(nlayers::Int,
                dev = CPU();
                 nx = 128,
                 ny = nx,
                 Lx = 2π,
                 Ly = Lx,
                 f₀ = 1.0,
                  β = 0.0,
                  g = 1.0,
                  U = zeros(nlayers),
                  H = 1/nlayers * ones(nlayers),
                  ρ = Array{Float64}(1:nlayers),
                eta = nothing,
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

Construct a multi-layer quasi-geostrophic `problem` with `nlayers` fluid layers on device `dev`.

Keyword arguments
=================
  - `nlayers`: (required) Number of fluid layers.
  - `dev`: (required) `CPU()` or `GPU()`; computer architecture used to time-step `problem`.
  - `nx`: Number of grid points in ``x``-domain.
  - `ny`: Number of grid points in ``y``-domain.
  - `Lx`: Extent of the ``x``-domain.
  - `Ly`: Extent of the ``y``-domain.
  - `f₀`: Constant planetary vorticity.
  - `β`: Planetary vorticity ``y``-gradient.
  - `g`: Gravitational acceleration constant.
  - `U`: The imposed constant zonal flow ``U(y)`` in each fluid layer.
  - `H`: Rest height of each fluid layer.
  - `ρ`: Density of each fluid layer.
  - `eta`: Topographic potential vorticity.
  - `μ`: Linear bottom drag coefficient.
  - `ν`: Small-scale (hyper)-viscosity coefficient.
  - `nν`: (Hyper)-viscosity order, `nν```≥ 1``.
  - `dt`: Time-step.
  - `stepper`: Time-stepping method.
  - `calcF`: Function that calculates the Fourier transform of the forcing, ``F̂``.
  - `stochastic`: `true` or `false`; boolean denoting whether `calcF` is temporally stochastic.
  - `aliased_fraction`: the fraction of high-wavenumbers that are zero-ed out by `dealias!()`.
  - `T`: `Float32` or `Float64`; floating point type used for `problem` data.
"""
function Problem(nlayers::Int,                        # number of fluid layers
                     dev = CPU();
              # Numerical parameters
                      nx = 128,
                      ny = nx,
                      Lx = 2π,
                      Ly = Lx,
              # Physical parameters
                      f₀ = 1.0,                       # Coriolis parameter
                       β = 0.0,                       # y-gradient of Coriolis parameter
                       g = 1.0,                       # gravitational constant
                       U = zeros(nlayers),            # imposed zonal flow U(y) in each layer
                       H = 1/nlayers * ones(nlayers), # rest fluid height of each layer
                       ρ = Array{Float64}(1:nlayers), # density of each layer
                     eta = nothing,                   # topographic PV
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

  if dev == GPU() && nlayers > 2
    @warn """MultiLayerQG module is not optimized on the GPU yet for configurations with
    3 fluid layers or more!
    
    See issues on Github at https://github.com/FourierFlows/GeophysicalFlows.jl/issues/112
    and https://github.com/FourierFlows/GeophysicalFlows.jl/issues/267.
    
    To use MultiLayerQG with 3 fluid layers or more we suggest, for now, to restrict running
    on CPU."""
  end
   
  # topographic PV
  eta === nothing && (eta = zeros(dev, T, (nx, ny)))
   
  grid = TwoDGrid(dev, nx, Lx, ny, Ly; aliased_fraction=aliased_fraction, T=T)
   
  params = Params(nlayers, g, f₀, β, ρ, H, U, eta, μ, ν, nν, grid, calcFq=calcFq, dev=dev)   
   
  vars = calcFq == nothingfunction ? DecayingVars(dev, grid, params) : (stochastic ? StochasticForcedVars(dev, grid, params) : ForcedVars(dev, grid, params))
   
  equation = linear ? LinearEquation(dev, params, grid) : Equation(dev, params, grid)

  FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
end

"""
    struct Params{T, Aphys3D, Aphys2D, Aphys1D, Atrans4D, Trfft} <: AbstractParams

The parameters for the MultiLayerQG problem.

$(TYPEDFIELDS)
"""
struct Params{T, Aphys3D, Aphys2D, Aphys1D, Atrans4D, Trfft} <: AbstractParams
  # prescribed params
    "number of fluid layers"
   nlayers :: Int
    "gravitational constant"
         g :: T
    "constant planetary vorticity"
        f₀ :: T
    "planetary vorticity ``y``-gradient"
         β :: T       
    "array with density of each fluid layer"
         ρ :: Aphys3D
    "array with rest height of each fluid layer"
         H :: Aphys3D 
    "array with imposed constant zonal flow ``U(y)`` in each fluid layer"
         U :: Aphys3D 
    "array containing the topographic PV"
       eta :: Aphys2D 
    "linear bottom drag coefficient"
         μ :: T
    "small-scale (hyper)-viscosity coefficient"
         ν :: T
    "(hyper)-viscosity order, `nν```≥ 1``"
        nν :: Int
    "function that calculates the Fourier transform of the forcing, ``F̂``"
   calcFq! :: Function

  # derived params
    "array with the reduced gravity constants for each fluid interface"
        g′ :: Aphys1D
    "array containing ``x``-gradient of PV due to eta in each fluid layer"
        Qx :: Aphys3D
    "array containing ``y``-gradient of PV due to ``β``, ``U``, and topographic PV in each fluid layer"
        Qy :: Aphys3D
    "array containing coeffients for getting PV from streamfunction"
         S :: Atrans4D
    "array containing coeffients for inverting PV to streamfunction"
       S⁻¹ :: Atrans4D
    "rfft plan for FFTs"
  rfftplan :: Trfft
end

"""
    struct SingleLayerParams{T, Aphys3D, Aphys2D, Trfft} <: AbstractParams

The parameters for the SingleLayerQG problem.

$(TYPEDFIELDS)
"""
struct SingleLayerParams{T, Aphys3D, Aphys2D, Trfft} <: AbstractParams
  # prescribed params
    "planetary vorticity y-gradient"
         β :: T
    "array with imposed constant zonal flow U(y)"
         U :: Aphys3D
    "array containing topographic PV"
       eta :: Aphys2D
    "linear drag coefficient"
         μ :: T
    "small-scale (hyper)-viscosity coefficient"
         ν :: T
    "(hyper)-viscosity order, `nν```≥ 1``"
        nν :: Int
    "function that calculates the Fourier transform of the forcing, ``F̂``"
   calcFq! :: Function

  # derived params
    "array containing x-gradient of PV due to eta"
        Qx :: Aphys3D
    "array containing y-gradient of PV due to β, U, and eta"
        Qy :: Aphys3D
    "rfft plan for FFTs"
  rfftplan :: Trfft
end

"""
    TwoLayerParams{T, Aphys3D, Aphys2D, Trfft} <: AbstractParams

The parameters for the TwoLayerQG problem.

$(TYPEDFIELDS)
"""
struct TwoLayerParams{T, Aphys3D, Aphys2D, Trfft} <: AbstractParams
  # prescribed params
    "gravitational constant"
         g :: T
    "constant planetary vorticity"
        f₀ :: T
    "planetary vorticity y-gradient"
         β :: T
    "array with density of each fluid layer"
         ρ :: Aphys3D
    "tuple with rest height of each fluid layer"
         H :: Tuple
   "array with imposed constant zonal flow U(y) in each fluid layer"
         U :: Aphys3D
    "array containing topographic PV"
       eta :: Aphys2D
    "linear bottom drag coefficient"
         μ :: T
    "small-scale (hyper)-viscosity coefficient"
         ν :: T
    "(hyper)-viscosity order, `nν```≥ 1``"
        nν :: Int
    "function that calculates the Fourier transform of the forcing, ``F̂``"
   calcFq! :: Function

  # derived params
    "the reduced gravity constants for the fluid interface"
        g′ :: T
    "array containing x-gradient of PV due to eta in each fluid layer"
        Qx :: Aphys3D
    "array containing y-gradient of PV due to β, U, and eta in each fluid layer"
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
  A = ArrayType(dev)
  U_3D = reshape(repeat([T(U)], outer=(grid.ny, 1)), (1, grid.ny, nlayers))
  return A(U_3D)
end

function Params(nlayers, g, f₀, β, ρ, H, U, eta, μ, ν, nν, grid; calcFq=nothingfunction, effort=FFTW.MEASURE, dev::Device=CPU()) where TU
  T = eltype(grid)
  A = ArrayType(dev)

   ny, nx = grid.ny , grid.nx
  nkr, nl = grid.nkr, grid.nl
   kr, l  = grid.kr , grid.l
  
    U = convert_U_to_U3D(dev, nlayers, grid, U)

  Uyy = real.(ifft(-l.^2 .* fft(U)))
  Uyy = CUDA.@allowscalar repeat(Uyy, outer=(nx, 1, 1))

  etah = rfft(A(eta))
  etax = irfft(im * kr .* etah, nx)
  etay = irfft(im * l  .* etah, nx)

  Qx = zeros(dev, T, (nx, ny, nlayers))
  @views @. Qx[:, :, nlayers] += etax

  Qy = zeros(dev, T, (nx, ny, nlayers))
  Qy = T(β) .- Uyy  # T(β) is needed to ensure that Qy remains same type as U
  @views @. Qy[:, :, nlayers] += etay
  
  rfftplanlayered = plan_flows_rfft(A{T, 3}(undef, grid.nx, grid.ny, nlayers), [1, 2]; flags=effort)
  
  if nlayers==1
    return SingleLayerParams(T(β), U, eta, T(μ), T(ν), nν, calcFq, Qx, Qy, rfftplanlayered)
  
  else # if nlayers≥2
    
    ρ = reshape(T.(ρ), (1,  1, nlayers))
    H = reshape(T.(H), (1,  1, nlayers))

    g′ = T(g) * (ρ[2:nlayers] - ρ[1:nlayers-1]) ./ ρ[2:nlayers] # reduced gravity at each interface

    Fm = @. T(f₀^2 / (g′ * H[2:nlayers]))
    Fp = @. T(f₀^2 / (g′ * H[1:nlayers-1]))

    typeofSkl = SArray{Tuple{nlayers, nlayers}, T, 2, nlayers^2} # StaticArrays of type T and dims = (nlayers x nlayers)
    
    S = Array{typeofSkl, 2}(undef, (nkr, nl))
    calcS!(S, Fp, Fm, nlayers, grid)

    S⁻¹ = Array{typeofSkl, 2}(undef, (nkr, nl))
    calcS⁻¹!(S⁻¹, Fp, Fm, nlayers, grid)
    
    S, S⁻¹, Fp, Fm  = A(S), A(S⁻¹), A(Fp), A(Fm)     # convert to appropriate ArrayType

    CUDA.@allowscalar @views Qy[:, :, 1] = @. Qy[:, :, 1] - Fp[1] * (U[:, :, 2] - U[:, :, 1])
    for j = 2:nlayers-1
      CUDA.@allowscalar @views Qy[:, :, j] = @. Qy[:, :, j] - Fp[j] * (U[:, :, j+1] - U[:, :, j]) + Fm[j-1] * (U[:, :, j-1] - U[:, :, j])
    end
    CUDA.@allowscalar @views Qy[:, :, nlayers] = @. Qy[:, :, nlayers] - Fm[nlayers-1] * (U[:, :, nlayers-1] - U[:, :, nlayers])

    if nlayers==2
      return TwoLayerParams(T(g), T(f₀), T(β), A(ρ), (T(H[1]), T(H[2])), U, eta, T(μ), T(ν), nν, calcFq, T(g′[1]), Qx, Qy, rfftplanlayered)
    else # if nlayers>2
      return Params(nlayers, T(g), T(f₀), T(β), A(ρ), A(H), U, eta, T(μ), T(ν), nν, calcFq, A(g′), Qx, Qy, S, S⁻¹, rfftplanlayered)
    end
  end
end

numberoflayers(params) = params.nlayers
numberoflayers(::SingleLayerParams) = 1
numberoflayers(::TwoLayerParams) = 2

# ---------
# Equations
# ---------

"""
    hyperviscosity(dev, params, grid)

Return the linear operator `L` that corresponds to (hyper)-viscosity of order ``n_ν`` with 
coefficient ``ν`` for ``n`` fluid layers.
```math
L_j = - ν |𝐤|^{2 n_ν}, \\ j = 1, ...,n .
```
"""
function hyperviscosity(dev, params, grid)
  T = eltype(grid)
  L = ArrayType(dev){T}(undef, (grid.nkr, grid.nl, numberoflayers(params)))
  @. L = - params.ν * grid.Krsq^params.nν
  @views @. L[1, 1, :] = 0
  
  return L
end

"""
    LinearEquation(dev, params, grid)

Return the equation for a multi-layer quasi-geostrophic problem with `params` and `grid`. 
The linear opeartor ``L`` includes only (hyper)-viscosity and is computed via 
`hyperviscosity(dev, params, grid)`.

The nonlinear term is computed via function `calcNlinear!`.
"""
function LinearEquation(dev, params, grid)
  L = hyperviscosity(dev, params, grid)
  
  return FourierFlows.Equation(L, calcNlinear!, grid)
end
 
"""
    Equation(dev, params, grid)

Return the equation for a multi-layer quasi-geostrophic problem with `params` and `grid`. 
The linear opeartor ``L`` includes only (hyper)-viscosity and is computed via 
`hyperviscosity(dev, params, grid)`.

The nonlinear term is computed via function `calcN!`.
"""
function Equation(dev, params, grid)
  L = hyperviscosity(dev, params, grid)
  
  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

"""
    struct Vars{Aphys, Atrans, F, P} <: AbstractVars

The variables for MultiLayer QG.

$(FIELDS)
"""
struct Vars{Aphys, Atrans, F, P} <: AbstractVars
    "relative vorticity + vortex stretching"
        q :: Aphys
    "streamfunction"
        ψ :: Aphys
    "x-component of velocity"
        u :: Aphys
    "y-component of velocity"
        v :: Aphys
    "Fourier transform of relative vorticity + vortex stretching"
       qh :: Atrans
    "Fourier transform of streamfunction"
       ψh :: Atrans
    "Fourier transform of x-component of velocity"
       uh :: Atrans
    "Fourier transform of y-component of velocity"
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
    DecayingVars(dev, grid, params)

Return the variables for an unforced multi-layer QG problem with `grid` and `params`.
"""
function DecayingVars(dev::Dev, grid, params) where Dev
  T = eltype(grid)
  nlayers = numberoflayers(params)
  
  @devzeros Dev T (grid.nx, grid.ny, nlayers) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, nlayers) qh ψh uh vh
  
  return Vars(q, ψ, u, v, qh, ψh, uh, vh, nothing, nothing)
end

"""
    ForcedVars(dev, grid, params)

Return the variables for a forced multi-layer QG problem with `grid` and `params`.
"""
function ForcedVars(dev::Dev, grid, params) where Dev
  T = eltype(grid)
  nlayers = numberoflayers(params)
  
  @devzeros Dev T (grid.nx, grid.ny, nlayers) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, nlayers) qh ψh uh vh Fqh
  
  return Vars(q, ψ, u, v, qh, ψh, uh, vh, Fqh, nothing)
end

"""
    StochasticForcedVars(dev, rid, params)

Return the variables for a forced multi-layer QG problem with `grid` and `params`.
"""
function StochasticForcedVars(dev::Dev, grid, params) where Dev
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
    pvfromstreamfunction!(qh, ψh, params, grid)

Obtain the Fourier transform of the PV from the streamfunction `ψh` in each layer using 
`qh = params.S * ψh`.
"""
function pvfromstreamfunction!(qh, ψh, params, grid)
  for j=1:grid.nl, i=1:grid.nkr
    CUDA.@allowscalar @views qh[i, j, :] .= params.S[i, j] * ψh[i, j, :]    
  end
  
  return nothing
end

"""
    pvfromstreamfunction!(qh, ψh, params::SingleLayerParams, grid)

Obtain the Fourier transform of the PV from the streamfunction `ψh` for the special
case of a single fluid layer configuration. In this case, ``q̂ = - k² ψ̂``.
"""
function pvfromstreamfunction!(qh, ψh, params::SingleLayerParams, grid)
  @. qh = -grid.Krsq * ψh
  
  return nothing
end

"""
    pvfromstreamfunction!(qh, ψh, params::TwoLayerParams, grid)

Obtain the Fourier transform of the PV from the streamfunction `ψh` for the special
case of a two fluid layer configuration. In this case we have,

```math
q̂₁ = - k² ψ̂₁ + f₀² / (g′ H₁) * (ψ̂₂ - ψ̂₁) ,
```

```math
q̂₂ = - k² ψ̂₂ + f₀² / (g′ H₂) * (ψ̂₁ - ψ̂₂) .
```

(Here, the PV-streamfunction relationship is hard-coded to avoid scalar operations
on the GPU.)
"""
function pvfromstreamfunction!(qh, ψh, params::TwoLayerParams, grid)
  f₀, g′, H₁, H₂ = params.f₀, params.g′, params.H[1], params.H[2]
  
  ψ1h, ψ2h = view(ψh, :, :, 1), view(ψh, :, :, 2)

  @views @. qh[:, :, 1] = - grid.Krsq * ψ1h + f₀^2 / (g′ * H₁) * (ψ2h - ψ1h)
  @views @. qh[:, :, 2] = - grid.Krsq * ψ2h + f₀^2 / (g′ * H₂) * (ψ1h - ψ2h)
  
  return nothing
end

"""
    streamfunctionfrompv!(ψh, qh, params, grid)

Invert the PV to obtain the Fourier transform of the streamfunction `ψh` in each layer from
`qh` using `ψh = params.S⁻¹ qh`.
"""
function streamfunctionfrompv!(ψh, qh, params, grid)
  for j=1:grid.nl, i=1:grid.nkr
    CUDA.@allowscalar @views ψh[i, j, :] .= params.S⁻¹[i, j] * qh[i, j, :]
  end
  
  return nothing
end

"""
    streamfunctionfrompv!(ψh, qh, params::SingleLayerParams, grid)

Invert the PV to obtain the Fourier transform of the streamfunction `ψh` for the special
case of a single fluid layer configuration. In this case, ``ψ̂ = - k⁻² q̂``.
"""
function streamfunctionfrompv!(ψh, qh, params::SingleLayerParams, grid)
  @. ψh = -grid.invKrsq * qh
  
  return nothing
end

"""
    streamfunctionfrompv!(ψh, qh, params::TwoLayerParams, grid)

Invert the PV to obtain the Fourier transform of the streamfunction `ψh` for the special
case of a two fluid layer configuration. In this case we have,

```math
ψ̂₁ = - [k⁻² q̂₁ + (f₀² / g′) (q̂₁ / H₂ + q̂₂ / H₁)] / Δ ,
```

```math
ψ̂₂ = - [k⁻² q̂₂ + (f₀² / g′) (q̂₁ / H₂ + q̂₂ / H₁)] / Δ ,
```

where ``Δ = k² [k² + f₀² (H₁ + H₂) / (g′ H₁ H₂)]``.
  
(Here, the PV-streamfunction relationship is hard-coded to avoid scalar operations
on the GPU.)
"""
function streamfunctionfrompv!(ψh, qh, params::TwoLayerParams, grid)
  f₀, g′, H₁, H₂ = params.f₀, params.g′, params.H[1], params.H[2]
  
  q1h, q2h = view(qh, :, :, 1), view(qh, :, :, 2)

  @views @. ψh[:, :, 1] = - grid.Krsq * q1h - f₀^2 / g′ * (q1h / H₂ + q2h / H₁)
  @views @. ψh[:, :, 2] = - grid.Krsq * q2h - f₀^2 / g′ * (q1h / H₂ + q2h / H₁)
  
  for j in 1:2
    @views @. ψh[:, :, j] *= grid.invKrsq / (grid.Krsq + f₀^2 / g′ * (H₁ + H₂) / (H₁ * H₂))
  end

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

Compute the advection term and stores it in `N`:

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
  @. uQx  = vars.u * params.Qx             # (U+u)*∂Q/∂x
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

Compute the advection term of the linearized equations and stores it in `N`:

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
  
  for j = 1:nlayers
    views(KE, j) = 1 / (2 * grid.Lx * grid.Ly) * parsevalsum(abs²∇𝐮h[:, :, j], grid) * params.H[j] / sum(params.H)
  end

  for j = 1:nlayers-1
    views(PE, j) = 1 / (2 * grid.Lx * grid.Ly * sum(params.H)) * params.f₀^2 / params.g′[j] * parsevalsum(abs2.(vars.ψh[:, :, j] .- vars.ψh[:, :, j+1]), grid)
  end

  return KE, PE
end

function energies(vars, params::TwoLayerParams, grid, sol)
  nlayers = numberoflayers(params)
  KE, PE = zeros(nlayers), zeros(nlayers-1)
  
  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  
  abs²∇𝐮h = vars.uh        # use vars.uh as scratch variable
  @. abs²∇𝐮h = grid.Krsq * abs2(vars.ψh)

  ψ1h, ψ2h = view(vars.ψh, :, :, 1), view(vars.ψh, :, :, 2)

  for j = 1:nlayers
    views(KE, j) = 1 / (2 * grid.Lx * grid.Ly) * parsevalsum(abs²∇𝐮h[:, :, j], grid) * params.H[j] / sum(params.H)
  end

  PE = 1 / (2 * grid.Lx * grid.Ly * sum(params.H)) * params.f₀^2 / params.g′ * parsevalsum(abs2.(ψ1h .- ψ2h), grid)
  
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
(When ``n=1``, only the lateral fluxes are returned.)

The lateral eddy fluxes within the ``j``-th fluid layer are

```math
\\textrm{lateralfluxes}_j = \\frac{H_j}{H} \\int U_j v_j ∂_y u_j 
\\frac{𝖽x 𝖽y}{L_x L_y} , \\  j = 1, ..., n ,
```

while the vertical eddy fluxes at the ``j+1/2``-th fluid interface  (i.e., interface between 
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

  lateralfluxes = (sum(@. params.H * params.U * vars.v * ∂u∂y; dims=(1, 2)))[1, 1, :]
  lateralfluxes *= grid.dx * grid.dy / (grid.Lx * grid.Ly * sum(params.H))

  for j = 1:nlayers-1
    @views verticalfluxes[j] = sum(@views @. params.f₀^2 / params.g′[j] * (params.U[: ,:, j] - params.U[:, :, j+1]) * vars.v[:, :, j+1] * vars.ψ[:, :, j]; dims=(1, 2))[1]
    @views verticalfluxes[j] *= grid.dx * grid.dy / (grid.Lx * grid.Ly * sum(params.H))
  end

  return lateralfluxes, verticalfluxes
end

function fluxes(vars, params::TwoLayerParams, grid, sol)
  nlayers = numberoflayers(params)
  
  lateralfluxes, verticalfluxes = zeros(nlayers), zeros(nlayers-1)

  updatevars!(vars, params, grid, sol)

  ∂u∂yh = vars.uh           # use vars.uh as scratch variable
  ∂u∂y  = vars.u            # use vars.u  as scratch variable

  @. ∂u∂yh = im * grid.l * vars.uh
  invtransform!(∂u∂y, ∂u∂yh, params)

  lateralfluxes = (sum(@. params.U * vars.v * ∂u∂y; dims=(1, 2)))[1, 1, :]
  @. lateralfluxes *= params.H
  lateralfluxes *= grid.dx * grid.dy / (grid.Lx * grid.Ly * sum(params.H))

  U₁, U₂ = view(params.U, :, :, 1), view(params.U, :, :, 2)
  ψ₁ = view(vars.ψ, :, :, 1)
  v₂ = view(vars.v, :, :, 2)
  
  verticalfluxes = sum(@views @. params.f₀^2 / params.g′ * (U₁ - U₂) * v₂ * ψ₁; dims=(1, 2))
  verticalfluxes *= grid.dx * grid.dy / (grid.Lx * grid.Ly * sum(params.H))

  return lateralfluxes, verticalfluxes
end

function fluxes(vars, params::SingleLayerParams, grid, sol)
  updatevars!(vars, params, grid, sol)

  ∂u∂yh = vars.uh           # use vars.uh as scratch variable
  ∂u∂y  = vars.u            # use vars.u  as scratch variable

  @. ∂u∂yh = im * grid.l * vars.uh
  invtransform!(∂u∂y, ∂u∂yh, params)

  lateralfluxes = (sum(@. params.U * vars.v * ∂u∂y; dims=(1, 2)))[1, 1, :]
  lateralfluxes *= grid.dx * grid.dy / (grid.Lx * grid.Ly)

  return lateralfluxes
end

fluxes(prob) = fluxes(prob.vars, prob.params, prob.grid, prob.sol)

end # module
