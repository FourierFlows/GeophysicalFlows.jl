module ThreeDQG

export
  Problem

  

using
  FFTW,
  CUDA,
  LinearAlgebra,
  StaticArrays,
  #Reexport,
  #DocStringExtensions,
  KernelAbstractions

#@reexport using FourierFlows
using FourierFlows

using FourierFlows: parsevalsum, parsevalsum2, superzeros, plan_flows_rfft, CPU, GPU
using KernelAbstractions.Extras.LoopInfo: @unroll

#nothingfunction(args...) = nothing





"""
    SpecMatrix(x, w, w′)



"""
function SpecMatrix(x::AbstractVector, w::AbstractVector, w′::AbstractVector)

    N = length(x)
    D = x' .- x

    D₁ = D
    D₁[D .== 0] .= 1

    A = ones(N, 1) * (w' .* prod(D₁; dims = 1))

    M = A' ./ (A .* D')
    M[1:(N+1):N^2] .= sum(1 ./ D₁; dims = 1)' .- 1 + (w′ ./ w)

    return M

end

"""
    SpecMatrix(x)



"""
SpecMatrix(x) = SpecMatrix(x, ones(length(x)), zeros(length(x)))


"""
    GridChebyshev(N, L)

Creates a grid of Chebyshev points of the second kind and a spectral collocation
differentiation matrix.

# Arguments:
 - `N`: number of gridpoints
 - `L`: vector of endpoints, ``x ∈ [L₁, L₂]``

"""
function GridChebyshev(N::Int, L::Vector)

    x = (L[1] + L[2]) / 2 .- (L[2] - L[1]) / 2 * cos.((0:(N-1))*π/(N-1))

    M = SpecMatrix(x)

    return x, M

end

"""
    GridLaguerre(N, L)


"""
function GridLaguerre(N::Int, L::Vector)

    J = diagm(0 => 1:2:2*N-3) - diagm(1 => 1:N-2) - diagm(-1 => 1:N-2)
    p = sort(eigvals(J))
    if L[2] > L[1]
        x = L[1] .+ [0; p / p[end] * (L[2] - L[1])]
        w = [1; exp.(-p/2)]                          # weight function w(x)
        w′ = -p[end] / (L[2] - L[1]) / 2 * w         # dw/dx = dw/dp * dp/dx
    else
        x = [0; p]
        w = [1; exp.(-p/2)]
        w′ = -1 / 2 * w
    end

    M = SpecMatrix(x, w, w′)

    return x, M

end







"""
define problem ...

"""
function Problem(nz::Int,                                     # number of fluid layers
                     dev = CPU();
                      nx = 128,
                      ny = nx,
                      Lx = 2π,
                      Ly = Lx,
                      f₀ = 1.0,                               # Coriolis parameter
                       β = 0.0,                               # y-gradient of Coriolis parameter
                      N² = 1.0,                               
                       H = 1.0,                               # layer depth
                       U = 0.0,
                       ν = 0,
                      nν = 1,
                      dt = 0.01,
                 stepper = "RK4",
        aliased_fraction = 1/3,
                       T = Float64)

  grid = TwoDGrid(dev; nx, Lx, ny, Ly, aliased_fraction, T)

  params = Params(nz, f₀, β, N², H, U, ν, nν, grid)

  vars = Vars(grid, params)

  equation = Equation(params, grid)

  FourierFlows.Problem(equation, stepper, dt, grid, vars, params)
end

"""
parameter structure ...

"""
struct Params{T, Aphys3D, Atrans4D, Trfft, Aphys2D, Aphys1D} <: AbstractParams
        nz :: Int
        f₀ :: T
         β :: Aphys3D
        N² :: T
         H :: T
         U :: T
         ν :: T
        nν :: Int
         S :: Atrans4D
       S⁻¹ :: Atrans4D
  rfftplan :: Trfft
        Dz :: Aphys2D
         z :: Aphys1D
end


function Params(nz::Int, f₀, β, N², H, U, ν, nν, grid::TwoDGrid; effort=FFTW.MEASURE)
  dev = grid.device
  T = eltype(grid)
  A = device_array(dev)

   ny, nx = grid.ny , grid.nx
  nkr, nl = grid.nkr, grid.nl

  rfftplanlayered = plan_flows_rfft(A{T, 3}(undef, grid.nx, grid.ny, nz), [1, 2]; flags=effort)

  if isinf(H)
    z, Dz = GridLaguerre(nz, [0, 0])
  else
    z, Dz = GridChebyshev(nz, [-H, 0])
  end

  typeofSkl = SArray{Tuple{nz, nz}, T, 2, nz^2} # StaticArrays of type T and dims = (nz, nz)

  S = Array{typeofSkl, 2}(undef, (nkr, nl))    # Array of StaticArrays
  calcS!(S, Dz, nz, grid, f₀, N², H)

  S⁻¹ = Array{typeofSkl, 2}(undef, (nkr, nl))  # Array of StaticArrays
  calcS⁻¹!(S⁻¹, Dz, nz, grid, f₀, N², H)

  β = T(β) * ones((1, 1, nz))
  β[:, :, 1] .= 0
  if !isinf(H)
    β[:, :, nz] .= 0
  end

  S, S⁻¹, β = A(S), A(S⁻¹), A(β) # convert to appropriate ArrayType

  return Params(nz, T(f₀), β, T(N²), T(H), T(U), T(ν), nν, S, S⁻¹, rfftplanlayered, Dz, z)
end

numberoflayers(params) = params.nz

# ---------
# Equations
# ---------

"""
    Equation(params, grid)

Return the equation for a multi-layer quasi-geostrophic problem with `params` and `grid`.
The linear operator ``L`` includes only (hyper)-viscosity and is computed via
`hyperviscosity(params, grid)`.

The nonlinear term is computed via [`calcN!`](@ref GeophysicalFlows.MultiLayerQG.calcN!).
"""
function Equation(params, grid)
  dev = grid.device
  T = eltype(grid)

  L = device_array(dev){Complex{T}}(undef, (grid.nkr, grid.nl, numberoflayers(params)))
  @. L = - params.ν * grid.Krsq^params.nν - im * params.U * grid.kr
  @views @. L[1, 1, :] = 0

  return FourierFlows.Equation(L, calcN!, grid)
end




# ----
# Vars
# ----

"""
    struct Vars{Aphys, Atrans} <: AbstractVars

The variables for multi-layer QG problem.

"""
struct Vars{Aphys, Atrans} <: AbstractVars
        q :: Aphys
        ψ :: Aphys
        u :: Aphys
        v :: Aphys
       qh :: Atrans
       ψh :: Atrans
       uh :: Atrans
       vh :: Atrans
end

"""
    Vars(grid, params)

...

"""
function Vars(grid, params)
  Dev = typeof(grid.device)
  T = eltype(grid)
  nz = numberoflayers(params)

  @devzeros Dev T (grid.nx, grid.ny, nz) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, nz) qh ψh uh vh

  return Vars(q, ψ, u, v, qh, ψh, uh, vh)
end


fwdtransform!(varh, var, params::AbstractParams) = mul!(varh, params.rfftplan, var)
invtransform!(var, varh, params::AbstractParams) = ldiv!(var, params.rfftplan, varh)




"""
    pv_streamfunction_kernel!(y, M, x, ::Val{N}) where N

Kernel for the PV to streamfunction conversion steps. The kernel performs the
matrix multiplication

```math
y = M x
```

for every wavenumber, where ``y`` and ``x`` are column-vectors of length `nz`.
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
  S, nz = params.S, params.nz
  kernel!(qh, S, ψh, Val(nz))

  # Ensure that no other operations occur until the kernel has finished
  KernelAbstractions.synchronize(backend)

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
  S⁻¹, nz = params.S⁻¹, params.nz
  kernel!(ψh, S⁻¹, qh, Val(nz))

  # Ensure that no other operations occur until the kernel has finished
  KernelAbstractions.synchronize(backend)

  return nothing
end




"""
    calcS!(S, Dz, nz, grid, f₀, N², H)

Construct the array ``𝕊``, which consists of `nlayer` x `nlayer` static arrays ``𝕊_𝐤`` that
relate the ``q̂_j``'s and ``ψ̂_j``'s for every wavenumber: ``q̂_𝐤 = 𝕊_𝐤 ψ̂_𝐤``.
"""
function calcS!(S, Dz, nz, grid, f₀, N², H)

  L = f₀^2 / N² * Dz^2
  L[1, :] = Dz[1, :]

  if isinf(H)
    I₀ = diagm([0; ones(nz - 1)])
  else
    L[nz, :] = Dz[nz, :]
    I₀ = diagm([0; ones(nz - 2); 0])
  end

  for n=1:grid.nl, m=1:grid.nkr
    k² = CUDA.@allowscalar grid.Krsq[m, n]
    Skl = SMatrix{nz, nz}(- k² * I₀ + L)
    S[m, n] = Skl
  end

  return nothing
end

"""
    calcS⁻¹!(S, Dz, nz, grid, f₀, N², H)

Construct the array ``𝕊⁻¹``, which consists of `nlayer` x `nlayer` static arrays ``(𝕊_𝐤)⁻¹``
that relate the ``q̂_j``'s and ``ψ̂_j``'s for every wavenumber: ``ψ̂_𝐤 = (𝕊_𝐤)⁻¹ q̂_𝐤``.
"""
function calcS⁻¹!(S⁻¹, Dz, nz, grid, f₀, N², H)

  L = f₀^2 / N² * Dz^2
  L[1, :] = Dz[1, :]

  if isinf(H)
    I₀ = diagm([0; ones(nz - 1)])
  else
    L[nz, :] = Dz[nz, :]
    I₀ = diagm([0; ones(nz - 2); 0])
  end

  for n=1:grid.nl, m=1:grid.nkr
    k² = CUDA.@allowscalar grid.Krsq[m, n] == 0 ? 1 : grid.Krsq[m, n]
    Skl = - k² * I₀ + L
    S⁻¹[m, n] = SMatrix{nz, nz}(I / Skl)
  end

  T = eltype(grid)
  S⁻¹[1, 1] = SMatrix{nz, nz}(zeros(T, (nz, nz)))

  return nothing
end


# -------
# Solvers
# -------

"""
    calcN!(N, sol, t, clock, vars, params, grid)

Compute the advection term:

...
"""
function calcN!(N, sol, t, clock, vars, params, grid)
  nz = numberoflayers(params)

  dealias!(sol, grid)

  calcN_advection!(N, sol, vars, params, grid)

  return nothing
end


"""
    calcN_advection!(N, sol, vars, params, grid)

Compute the advection term and store it in `N`:

...

"""
function calcN_advection!(N, sol, vars, params, grid)
  @. vars.qh = sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  @. N = - vars.vh * params.β                          # -β*\hat{v}

  invtransform!(vars.q, vars.qh, params)
  invtransform!(vars.u, vars.uh, params)
  invtransform!(vars.v, vars.vh, params)

  uq , vq  = vars.u , vars.v               # use vars.u and vars.v as scratch variables
  uqh, vqh = vars.uh, vars.vh              # use vars.uh and vars.vh as scratch variables
  @. uq *= vars.q                          # u*q
  @. vq *= vars.q                          # v*q

  fwdtransform!(uqh, uq, params)
  fwdtransform!(vqh, vq, params)

  @. N -= im * grid.kr * uqh + im * grid.l * vqh    # -\hat{∂[(U+u)q]/∂x} - \hat{∂[vq]/∂y}

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

set_ψ!(prob, ψ) = set_ψ!(prob.sol, prob.params, prob.vars, prob.grid, ψ)

nothing



end # module