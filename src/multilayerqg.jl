module MultilayerQG

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
  LinearAlgebra,
  Reexport

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FFTW: rfft, irfft
using FourierFlows: getfieldspecs, varsexpression, parsevalsum, parsevalsum2, superzeros

nothingfunction(args...) = nothing

"""
    Problem(; parameters...)

Construct a multi-layer QG problem.
"""
function Problem(dev = CPU(), T = Float64;
    # Numerical parameters
          nx = 128,
          Lx = 2π,
          ny = nx,
          Ly = Lx,
          dt = 0.01,
    # Physical parameters
     nlayers = 2,                             # number of fluid layers
          f0 = 1.0,                           # Coriolis parameter
           β = 0.0,                           # y-gradient of Coriolis parameter
           g = 1.0,                           # gravitational constant
           U = zeros(dev, T, (ny, nlayers)),  # imposed zonal flow U(y) in each layer
           H = [0.2, 0.8],                    # rest fluid height of each layer
           ρ = [4.0, 5.0],                    # density of each layer
         eta = zeros(dev, T, (nx, ny)),       # topographic PV
    # Bottom Drag and/or (hyper)-viscosity
           μ = 0.0,
           ν = 0.0,
          nν = 1,
    # Timestepper and eqn options
     stepper = "RK4",
      calcFq = nothingfunction,
      linear = false)

   grid = TwoDGrid(dev, nx, Lx, ny, Ly; T=T)
   params = Params(nlayers, g, f0, β, ρ, H, U, eta, μ, ν, nν, grid, calcFq=calcFq, dev=dev)
   vars = calcFq == nothingfunction ? Vars(grid, params) : ForcedVars(grid, params)
   eqn = linear ? LinearEquation(dev, params, grid) : Equation(dev, params, grid)

  FourierFlows.Problem(eqn, stepper, dt, grid, vars, params, dev)
end

abstract type BarotropicParams <: AbstractParams end

struct Params{T, TFFT, Aphys3D, Aphys2D, Aphys1D, Atrans4D} <: AbstractParams
  # prescribed params
   nlayers :: Int            # Number of fluid layers
         g :: T              # Gravitational constant
        f0 :: T              # Constant planetary vorticity
         β :: T              # Planetary vorticity y-gradient
         ρ :: Aphys3D    # Array with density of each fluid layer
         H :: Aphys3D    # Array with rest height of each fluid layer
         U :: Aphys3D    # Array with imposed constant zonal flow U(y) in each fluid layer
       eta :: Aphys2D    # Array containing topographic PV
         μ :: T              # Linear bottom drag
         ν :: T              # Viscosity coefficient
        nν :: Int            # Hyperviscous order (nν=1 is plain old viscosity)
   calcFq! :: Function       # Function that calculates the forcing on QGPV q

  # derived params
        g′ :: Aphys1D    # Array with the reduced gravity constants for each fluid interface
        Qx :: Aphys3D    # Array containing x-gradient of PV due to eta in each fluid layer
        Qy :: Aphys3D    # Array containing y-gradient of PV due to β, U, and eta in each fluid layer
         S :: Atrans4D    # Array containing coeffients for getting PV from  streamfunction
      invS :: Atrans4D    # Array containing coeffients for inverting PV to streamfunction
  rfftplan :: FFTW.rFFTWPlan{TFFT, -1, false, 3}  # rfft plan for FFTs
end

struct SingleLayerParams{T, TFFT, Aphys3D, Aphys2D} <: BarotropicParams
  # prescribed params
         β :: T              # Planetary vorticity y-gradient
         U :: Aphys3D    # Imposed constant zonal flow U(y)
       eta :: Aphys2D    # Array containing topographic PV
         μ :: T              # Linear bottom drag
         ν :: T              # Viscosity coefficient
        nν :: Int            # Hyperviscous order (nν=1 is plain old viscosity)
   calcFq! :: Function       # Function that calculates the forcing on QGPV q

  # derived params
        Qx :: Aphys3D    # Array containing x-gradient of PV due to eta
        Qy :: Aphys3D    # Array containing meridional PV gradient due to β, U, and eta
  rfftplan :: FFTW.rFFTWPlan{TFFT, -1, false, 3}  # rfft plan for FFTs
end

function convert_U_to_U3D(nlayers, grid, U::AbstractArray{TU, 1}) where TU
  if length(U) == nlayers
    U = reshape(U, (1, nlayers))
    U = repeat(U, outer=(grid.ny, 1))
  else
    U = reshape(U, (grid.ny, 1))
  end
  
  U_3D = reshape(U, (1, grid.ny, nlayers))
  return U_3D
end

function convert_U_to_U3D(nlayers, grid, U::AbstractArray{TU, 2}) where TU
  U_3D = reshape(U, (1, grid.ny, nlayers))
  return U_3D
end

function convert_U_to_U3D(nlayers, grid, U::Number)
  U_3D =  reshape(repeat([U], outer=(grid.ny, 1)), (1, grid.ny, nlayers))
  return U_3D
end


function Params(nlayers, g, f0, β, ρ, H, U, eta, μ, ν, nν, grid; calcFq=nothingfunction, effort=FFTW.MEASURE, dev::Device=CPU()) where TU
  
  T = eltype(grid)

   ny, nx = grid.ny , grid.nx
  nkr, nl = grid.nkr, grid.nl
   kr, l  = grid.kr , grid.l
  
    U = convert_U_to_U3D(nlayers, grid, U)
    
  Uyy = repeat(irfft( -l.^2 .* rfft(U, [1, 2]),  1, [1, 2]), outer=(1, 1, 1))
  Uyy = repeat(Uyy, outer=(nx, 1, 1))

  etah = rfft(eta)
  etax = irfft(im * kr .* etah, nx)
  etay = irfft(im * l  .* etah, nx)

  Qx = zeros(dev, T, (nx, ny, nlayers))
  @views @. Qx[:, :, nlayers] += etax

  Qy = zeros(dev, T, (nx, ny, nlayers))
  Qy = @. T(β) - Uyy  # T(β) is needed to ensure that Qy remains same type as U
  @views @. Qy[:, :, nlayers] += etay

  rfftplanlayered = plan_rfft(ArrayType(dev){T, 3}(undef, grid.nx, grid.ny, nlayers), [1, 2]; flags=effort)
  
  if nlayers==1
    return SingleLayerParams(β, U, eta, μ, ν, nν, calcFq, Qx, Qy, rfftplanlayered)
  
  else # if nlayers≥2
    
    ρ = reshape(ρ, (1,  1, nlayers))
    H = reshape(H, (1,  1, nlayers))

    g′ = g * (ρ[2:nlayers] - ρ[1:nlayers-1]) ./ ρ[2:nlayers] # reduced gravity at each interface;

    Fm = @. f0^2 / ( g′ * H[2:nlayers  ] )
    Fp = @. f0^2 / ( g′ * H[1:nlayers-1] )

    @views @. Qy[:, :, 1] -= Fp[1] * ( U[:, :, 2] - U[:, :, 1] )
    for j = 2:nlayers-1
      @. Qy[:, :, j] -= Fp[j] * ( U[:, :, j+1] - U[:, :, j] ) + Fm[j-1] * ( U[:, :, j-1] - U[:, :, j] )
    end
    @views @. Qy[:, :, nlayers] -= Fm[nlayers-1] * ( U[:, :, nlayers-1] - U[:, :, nlayers] )

    S = ArrayType(dev){T}(undef, (nkr, nl, nlayers, nlayers))
    calcS!(S, Fp, Fm, grid)

    invS = ArrayType(dev){T}(undef, (nkr, nl, nlayers, nlayers))
    calcinvS!(invS, Fp, Fm, grid)

    return Params(nlayers, g, f0, β, T.(ρ), T.(H), U, eta, μ, ν, nν, calcFq, T.(g′), Qx, Qy, S, invS, rfftplanlayered)  
  end
end

numberoflayers(params) = params.nlayers
numberoflayers(::SingleLayerParams) = 1

# ---------
# Equations
# ---------

function hyperdissipation(dev, params, grid)
  T = eltype(grid)
  L = ArrayType(dev){T}(undef, (grid.nkr, grid.nl, numberoflayers(params)))
  @. L = - params.ν * grid.Krsq^params.nν
  @views @. L[1, 1, :] = 0
  return L
end

function LinearEquation(dev, params, grid)
  L = hyperdissipation(dev, params, grid)
  return FourierFlows.Equation(L, calcNlinear!, grid)
end

function Equation(dev, params, grid)
  L = hyperdissipation(dev, params, grid)
  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

const physicalvars = [:q, :ψ, :u, :v]
const fouriervars = [ Symbol(var, :h) for var in physicalvars ]
const forcedfouriervars = cat(fouriervars, [:Fqh], dims=1)

varspecs = cat(
  getfieldspecs(physicalvars, :(Array{T, 3})),
  getfieldspecs(fouriervars, :(Array{Complex{T}, 3})),
  dims=1)

forcedvarspecs = cat(
  getfieldspecs(physicalvars, :(Array{T, 3})),
  getfieldspecs(forcedfouriervars, :(Array{Complex{T}, 3})),
  dims=1)

# Construct Vars types
eval(varsexpression(:Vars, physicalvars, fouriervars))
eval(varsexpression(:ForcedVars, physicalvars, forcedfouriervars))

"""
    Vars(grid, params)

Returns the vars for unforced multi-layer QG problem with `grid` and `params`.
"""
function Vars(grid, params)
  T = eltype(grid)
  nx , ny = grid.nx , grid.ny
  nkr, nl = grid.nkr, grid.nl
  nlayers = numberoflayers(params)
  
  @zeros T (nx, ny, nlayers) q ψ u v
  @zeros Complex{T} (nkr, nl, nlayers) qh ψh uh vh
  
  return Vars(q, ψ, u, v, qh, ψh, uh, vh)
end

"""
    ForcedVars(grid, params)

Returns the vars for forced multi-layer QG problem with `grid` and `params`.
"""
function ForcedVars(grid, params)
  T = eltype(grid)
  vars = Vars(grid, params)
  nlayers = numberoflayers(params)
  Fqh = zeros(Complex{T}, (grid.nkr, grid.nl, nlayers))
  
  return ForcedVars(getfield.(Ref(vars), fieldnames(typeof(vars)))..., Fqh)
end

fwdtransform!(varh, var, params::AbstractParams) = mul!(varh, params.rfftplan, var)
invtransform!(var, varh, params::AbstractParams) = ldiv!(var, params.rfftplan, varh)

function streamfunctionfrompv!(ψh, qh, params, grid)
  for j=1:grid.nl, i=1:grid.nkr
    @views ψh[i, j, :] .= params.invS[i, j, :, :] * qh[i, j, :]
  end
end

function pvfromstreamfunction!(qh, ψh, params, grid)
  for j=1:grid.nl, i=1:grid.nkr
    @views qh[i, j, :] .= params.S[i, j, :, :] * ψh[i, j, :]
  end
end

function streamfunctionfrompv!(ψh, qh, params::SingleLayerParams, grid)
  @. ψh = -grid.invKrsq * qh
end

function pvfromstreamfunction!(qh, ψh, params::SingleLayerParams, grid)
  @. qh = -grid.Krsq * ψh
end

"""
    calcS!(S, Fp, Fm, grid)

Constructs the stretching matrix S that connects q and ψ: q_{k,l} = S * ψ_{k,l}.
"""
function calcS!(S, Fp, Fm, grid)
  F = Matrix(Tridiagonal(Fm, -([Fp; 0] + [0; Fm]), Fp))
  for n=1:grid.nl, m=1:grid.nkr
     k² = grid.Krsq[m, n]
    Skl = - k²*I + F
    @views S[m, n, :, :] .= Skl
  end
  return nothing
end

"""
    calcinvS!(S, Fp, Fm, grid)

Constructs the inverse of the stretching matrix S that connects q and ψ:
ψ_{k,l} = invS * q_{k,l}.
"""
function calcinvS!(invS, Fp, Fm, grid)
  F = Matrix(Tridiagonal(Fm, -([Fp; 0] + [0; Fm]), Fp))
  for n=1:grid.nl, m=1:grid.nkr
    k² = grid.Krsq[m, n]
    if k² == 0
      k² = 1
    end
    Skl = - k²*I + F
    @views invS[m, n, :, :] .= I / Skl
  end
  @views invS[1, 1, :, :] .= 0
  return nothing
end


# -------
# Solvers
# -------

function calcN!(N, sol, t, clock, vars, params, grid)
  nlayers = numberoflayers(params)
  calcN_advection!(N, sol, vars, params, grid)
  @views @. N[:, :, nlayers] += params.μ * grid.Krsq * vars.ψh[:, :, nlayers]   # bottom linear drag
  addforcing!(N, sol, t, clock, vars, params, grid)
  return nothing
end

function calcNlinear!(N, sol, t, clock, vars, params, grid)
  nlayers = numberoflayers(params)
  calcN_linearadvection!(N, sol, vars, params, grid)
  @views @. N[:, :, nlayers] += params.μ * grid.Krsq * vars.ψh[:, :, nlayers]   # bottom linear drag
  addforcing!(N, sol, t, clock, vars, params, grid)
  return nothing
end

"""
    calcN_advection!(N, sol, vars, params, grid)

Calculates the advection term.
"""
function calcN_advection!(N, sol, vars, params, grid)
  @. vars.qh = sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  invtransform!(vars.u, vars.uh, params)
  @. vars.u += params.U                    # add the imposed zonal flow U
  @. vars.q  = vars.u * params.Qx
  fwdtransform!(vars.uh, vars.q, params)
  @. N = -vars.uh                          # -(U+u)*∂Q/∂x

  invtransform!(vars.v, vars.vh, params)
  @. vars.q = vars.v * params.Qy
  fwdtransform!(vars.vh, vars.q, params)
  @. N -= vars.vh                          # -v*∂Q/∂y

  invtransform!(vars.q, vars.qh, params)

  @. vars.u *= vars.q                      # u*q
  @. vars.v *= vars.q                      # v*q

  fwdtransform!(vars.uh, vars.u, params)
  fwdtransform!(vars.vh, vars.v, params)

  @. N -= im * grid.kr * vars.uh + im * grid.l * vars.vh    # -∂[(U+u)q]/∂x-∂[vq]/∂y

  return nothing
end


"""
    calcN_linearadvection!(N, sol, vars, params, grid)

Calculates the advection term of the linearized equations.
"""
function calcN_linearadvection!(N, sol, vars, params, grid)
  @. vars.qh = sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  invtransform!(vars.u, vars.uh, params)
  @. vars.u += params.U                    # add the imposed zonal flow U
  @. vars.q  = vars.u * params.Qx
  fwdtransform!(vars.uh, vars.q, params)
  @. N = -vars.uh                          # -(U+u)*∂Q/∂x

  invtransform!(vars.v, vars.vh, params)
  @. vars.q = vars.v * params.Qy
  fwdtransform!(vars.vh, vars.q, params)
  @. N -= vars.vh                          # -v*∂Q/∂y

  invtransform!(vars.q, vars.qh, params)
  @. vars.u  = params.U
  @. vars.u *= vars.q                      # u*q

  fwdtransform!(vars.uh, vars.u, params)

  @. N -= im * grid.kr * vars.uh           # -∂[U*q]/∂x

  return nothing
end

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
    updatevars!(params, vars, grid, sol)
    updatevars!(prob)

Update all problem variables using `sol`.
"""
function updatevars!(params, vars, grid, sol)
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

updatevars!(prob) = updatevars!(prob.params, prob.vars, prob.grid, prob.sol)


"""
    set_q!(params, vars, grid, sol, q)
    set_q!(prob)

Set the solution `prob.sol` as the transform of `q` and updates variables.
"""
function set_q!(params, vars, grid, sol, q)
  fwdtransform!(vars.qh, q, params)
  @. vars.qh[1, 1, :] = 0
  @. sol = vars.qh
  updatevars!(params, vars, grid, sol)
  return nothing
end

function set_q!(params::SingleLayerParams, vars, grid, sol, q::Array{T, 2}) where T
  q_3D = reshape(q, (grid.nx, grid.ny, 1))
  set_q!(params, vars, grid, sol, q_3D)
  return nothing
end

set_q!(prob, q) = set_q!(prob.params, prob.vars, prob.grid, prob.sol, q)


"""
    set_ψ!(params, vars, grid, sol, ψ)
    set_ψ!(prob)

Set the solution `prob.sol` to correspond to the transform of streamfunction `ψ` and
updates variables.
"""
function set_ψ!(params, vars, grid, sol, ψ)
  fwdtransform!(vars.ψh, ψ, params)
  pvfromstreamfunction!(vars.qh, vars.ψh, params, grid)
  invtransform!(vars.q, vars.qh, params)
  set_q!(params, vars, grid, sol, vars.q)
  return nothing
end

function set_ψ!(params::SingleLayerParams, vars, grid, sol, ψ::Array{T, 2}) where T 
  ψ_3D = reshape(ψ, (grid.nx, grid.ny, 1))
  set_ψ!(params, vars, grid, sol, ψ_3D)
  return nothing  
end

set_ψ!(prob, ψ) = set_ψ!(prob.params, prob.vars, prob.grid, prob.sol, ψ)


"""
    energies(prob)

Returns the kinetic energy of each fluid layer KE_1, ..., KE_nlayers, and the
potential energy of each fluid interface PE_{3/2}, ..., PE_{nlayers-1/2}.
(When `nlayers=1` only kinetic energy is returned.)
"""
function energies(vars, params, grid, sol)
  nlayers = numberoflayers(params)
  KE, PE = zeros(nlayers), zeros(nlayers-1)

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  @. vars.uh = grid.Krsq * abs2(vars.ψh)
  for j=1:nlayers
    KE[j] = 1/(2*grid.Lx*grid.Ly)*parsevalsum(vars.uh[:, :, j], grid)*params.H[j]/sum(params.H)
  end

  for j=1:nlayers-1
    PE[j] = 1/(2*grid.Lx*grid.Ly)*params.f0^2/params.g′[j]*parsevalsum(abs2.(vars.ψh[:, :, j+1].-vars.ψh[:, :, j]), grid)
  end

  return KE, PE
end

function energies(vars, params::SingleLayerParams, grid, sol)
  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  
  KE = 1/(2*grid.Lx*grid.Ly)*parsevalsum(grid.Krsq .* abs2.(vars.ψh), grid)
  
  return KE
end

energies(prob) = energies(prob.vars, prob.params, prob.grid, prob.sol)

"""
    fluxes(prob)

Returns the lateral eddy fluxes within each fluid layer
lateralfluxes_1, ..., lateralfluxes_nlayers and also the vertical eddy fluxes for
each fluid interface verticalfluxes_{3/2}, ..., verticalfluxes_{nlayers-1/2}.
(When `nlayers=1` only the lateral fluxes are returned.)
"""
function fluxes(vars, params, grid, sol)
  nlayers = numberoflayers(params)
  
  lateralfluxes, verticalfluxes = zeros(nlayers), zeros(nlayers-1)

  updatevars!(params, vars, grid, sol)

  @. vars.uh = im * grid.l * vars.uh      # ∂u/∂y
  invtransform!(vars.u, vars.uh, params)

  lateralfluxes = (sum( @. params.H * params.U * vars.v * vars.u; dims=(1, 2) ))[1, 1, :]
  lateralfluxes *= grid.dx * grid.dy / (grid.Lx * grid.Ly * sum(params.H))

  for j=1:nlayers-1
    verticalfluxes[j] = sum( @views @. params.f0^2 / params.g′[j] * (params.U[: ,:, j] - params.U[:, :, j+1]) * vars.v[:, :, j+1] * vars.ψ[:, :, j] ; dims=(1,2) )[1]
    verticalfluxes[j] *= grid.dx * grid.dy / (grid.Lx * grid.Ly * sum(params.H))
  end

  return lateralfluxes, verticalfluxes
end

function fluxes(vars, params::SingleLayerParams, grid, sol)
  updatevars!(params, vars, grid, sol)

  @. vars.uh = im * grid.l * vars.uh
  invtransform!(vars.u, vars.uh, params)

  lateralfluxes = (sum( @. params.U * vars.v * vars.u; dims=(1, 2) ))[1, 1, :]
  lateralfluxes *= grid.dx * grid.dy / (grid.Lx * grid.Ly)

  return lateralfluxes
end

fluxes(prob) = fluxes(prob.vars, prob.params, prob.grid, prob.sol)

end # module
