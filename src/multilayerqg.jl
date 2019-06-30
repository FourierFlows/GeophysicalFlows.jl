# TODO make sure that setting nlayers=1 works

module MultilayerQG

export
  fwdtransform!,
  invtransform!,
  streamfunctionfrompv!,
  pvfromstreamfunction!,
  updatevars!,

  set_q!,
  set_psi!,
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
function Problem(;
    # Numerical parameters
          nx = 128,
          Lx = 2π,
          ny = nx,
          Ly = Lx,
          dt = 0.01,
    # Physical parameters
     nlayers = 2,                       # number of fluid layers
          f0 = 1.0,                     # Coriolis parameter
        beta = 0.0,                     # y-gradient of Coriolis parameter
           g = 1.0,                     # gravitational constant
           U = zeros(ny, nlayers),      # imposed zonal flow U(y) in each layer
           H = [0.2, 0.8],              # rest fluid height of each layer
         rho = [4.0, 5.0],              # density of each layer
         eta = zeros(nx, ny),           # topographic PV
    # Bottom Drag and/or (hyper)-viscosity
          mu = 0.0,
          nu = 0.0,
         nnu = 1,
    # Timestepper and eqn options
     stepper = "RK4",
      calcFq = nothingfunction,
      linear = false,
           T = Float64)

   grid = TwoDGrid(nx, Lx, ny, Ly; T=T)
   params = Params(nlayers, g, f0, beta, rho, H, U, eta, mu, nu, nnu, grid, calcFq=calcFq)
   vars = calcFq == nothingfunction ? Vars(grid, params) : ForcedVars(grid, params)
   eqn = linear ? LinearEquation(params, grid) : Equation(params, grid)

  FourierFlows.Problem(eqn, stepper, dt, grid, vars, params)
end

abstract type BarotropicParams <: AbstractParams end

struct Params{T} <: AbstractParams
  # prescribed params
  nlayers::Int               # Number of fluid layers
  g::T                       # Gravitational constant
  f0::T                      # Constant planetary vorticity
  beta::T                    # Planetary vorticity y-gradient
  rho::Array{T,3}            # Array with density of each fluid layer
  H::Array{T,3}              # Array with rest height of each fluid layer
  U::Array{T,3}              # Array with imposed constant zonal flow U(y) in each fluid layer
  eta::Array{T,2}            # Array containing topographic PV
  mu::T                      # Linear bottom drag
  nu::T                      # Viscosity coefficient
  nnu::Int                   # Hyperviscous order (nnu=1 is plain old viscosity)
  calcFq!::Function          # Function that calculates the forcing on QGPV q

  # derived params
  greduced::Array{T,1}       # Array with the reduced gravity constants for each fluid interface
  Qx::Array{T,3}             # Array containing x-gradient of PV due to U and eta in each fluid layer
  Qy::Array{T,3}             # Array containing y-gradient of PV due to beta, U, and eta in each fluid layer
  S::Array{T,4}              # Array containing coeffients for getting PV from  streamfunction
  invS::Array{T,4}           # Array containing coeffients for inverting PV to streamfunction
  rfftplan::FFTW.rFFTWPlan{Float64,-1,false,3}  # rfft plan for FFTs
end

struct SingleLayerParams{T} <: BarotropicParams
  # prescribed params
  g::T                       # Gravitational constant
  f0::T                      # Constant planetary vorticity
  beta::T                    # Planetary vorticity y-gradient
  rho::T                     # Fluid layer density
  H::T                       # Fluid layer rest height
  U::Array{T,1}              # Imposed constant zonal flow U(y)
  eta::Array{T,2}            # Array containing topographic PV
  mu::T                      # Linear bottom drag
  nu::T                      # Viscosity coefficient
  nnu::Int                   # Hyperviscous order (nnu=1 is plain old viscosity)
  calcFq!::Function          # Function that calculates the forcing on QGPV q

  # derived params
  Qx::Array{T,2}             # Array containing zonal PV gradient due to beta, U, and eta in each fluid layer
  Qy::Array{T,2}             # Array containing meridional PV gradient due to beta, U, and eta in each fluid layer
  rfftplan::FFTW.rFFTWPlan{Float64,-1,false,2}  # rfft plan for FFTs
end

function Params(nlayers, g, f0, beta, rho, H, U::Array{T,2}, eta, mu, nu, nnu, grid::AbstractGrid{T}; calcFq=nothingfunction, effort=FFTW.MEASURE) where T

  nkr, nl, ny, nx = grid.nkr, grid.nl,  grid.ny, grid.nx
  kr, l = grid.kr, grid.l

  U = reshape(U, (1, ny, nlayers))

  # greduced = g*(rho[2:nlayers]-rho[1:nlayers-1]) ./ rho[1:nlayers-1] # definition match PYQG
  greduced = g*(rho[2:nlayers]-rho[1:nlayers-1]) ./ rho[2:nlayers] # correct definition

  Fm = @. f0^2 / ( greduced*H[2:nlayers  ] )
  Fp = @. f0^2 / ( greduced*H[1:nlayers-1] )

  rho = reshape(rho, (1,  1, nlayers))
  H = reshape(H, (1,  1, nlayers))

  Uyy = repeat(irfft( -l.^2 .* rfft(U, [1, 2]),  1, [1, 2]), outer=(1, 1, 1))
  Uyy = repeat(Uyy, outer=(nx, 1, 1))

  etah = rfft(eta)
  etax = irfft(im*kr.*etah, nx)
  etay = irfft(im*l .*etah, nx)

  Qx = zeros(nx, ny, nlayers)
  @views @. Qx[:, :, nlayers] += etax

  Qy = zeros(nx, ny, nlayers)
  Qy[:, :, 1] = @. beta - Uyy[:, :, 1] - Fp[1]*( U[:, :, 2] - U[:, :, 1] )
  for j = 2:nlayers-1
    Qy[:, :, j] = @. beta - Uyy[:, :, j] - Fp[j]*( U[:, :, j+1] - U[:, :, j] ) - Fm[j-1]*( U[:, :, j-1] - U[:, :, j] )
  end
  @views Qy[:, :, nlayers] = @. beta - Uyy[:, :, nlayers] - Fm[nlayers-1]*( U[:, :, nlayers-1] - U[:, :, nlayers] )
  @views @. Qy[:, :, nlayers] += etay

  S = Array{Float64}(undef, (nkr, nl, nlayers, nlayers))
  calcS!(S, Fp, Fm, grid)

  invS = Array{Float64}(undef, (nkr, nl, nlayers, nlayers))
  calcinvS!(invS, Fp, Fm, grid)

  rfftplanlayered = plan_rfft(Array{T,3}(undef, grid.nx, grid.ny, nlayers), [1, 2]; flags=effort)

  if nlayers == 1
    SingleLayerParams{T}(g, f0, beta, rho, H, U, eta, mu, nu, nnu, calcFq, Qx, Qy, grid.rfftplan)
  else
    Params{T}(nlayers, g, f0, beta, rho, H, U, eta, mu, nu, nnu, calcFq, greduced, Qx, Qy, S, invS, rfftplanlayered)
  end
end

function Params(nlayers, g, f0, beta, rho, H, U::Array{T,1}, eta, mu, nu, nnu, grid::AbstractGrid{T}; calcFq=nothingfunction, effort=FFTW.MEASURE) where T
  U = reshape(U, (1, nlayers))
  U = repeat(U, outer=(grid.ny, 1))
  Params(nlayers, g, f0, beta, rho, H, U, eta, mu, nu, nnu, grid; calcFq=calcFq, effort=effort)
end


# ---------
# Equations
# ---------

function hyperdissipation(nu, nnu, Krsq, nkr, nl, nlayers, T)
  L = Array{Complex{T}}(undef, (nkr, nl, nlayers))
  @. L = -nu*Krsq^nnu
  @views @. L[1, 1, :] = 0
  L
end

function LinearEquation(p, g::AbstractGrid{T}) where T
  L = hyperdissipation(p.nu, p.nnu, g.Krsq, g.nkr, g.nl, p.nlayers, T)
  FourierFlows.Equation(L, calcNlinear!, g)
end

function Equation(p, g::AbstractGrid{T}) where T
  L = hyperdissipation(p.nu, p.nnu, g.Krsq, g.nkr, g.nl, p.nlayers, T)
  FourierFlows.Equation(L, calcN!, g)
end

function Equation(p::SingleLayerParams, g::AbstractGrid{T}) where T
  L = @. -p.mu - p.nu*g.Krsq^p.nnu + im*p.beta*g.kr*g.invKrsq
  L[1, 1] = 0
  FourierFlows.Equation(L, calcN!, g)
end


# ----
# Vars
# ----

abstract type BarotropicVars <: AbstractVars end

const physicalvars = [:q, :psi, :u, :v]
const fouriervars = [ Symbol(var, :h) for var in physicalvars ]
const forcedfouriervars = cat(fouriervars, [:Fqh], dims=1)

varspecs = cat(
  getfieldspecs(physicalvars, :(Array{T,3})),
  getfieldspecs(fouriervars, :(Array{Complex{T},3})),
  dims=1)

forcedvarspecs = cat(
  getfieldspecs(physicalvars, :(Array{T,3})),
  getfieldspecs(forcedfouriervars, :(Array{Complex{T},3})),
  dims=1)

singlelayervarsspecs = cat(
  getfieldspecs(physicalvars, :(Array{T,2})),
  getfieldspecs(fouriervars, :(Array{Complex{T},2})),
  dims=1)

# Construct Vars types
eval(varsexpression(:Vars, physicalvars, fouriervars))
eval(varsexpression(:ForcedVars, physicalvars, forcedfouriervars))
eval(varsexpression(:SingleLayerVars, singlelayervarsspecs; parent=:BarotropicVars, typeparams=:T))

"""
    Vars(g)

Returns the vars for unforced multi-layer QG problem with grid g.
"""
function Vars(g::AbstractGrid{T}, p) where T
  @zeros T (g.nx, g.ny, p.nlayers) q psi u v
  @zeros Complex{T} (g.nkr, g.nl, p.nlayers) qh psih uh vh
  Vars(q, psi, u, v, qh, psih, uh, vh)
end

"""
    ForcedVars(g)

Returns the vars for forced multi-layer QG problem with grid g.
"""
function ForcedVars(g::AbstractGrid{T}, p) where T
  v = Vars(g, p)
  Fqh = zeros(Complex{T}, (g.nkr, g.nl, p.nlayers))
  ForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., Fqh)
end

function SingleLayerVars(g::AbstractGrid{T}) where T
  @zeros T (g.nx, g.ny) q psi u v
  @zeros Complex{T} (g.nkr, g.nl) qh psih uh vh
  SingleLayerParams(q, psi, u, v, qh, psih, uh, vh)
end

fwdtransform!(varh, var, p::AbstractParams) = mul!(varh, p.rfftplan, var)
invtransform!(var, varh, p::AbstractParams) = ldiv!(var, p.rfftplan, varh)

function streamfunctionfrompv!(psih, qh, invS, g)
  for j=1:g.nl, i=1:g.nkr
    @views psih[i, j, :] .= invS[i, j, :, :]*qh[i, j, :]
  end
end

function pvfromstreamfunction!(qh, psih, S, g)
  for j=1:g.nl, i=1:g.nkr
    @views qh[i, j, :] .= S[i, j, :, :]*psih[i, j, :]
  end
end

"""
    calcS!(S, Fp, Fm, g)

Constructs the stretching matrix S that connects q and ψ: q_{k,l} = S * ψ_{k,l}.
"""
function calcS!(S, Fp, Fm, g)
  F = Matrix(Tridiagonal(Fm, -([Fp; 0] + [0; Fm]), Fp))
  for n=1:g.nl, m=1:g.nkr
    k2 = g.Krsq[m, n]
    Skl = -k2*I + F
    @views S[m, n, :, :] .= Skl
  end
  nothing
end

"""
    calcinvS!(S, Fp, Fm, g)

Constructs the inverse of the stretching matrix S that connects q and ψ:
ψ_{k,l} = invS * q_{k,l}.
"""
function calcinvS!(invS, Fp, Fm, g)
  F = Matrix(Tridiagonal(Fm, -([Fp; 0] + [0; Fm]), Fp))
  for n=1:g.nl, m=1:g.nkr
    k2 = g.Krsq[m, n]
    if k2 == 0
      k2 = 1
    end
    Skl = -k2*I + F
    @views invS[m, n, :, :] .= I/Skl
  end
  @views invS[1, 1, :, :] .= 0
  nothing
end


# -------
# Solvers
# -------

function calcN!(N, sol, t, cl, v, p, g)
  calcN_advection!(N, sol, v, p, g)
  @views @. N[:, :, p.nlayers] += p.mu*g.Krsq*v.psih[:, :, p.nlayers]   # bottom linear drag
  addforcing!(N, sol, t, cl, v, p, g)
  nothing
end

function calcNlinear!(N, sol, t, cl, v, p, g)
  calcN_linearadvection!(N, sol, v, p, g)
  @views @. N[:, :, p.nlayers] += p.mu*g.Krsq*v.psih[:, :, p.nlayers]   # bottom linear drag
  addforcing!(N, sol, t, cl, v, p, g)
  nothing
end

"""
    calcN_advection!(N, sol, v, p, g)

Calculates the advection term.
"""
function calcN_advection!(N, sol, v, p, g)
  @. v.qh = sol

  streamfunctionfrompv!(v.psih, v.qh, p.invS, g)

  @. v.uh = -im*g.l *v.psih
  @. v.vh =  im*g.kr*v.psih

  invtransform!(v.u, v.uh, p)
  @. v.u += p.U                         # add the imposed zonal flow U
  @. v.q = v.u*p.Qx
  fwdtransform!(v.uh, v.q, p)
  @. N = -v.uh                          # -(U+u)*∂Q/∂x

  invtransform!(v.v, v.vh, p)
  @. v.q = v.v*p.Qy
  fwdtransform!(v.vh, v.q, p)
  @. N -= v.vh                          # -v*∂Q/∂y

  invtransform!(v.q, v.qh, p)

  @. v.u *= v.q # u*q
  @. v.v *= v.q # v*q

  fwdtransform!(v.uh, v.u, p)
  fwdtransform!(v.vh, v.v, p)

  @. N -= im*g.kr*v.uh + im*g.l*v.vh    # -∂[(U+u)q]/∂x-∂[vq]/∂y

  nothing
end


"""
    calcN_linearadvection!(N, sol, v, p, g)

Calculates the advection term of the linearized equations.
"""
function calcN_linearadvection!(N, sol, v, p, g)
  @. v.qh = sol

  streamfunctionfrompv!(v.psih, v.qh, p.invS, g)

  @. v.uh = -im*g.l *v.psih
  @. v.vh =  im*g.kr*v.psih

  invtransform!(v.u, v.uh, p)
  @. v.u += p.U                        # add the imposed zonal flow U
  @. v.q = v.u*p.Qx
  fwdtransform!(v.uh, v.q, p)
  @. N = -v.uh                         # -(U+u)*∂Q/∂x

  invtransform!(v.v, v.vh, p)
  @. v.q = v.v*p.Qy
  fwdtransform!(v.vh, v.q, p)
  @. N -= v.vh                         # -v*∂Q/∂y

  invtransform!(v.q, v.qh, p)
  @. v.u = p.U
  @. v.u *= v.q # u*q

  fwdtransform!(v.uh, v.u, p)

  @. N -= im*g.kr*v.uh                 # -∂[U*q]/∂x

  nothing
end

addforcing!(N, sol, t, cl, v::Vars, p, g) = nothing

function addforcing!(N, sol, t, cl, v::ForcedVars, p, g)
  p.calcFq!(v.Fqh, sol, t, cl, v, p, g)
  @. N += v.Fqh
  nothing
end


# ----------------
# Helper functions
# ----------------

"""
    updatevars!(prob)

Update `prob.vars` using `prob.sol`.
"""
function updatevars!(prob)
  p, v, g, sol = prob.params, prob.vars, prob.grid, prob.sol

  @. v.qh = sol
  streamfunctionfrompv!(v.psih, v.qh, p.invS, g)
  @. v.uh = -im*g.l*v.psih
  @. v.vh =  im*g.kr*v.psih

  invtransform!(v.q, deepcopy(v.qh), p)
  invtransform!(v.psi, deepcopy(v.psih), p)
  invtransform!(v.u, deepcopy(v.uh), p)
  invtransform!(v.v, deepcopy(v.vh), p)
  nothing
end


"""
    set_q!(prob)

Set the solution `prob.sol` as the transform of `q` and updates variables.
"""
function set_q!(prob, q)
  p, v, sol = prob.params, prob.vars, prob.sol

  fwdtransform!(v.qh, q, p)
  @. v.qh[1, 1, :] = 0
  @. sol = v.qh

  updatevars!(prob)
  nothing
end


"""
    set_psi!(prob)

Set the solution `prob.sol` to correspond to a streamfunction `psi` and
updates variables.
"""
function set_psi!(prob, psi)
  p, v, g = prob.params, prob.vars, prob.grid

  fwdtransform!(v.psih, psi, p)
  pvfromstreamfunction!(v.qh, v.psih, p.S, g)
  invtransform!(v.q, v.qh, p)
  set_q!(prob, v.q)

  nothing
end


"""
    energies(prob)

Returns the kinetic energy of each fluid layer KE_1,...,KE_nlayers, and the
potential energy of each fluid interface PE_{3/2},...,PE_{nlayers-1/2}.
"""
function energies(prob)
  v, p, g, sol = prob.vars, prob.params, prob.grid, prob.sol
  KE, PE = zeros(p.nlayers), zeros(p.nlayers-1)

  @. v.qh = sol
  streamfunctionfrompv!(v.psih, v.qh, p.invS, g)

  @. v.uh = g.Krsq * abs2(v.psih)
  for j=1:p.nlayers
    KE[j] = 1/(2*g.Lx*g.Ly)*parsevalsum(v.uh[:, :, j], g)*p.H[j]/sum(p.H)
  end

  for j=1:p.nlayers-1
    PE[j] = 1/(2*g.Lx*g.Ly)*p.f0^2/p.greduced[j]*parsevalsum(abs2.(v.psih[:, :, j+1].-v.psih[:, :, j]), g)
  end

  KE, PE
end


"""
    fluxes(prob)

Returns the lateral eddy fluxes within each fluid layer
lateralfluxes_1,...,lateralfluxes_nlayers and also the vertical eddy fluxes for
each fluid interface verticalfluxes_{3/2},...,verticalfluxes_{nlayers-1/2}
"""
function fluxes(prob)
  v, p, g, sol = prob.vars, prob.params, prob.grid, prob.sol
  lateralfluxes, verticalfluxes = zeros(p.nlayers), zeros(p.nlayers-1)

  updatevars!(prob)

  @. v.uh = -im*g.l*v.uh
  invtransform!(v.u, v.uh, p)

  lateralfluxes = (sum(@. p.H*p.U*v.v*v.u; dims=(1,2)))[1, 1, :]
  lateralfluxes *= g.dx*g.dy/(g.Lx*g.Ly*sum(p.H))

  for j=1:p.nlayers-1
    verticalfluxes[j] = sum( @views @. p.f0^2/p.greduced[j] * (p.U[: ,:, j] - p.U[:, :, j+1])*v.v[:, :, j+1]*v.psi[:, :, j] ; dims=(1,2) )[1]
    verticalfluxes[j] *= g.dx*g.dy/(g.Lx*g.Ly*sum(p.H))
  end

  lateralfluxes, verticalfluxes
end

end # module
