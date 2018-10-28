module BarotropicQG

export
  Problem,
  set_zeta!,
  updatevars!,

  energy,
  enstrophy,
  meanenergy,
  meanenstrophy,
  dissipation,
  work,
  drag

using Reexport

@reexport using FourierFlows

using FFTW: rfft
using LinearAlgebra: mul!, ldiv!
using FourierFlows: getfieldspecs, varsexpression, parsevalsum, parsevalsum2, superzeros

abstract type BarotropicQGVars <: AbstractVars end
abstract type BarotropicQGForcedVars <: BarotropicQGVars end

const realscalarvars = [:U]
const physicalvars = [:q, :zeta, :psi, :u, :v]
const fouriervars = [ Symbol(var, :h) for var in physicalvars ]
const forcedfouriervars = cat(fouriervars, [:Fqh], dims=1)
const stochfouriervars = cat(forcedfouriervars, [:prevsol], dims=1)

nothingfunction(args...) = nothing

"""
    Problem(; parameters...)

Construct a BarotropicQG turbulence problem.
"""
function Problem(;
    # Numerical parameters
            nx = 256,
            Lx = 2π,
            ny = nx,
            Ly = Lx,
            dt = 0.01,
    # Physical parameters
          beta = 0.0,
           eta = nothing,
    # Drag and/or hyper-/hypo-viscosity
            nu = 0.0,
           nnu = 1,
            mu = 0.0,
   # Timestepper and eqn options
       stepper = "RK4",
        calcFU = nothingfunction,
        calcFq = nothingfunction,
    stochastic = false,
             T = Float64)

  # the grid
  grid  = TwoDGrid(nx, Lx, ny, Ly; T=T)

  # params
  if eta==nothing
    params = Params{T}(beta, mu, nu, nnu, calcFq)
  else
    if typeof(eta) != Array{T,2} #this is true if eta was passes in Problem as a function
      eta = eta.(grid.x, grid.y)
    end
    if calcFU == nothingfunction
      params = TopographicParams(beta, eta, mu, nu, nnu, calcFq)
    else
      params = TopographicParamsWithU(beta, eta, rfft(eta), mu, nu, nnu, calcFU, calcFq)
    end
  end

  # equation
  eqn = Equation(params, grid)

  # vars
  if calcFq == nothingfunction && calcFU == nothingfunction
    vars = Vars(grid)
  else
    if stochastic ==  false
      vars = ForcedVars(grid)
    elseif stochastic == true
      vars = StochasticForcedVars(grid, params, eqn)
    end
  end

  # problem
  FourierFlows.Problem(eqn, stepper, dt, grid, vars, params)
end

InitialValueProblem(; kwargs...) = Problem(; kwargs...)
ForcedProblem(; kwargs...) = Problem(; kwargs...)


# ----------
# Parameters
# ----------

abstract type ParamsWithU <: AbstractParams end

"""
    Params(beta, mu, nu, nnu, calcFq)
    Params(beta, eta, mu, nu, nnu, calcFq)
    Params(beta, eta, etah, mu, nu, nnu, calcFU, calcFU)

Returns the params for an unforced two-dimensional barotropic QG problem.
"""
struct Params{T} <: AbstractParams
  beta::T                    # Planetary vorticity y-gradient
  mu::T                      # Linear drag
  nu::T                      # Viscosity coefficient
  nnu::Int                   # Hyperviscous order (nnu=1 is plain old viscosity)
  calcFq!::Function          # Function that calculates the forcing on QGPV q
end

struct TopographicParams{T} <: AbstractParams
  beta::T                    # Planetary vorticity y-gradient
  eta::Array{T,2}            # Topographic PV
  mu::T                      # Linear drag
  nu::T                      # Viscosity coefficient
  nnu::Int                   # Hyperviscous order (nnu=1 is plain old viscosity)
  calcFq!::Function          # Function that calculates the forcing on QGPV q
end

struct TopographicParamsWithU{T} <: ParamsWithU
  beta::T                    # Planetary vorticity y-gradient
  eta::Array{T,2}            # Topographic PV
  etah::Array{Complex{T},2}  # FFT of Topographic PV
  mu::T                      # Linear drag
  nu::T                      # Viscosity coefficient
  nnu::Int                   # Hyperviscous order (nnu=1 is plain old viscosity)
  calcFU::Function           # Function that calculates the forcing F(t) on
                             # domain-averaged zonal flow U(t)
  calcFq!::Function          # Function that calculates the forcing on QGPV q
end

"""
    Params(g, beta, eta::Function, mu, nu, nnu, calcFq)
    Params(g, beta, eta::Function, mu, nu, nnu, calcFU, calcFq)

Constructor for topographic Params that accepts a generating function for the topographic PV.
"""
function TopographicParams(g::AbstractGrid{T}, beta, eta::Function, mu, nu, nnu, calcFq) where T
  etagrid = @. eta(g.x, g.y)
  TopographicParams{T}(beta, etagrid, mu, nu, nnu, calcFq)
end

function TopographicParamsWithU(g::AbstractGrid{T}, beta, eta::Function, mu, nu, nnu, calcFU, calcFq) where T
  p = TopographicParams(g, beta, eta, mu, nu, nnu, calcFq)
  etah = rfft(p.eta)
  Params{T}(beta, p.eta, etah, mu, nu, nnu, calcFU, calcFq)
end


# ---------
# Equations
# ---------

"""
    Equation(p, g)

Returns the equation for two-dimensional barotropic QG problem with params p and grid g.
"""
function Equation(p, g)
  L = @. -p.mu - p.nu*g.Krsq^p.nnu + im*p.beta*g.kr*g.invKrsq
  L[1, 1] = 0
  FourierFlows.Equation(L, calcN!, g)
end

function Equation(p::ParamsWithU, g)
  LU = [ -p.mu ]
  Lq = @. -p.mu - p.nu*g.Krsq^p.nnu + im*p.beta*g.kr*g.invKrsq
  Lq[1, 1] = 0
  L = [Lq, LU]
  T = (Complex{Float64}, Float64)
  FourierFlows.Equation(L, calcN!, g; T=T)
end


# ----
# Vars
# ----

varspecs = cat(
  getfieldspecs(realscalarvars, :(Array{T,1})),
  getfieldspecs(physicalvars, :(Array{T,2})),
  getfieldspecs(fouriervars, :(Array{Complex{T},2})),
  dims=1)

forcedvarspecs = cat(
  getfieldspecs(realscalarvars, :(Array{T,1})),
  getfieldspecs(physicalvars, :(Array{T,2})),
  getfieldspecs(forcedfouriervars, :(Array{Complex{T},2})),
  dims=1)

stochforcedvarspecs = cat(
  getfieldspecs(realscalarvars, :(Array{T,1})),
  getfieldspecs(physicalvars, :(Array{T,2})),
  getfieldspecs(stochfouriervars, :(Array{Complex{T},2})),
  dims=1)


# Construct Vars types
eval(varsexpression(:Vars, varspecs; parent=:BarotropicQGVars, typeparams=:T))
eval(varsexpression(:ForcedVars, forcedvarspecs; parent=:BarotropicQGForcedVars, typeparams=:T))
eval(varsexpression(:StochasticForcedVars, stochforcedvarspecs; parent=:BarotropicQGForcedVars, typeparams=:T))


"""
    Vars(g)

Returns the vars for unforced two-dimensional barotropic QG problem with grid g.
"""
function Vars(g::AbstractGrid{T}) where T
  @zeros T (1, ) U
  @zeros T (g.nx, g.ny) q zeta psi u v
  @zeros Complex{T} (g.nkr, g.nl) qh zetah psih uh vh
  Vars(U, q, zeta, psi, u, v, qh, zetah, psih, uh, vh)
end


"""
    ForcedVars(g)

Returns the vars for forced two-dimensional barotropic QG problem with grid g.
"""
function ForcedVars(g::AbstractGrid{T}) where T
  v = Vars(g)
  Fqh = zeros(Complex{T}, (g.nkr, g.nl))
  ForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., Fqh)
end

"""
    StochasticForcedVars(g, p, eqn)

Returns the vars for stochastically forced two-dimensional barotropic QG problem with grid g.
"""
function StochasticForcedVars(g::AbstractGrid{T}, p, eqn) where T
  v = ForcedVars(g)
  @zeros Complex{T} eqn.dims prevsol
  StochasticForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., prevsol)
end

function StochasticForcedVars(g::AbstractGrid{T}, p::ParamsWithU, eqn) where T
  v = ForcedVars(g)
  @superzeros (Complex{T}, T) eqn.dims prevsol
  StochasticForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., prevsol)
end


# -------
# Solvers
# -------

function calcN_advection!(N, sol, v, p::ParamsWithU, g)
  @. v.U = real(sol[2])
  calcuvq(sol, v, p, g)
  @. v.u += v.U
  calcuqhvqh(v, g)
  @. N[1] = -im*g.kr*v.uh - im*g.l*v.vh # -∂[(U+u)q]/∂x-∂[vq]/∂y
end

function calcN_advection!(N, sol, v, p, g)
  calcuvq(sol, v, p, g)
  calcuqhvqh(v, g)
  @. N = -im*g.kr*v.uh - im*g.l*v.vh # -∂[uq]/∂x-∂[vq]/∂y
end

function calcN!(N, sol, t, cl, v, p, g)
  calcN_advection!(N, sol, v, p, g)
  addforcing!(N, sol, t, cl, v, p, g)
  nothing
end

function calcN!(N, sol, t, cl, v, p::ParamsWithU, g)
  @. v.vh = -im * g.kr * g.invKrsq * sol[1]
  # 'Nonlinear' term for U(t) with topographic correlation.
  # Note: < v*eta > = 2*sum( conj(vh)*eta ) / (nx^2*ny^2) if rfft is used
  N[2] .= p.calcFU(t) + 2*sum(conj(v.vh).*p.etah).re / (g.nx^2*g.ny^2)
  # FFTW's irfft destroys its input; compute form stress before v.vh is destroyed
  calcN_advection!(N, sol, v, p, g)
  addforcing!(N, sol, t, cl, v, p, g)
  nothing
end

function calcuvq(sol, v, p, g)
  getzetah!(v.zetah, sol, p)
  @. v.uh =  im * g.l  * g.invKrsq * v.zetah
  @. v.vh = -im * g.kr * g.invKrsq * v.zetah
  ldiv!(v.u, g.rfftplan, v.uh)
  ldiv!(v.v, g.rfftplan, v.vh)
  ldiv!(v.zeta, g.rfftplan, v.zetah)
  calcq(v, p)
end

@inline function getzetah!(zetah, sol, p)
  @. zetah = sol
  zetah[1, 1] = 0.0
end

@inline function getzetah!(zetah, sol, p::ParamsWithU)
  @. zetah = sol[1]
  zetah[1, 1] = 0.0
end

@inline function calcq(v, p)
  @. v.q = v.zeta + p.eta
end

@inline function calcq(v, p::Params)
  @. v.q = v.zeta
end

function calcuqhvqh(v, g)
  @. v.u = v.u*v.q # u*q
  @. v.v = v.v*v.q # v*q
  mul!(v.uh, g.rfftplan, v.u) # \hat{u*q}
  mul!(v.vh, g.rfftplan, v.v) # \hat{v*q}
end

addforcing!(N, sol, t, cl, v::Vars, p, g) = nothing

function addforcing!(N, sol, t, cl, v::ForcedVars, p, g)
  p.calcFq!(v.Fqh, sol, t, cl, v, p, g)
  @. N += v.Fqh
  nothing
end

function addforcing!(N, sol, t, cl, v::ForcedVars, p::ParamsWithU, g)
  p.calcFq!(v.Fqh, sol, t, cl, v, p, g)
  @. N[1] += v.Fqh
  nothing
end

function addforcing!(N, sol, t, cl, v::StochasticForcedVars, p, g)
  if t == cl.t # not a substep
    @. v.prevsol = sol # sol at previous time-step is needed to compute budgets for stochastic forcing
    p.calcFq!(v.Fqh, sol, t, cl, v, p, g)
  end
  @. N += v.Fqh
  nothing
end

function addforcing!(N, sol, t, cl, v::StochasticForcedVars, p::ParamsWithU, g)
  if t == cl.t # not a substep
    @. v.prevsol = sol # sol at previous time-step is needed to compute budgets for stochastic forcing
    p.calcFq!(v.Fqh, sol, t, cl, v, p, g)
  end
  @. N[1] += v.Fqh
  nothing
end



# ----------------
# Helper functions
# ----------------

"""
    updatevars!(p, v, g, sol)

Update the vars in v on the grid g with the solution in sol.
"""
function updatevars!(p, v, g, sol)
  getzetah!(v.zetah, sol, p)
  updatezetapsiuv(v, g)
  calcq(v, p)
  nothing
end

function updatevars!(p::ParamsWithU, v, g, sol)
  @. v.U = real(sol[2])
  getzetah!(v.zetah, sol, p)
  updatezetapsiuv(v, g)
  calcq(v, p)
  nothing
end

function updatezetapsiuv(v, g)
  @. v.psih = -v.zetah * g.invKrsq
  @. v.uh = -im * g.l  * v.psih
  @. v.vh =  im * g.kr * v.psih

  ldiv!(v.zeta, g.rfftplan, deepcopy(v.zetah))
  ldiv!(v.psi, g.rfftplan, deepcopy(v.psih))
  ldiv!(v.u, g.rfftplan, deepcopy(v.uh))
  ldiv!(v.v, g.rfftplan, deepcopy(v.vh))
end


updatevars!(prob) = updatevars!(prob.params, prob.vars, prob.grid, prob.sol)

"""
    set_zeta!(prob, zeta)
    set_zeta!(p, v, g, sol, zeta)

Set the solution sol as the transform of zeta and update variables v
on the grid g.
"""
function set_zeta!(p::ParamsWithU, v, g, sol, zeta)
  mul!(v.zetah, g.rfftplan, zeta)
  v.zetah[1, 1] = 0.0
  @. sol[1] = v.zetah

  updatevars!(p, v, g, sol)
  nothing
end

function set_zeta!(p, v, g, sol, zeta)
  mul!(v.zetah, g.rfftplan, zeta)
  v.zetah[1, 1] = 0.0
  @. sol = v.zetah

  updatevars!(p, v, g, sol)
  nothing
end

set_zeta!(prob, zeta) = set_zeta!(prob.params, prob.vars, prob.grid, prob.sol, zeta)

"""
    set_U!(sol, prob, U)

Sets a value for U(t) in the relevantpart of the solution `sol`.
"""
function set_U!(sol, prob, U)
  p, v, g = prob.params, prob.vars, prob.grid
  sol[2] .= [U]
  updatevars!(p, v, g, sol)
  nothing
end


"""
Calculate the domain-averaged kinetic energy.
"""
function energy(prob)
  v, p, g, sol = prob.vars, prob.params, prob.grid, prob.sol
  getzetah!(v.zetah, sol, p)
  v.zetah[1, 1] = 0
  @. v.uh = g.invKrsq * abs2(v.zetah) # |\hat{q}|^2/k^2
  1/(2*g.Lx*g.Ly)*parsevalsum(v.uh, g)
end


"""
Returns the domain-averaged enstrophy.
"""
@inline function enstrophy(prob)
  v, p, g, sol = prob.vars, prob.params, prob.grid, prob.sol
  getzetah!(v.zetah, sol, p)
  v.zetah[1, 1] = 0
  0.5*parsevalsum2(v.zetah, g)/(g.Lx*g.Ly)
end

"""
Returns the energy of the domain-averaged U.
"""
@inline meanenergy(prob) = real(0.5*prob.sol[2][1]^2)

"""
Returns the enstrophy of the domain-averaged U.
"""
@inline meanenstrophy(prob) = real(prob.params.beta*prob.sol[2][1])

"""
    dissipation(prob)
    dissipation(s, v, p, g)

Returns the domain-averaged dissipation rate. nnu must be >= 1.
"""
@inline function dissipation(p, v, g, sol)
  @. v.uh = g.Krsq^(p.nnu-1) * abs2(sol)
  v.uh[1, 1] = 0
  p.nu/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline dissipation(prob) = dissipation(prob.params, prob.vars, prob.grid, prob.sol)

"""
    work(prob)
    work(s, v, p, g)

Returns the domain-averaged rate of work of energy by the forcing Fqh.
"""
@inline function work(sol, v::ForcedVars, g)
  @. v.uh = g.invKrsq * sol * conj(v.Fqh)
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline function work(sol, v::StochasticForcedVars, g)
  @. v.uh = g.invKrsq * (v.prevsol + sol)/2.0 * conj(v.Fqh) # Stratonovich
  # @. v.uh = g.invKrsq * v.prevsol * conj(v.Fqh)             # Ito
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline work(prob) = work(prob.sol, prob.vars, prob.grid)

"""
    drag(prob)
    drag(s, v, p, g)

Returns the extraction of domain-averaged energy by drag mu.
"""
@inline function drag(p, v, g, sol)
  @. v.uh = g.Krsq^(-1) * abs2(sol)
  v.uh[1, 1] = 0
  p.mu/(g.Lx*g.Ly)*FourierFlows.parsevalsum(v.uh, g)
end

@inline drag(prob) = drag(prob.params, prob.vars, prob.grid, prob.sol)


end # module
