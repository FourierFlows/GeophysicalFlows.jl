module BarotropicQGQL

export
  Problem,
  set_zeta!,
  updatevars!,

  energy,
  enstrophy,
  energy00,
  enstrophy00,
  dissipation,
  work,
  drag

using
  FFTW,
  Requires,
  Reexport

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum, parsevalsum2
import FFTW: rfft

abstract type BarotropicQGQLVars <: AbstractVars end
abstract type BarotropicQGQLForcedVars <: BarotropicQGQLVars end

const physicalvars = [:zeta, :psi, :u, :v, :uzeta, :vzeta, :U, :Zeta, :Psi]
const transformvars = [ Symbol(var, :h) for var in physicalvars ]
const forcedvars = [:Fh]
const stochforcedvars = [:prevsol]

nothingfunction(args...) = nothing

"""
    Problem(; parameters...)

Construct a BarotropicQGQL turbulence problem.
"""
function Problem(;
    # Numerical parameters
            nx = 256,
            Lx = 2π,
            ny = nx,
            Ly = Lx,
            dt = 0.01,
    # Physical parameters
            f0 = 1.0,
          beta = 0.0,
           eta = nothing,
    # Drag and/or hyper-/hypo-viscosity
            nu = 0.0,
           nnu = 1,
            mu = 0.0,
   # Timestepper and eqn options
       stepper = "RK4",
         calcF = nothingfunction,
    stochastic = false,
             T = Float64)

  # the grid
  g  = BarotropicQGQL.TwoDGrid(nx, Lx, ny, Ly)

  # topographic PV
  if eta==nothing
    eta = 0*g.X
    etah = rfft(eta)
  end

  if typeof(eta)!=Array{Float64,2} #this is true if eta was passes in Problem as a function
    pr = Params(g, f0, beta, eta, mu, nu, nnu, calcF)
  else
    pr = Params(f0, beta, eta, rfft(eta), mu, nu, nnu, calcF)
  end
  vs = calcF == nothingfunction ? Vars(g) : (stochastic ? StochasticForcedVars(g) : ForcedVars(g))
  eq = BarotropicQGQL.Equation(pr, g)
  ts = FourierFlows.autoconstructtimestepper(stepper, dt, eq.LC, g)
  FourierFlows.Problem(g, vs, pr, eq, ts)
end

InitialValueProblem(; kwargs...) = Problem(; kwargs...)
ForcedProblem(; kwargs...) = Problem(; kwargs...)


# ----------
# Parameters
# ----------

"""
    Params(g::TwoDGrid, f0, beta, FU, eta, mu, nu, nnu)

Returns the params for an unforced two-dimensional barotropic QG problem.
"""
struct Params{T} <: AbstractParams
  f0::T                      # Constant planetary vorticity
  beta::T                    # Planetary vorticity y-gradient
  eta::Array{T,2}            # Topographic PV
  etah::Array{Complex{T},2}  # FFT of Topographic PV
  mu::T                      # Linear drag
  nu::T                      # Viscosity coefficient
  nnu::Int                   # Hyperviscous order (nnu=1 is plain old viscosity)
  calcF!::Function           # Function that calculates the forcing on QGPV q
end

"""
    Params(g::TwoDGrid, f0, beta, eta::Function, mu, nu, nnu)

Constructor for Params that accepts a generating function for the topographic PV.
"""
function Params(g::TwoDGrid, f0, beta, eta::Function, mu, nu, nnu, calcF)
  etagrid = eta(g.X, g.Y)
  etah = rfft(etagrid)
  Params(f0, beta, etagrid, etah, mu, nu, nnu, calcF)
end


# ---------
# Equations
# ---------

"""
    Equation(p, g)

Returns the equation for two-dimensional barotropic QG problem with params p and grid g.
"""
function Equation(p::Params, g; T=typeof(g.Lx))
  LC = @. -p.mu - p.nu*g.KKrsq^p.nnu + im*p.beta*g.kr*g.invKKrsq
  LC[1, 1] = 0
  FourierFlows.Equation{Complex{T},2}(LC, calcN!)
end


# ----
# Vars
# ----

"""
    Vars(g)

Returns the vars for unforced two-dimensional barotropic QG problem with grid g.
"""
mutable struct Vars{T} <: BarotropicQGQLVars
  u::Array{T,2}
  v::Array{T,2}
  U::Array{T,2}
  uzeta::Array{T,2}
  vzeta::Array{T,2}
  zeta::Array{T,2}
  Zeta::Array{T,2}
  psi::Array{T,2}
  Psi::Array{T,2}
  Nz::Array{Complex{T},2}
  NZ::Array{Complex{T},2}
  uh::Array{Complex{T},2}
  vh::Array{Complex{T},2}
  Uh::Array{Complex{T},2}
  zetah::Array{Complex{T},2}
  Zetah::Array{Complex{T},2}
  psih::Array{Complex{T},2}
  Psih::Array{Complex{T},2}
end

function Vars(g; T=typeof(g.Lx))
  @createarrays T (g.nx, g.ny) u v U uzeta vzeta zeta Zeta psi Psi
  @createarrays Complex{T} (g.nkr, g.nl) N NZ uh vh Uh zetah Zetah psih Psih
  Vars(u, v, U, uzeta, vzeta, zeta, Zeta, psi, Psi, N, NZ, uh, vh, Uh, zetah, Zetah, psih, Psih)
end

"""
    ForcedVars(g)

Returns the vars for forced two-dimensional barotropic QG problem with grid g.
"""
mutable struct ForcedVars{T} <: BarotropicQGQLForcedVars
  u::Array{T,2}
  v::Array{T,2}
  U::Array{T,2}
  uzeta::Array{T,2}
  vzeta::Array{T,2}
  zeta::Array{T,2}
  Zeta::Array{T,2}
  psi::Array{T,2}
  Psi::Array{T,2}
  Nz::Array{Complex{T},2}
  NZ::Array{Complex{T},2}
  uh::Array{Complex{T},2}
  vh::Array{Complex{T},2}
  Uh::Array{Complex{T},2}
  zetah::Array{Complex{T},2}
  Zetah::Array{Complex{T},2}
  psih::Array{Complex{T},2}
  Psih::Array{Complex{T},2}
  Fh::Array{Complex{T},2}
end

function ForcedVars(g; T=typeof(g.Lx))
  v = Vars(g; T=T)
  Fh = zeros(Complex{T}, (g.nkr, g.nl))
  ForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., Fh)
end


mutable struct StochasticForcedVars{T} <: BarotropicQGQLForcedVars
  u::Array{T,2}
  v::Array{T,2}
  U::Array{T,2}
  uzeta::Array{T,2}
  vzeta::Array{T,2}
  zeta::Array{T,2}
  Zeta::Array{T,2}
  psi::Array{T,2}
  Psi::Array{T,2}
  Nz::Array{Complex{T},2}
  NZ::Array{Complex{T},2}
  uh::Array{Complex{T},2}
  vh::Array{Complex{T},2}
  Uh::Array{Complex{T},2}
  zetah::Array{Complex{T},2}
  Zetah::Array{Complex{T},2}
  psih::Array{Complex{T},2}
  Psih::Array{Complex{T},2}
  Fh::Array{Complex{T},2}
  prevsol::Array{Complex{T},2}
end

function StochasticForcedVars(g; T=typeof(g.Lx))
  v = ForcedVars(g; T=T)
  prevsol = zeros(Complex{T}, (g.nkr, g.nl))
  StochasticForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., prevsol)
end


# -------
# Solvers
# -------

function calcN_advection!(N, sol, t, s, v, p, g)
  # Note that U = sol[1, 1]. For all other elements ζ = sol
  @. v.zetah = sol
  @. v.zetah[g.Kr.==0] = 0
  @. v.Zetah = sol
  @. v.Zetah[abs.(g.Kr).>0] = 0

  @. v.uh =  im * g.l  * g.invKKrsq * v.zetah
  @. v.vh = -im * g.kr * g.invKKrsq * v.zetah
  @. v.Uh =  im * g.l  * g.invKKrsq * v.Zetah

  ldiv!(v.zeta, g.rfftplan, v.zetah)
  ldiv!(v.u, g.rfftplan, v.uh)
  ldiv!(v.v, g.rfftplan, v.vh)

  ldiv!(v.Zeta, g.rfftplan, v.Zetah)
  ldiv!(v.U, g.rfftplan, v.Uh)

  @. v.uzeta = v.u*v.zeta # u*ζ
  @. v.vzeta = v.v*v.zeta # v*ζ

  mul!(v.uh, g.rfftplan, v.uzeta) # \hat{u*q}
  @. v.NZ = -im*g.kr*v.uh # -∂[u*q]/∂x
  mul!(v.vh, g.rfftplan, v.vzeta) # \hat{v*q}
  @. v.NZ += - im*g.l*v.vh # -∂[v*q]/∂y
  @. v.NZ[abs.(g.Kr).>0] = 0

  @. v.U = v.U*v.zeta # U*ζ
  @. v.u = v.u*v.Zeta # u*Ζ
  @. v.v = v.v*v.Zeta # v*Ζ

  mul!(v.uh, g.rfftplan, v.U + v.u) # \hat{U*ζ + u*Ζ}
  @. v.Nz = -im*g.kr*v.uh # -∂[U*ζ + u*Ζ]/∂x
  mul!(v.vh, g.rfftplan, v.v) # \hat{v*Z}
  @. v.Nz += - im*g.l*v.vh # -∂[v*Z]/∂y
  @. v.Nz[abs.(g.Kr).==0] = 0

  @. N = v.NZ + v.Nz
end

function calcN!(N, sol, t, s, v, p, g)
  calcN_advection!(N, sol, t, s, v, p, g)
  addforcing!(N, t, s, v, p, g)
  nothing
end

addforcing!(N, t, s, v::Vars, p, g) = nothing

function addforcing!(N, t, s, v::ForcedVars, p, g)
  p.calcF!(v.Fh, t, s, v, p, g)
  @. N += v.Fh
  nothing
end

function addforcing!(N, t, s, v::StochasticForcedVars, p, g)
  if t == s.t # not a substep
    @. v.prevsol = s.sol # sol at previous time-step is needed to compute budgets for stochastic forcing
    p.calcF!(v.Fh, t, s, v, p, g)
  end
  @. N += v.Fh
  nothing
end


# ----------------
# Helper functions
# ----------------

"""
    updatevars!(v, s, g)

Update the vars in v on the grid g with the solution in s.sol.
"""
function updatevars!(s, v, p, g)
  s.sol[1, 1] = 0
  @. v.zetah = s.sol
  @. v.zetah[g.Kr.==0] = 0
  @. v.Zetah = s.sol
  @. v.Zetah[abs.(g.Kr).>0] = 0

  @. v.Psih = -v.Zetah * g.invKKrsq
  @. v.psih = -v.zetah * g.invKKrsq
  @. v.uh = -im * g.l  * v.psih
  @. v.vh =  im * g.kr * v.psih
  @. v.Uh =  im * g.l  * v.Zetah * g.invKKrsq

  ldiv!(v.zeta, g.rfftplan, deepcopy(v.zetah))
  ldiv!(v.Zeta, g.rfftplan, deepcopy(v.Zetah))
  ldiv!(v.psi, g.rfftplan, v.psih)
  ldiv!(v.Psi, g.rfftplan, v.Psih)
  ldiv!(v.u, g.rfftplan, deepcopy(v.uh))
  ldiv!(v.v, g.rfftplan, deepcopy(v.vh))
  ldiv!(v.U, g.rfftplan, deepcopy(v.Uh))

  nothing
end

updatevars!(prob) = updatevars!(prob.state, prob.vars, prob.params, prob.grid)

"""
    set_zeta!(prob, zeta)
    set_zeta!(s, v, g, zeta)

Set the solution s.sol as the transform of zeta and update variables v
on the grid g.
"""
function set_zeta!(s, v, p, g, zeta)
  mul!(v.zetah, g.rfftplan, zeta)
  v.zetah[1, 1] = 0.0
  @. s.sol = v.zetah

  updatevars!(s, v, p, g)
  nothing
end

set_zeta!(prob::AbstractProblem, zeta) = set_zeta!(prob.state, prob.vars, prob.params, prob.grid, zeta)


"""
Calculate the domain-averaged kinetic energy.
"""
function energy(prob::AbstractProblem)
  s, g = prob.state, prob.grid
  0.5*(parsevalsum2(g.Kr.*g.invKKrsq.*s.sol, g)
        + parsevalsum2(g.Lr.*g.invKKrsq.*s.sol, g))/(g.Lx*g.Ly)
end


"""
Returns the domain-averaged enstrophy.
"""
function enstrophy(prob)
  s, v, g = prob.state, prob.vars, prob.grid
  @. v.uh = s.sol
  v.uh[1, 1] = 0
  0.5*parsevalsum2(v.uh, g)/(g.Lx*g.Ly)
end


"""
    dissipation(prob)
    dissipation(s, v, p, g)

Returns the domain-averaged dissipation rate. nnu must be >= 1.
"""
@inline function dissipation(s, v, p, g)
  @. v.uh = g.KKrsq^(p.nnu-1) * abs2(s.sol)
  v.uh[1, 1] = 0
  p.nu/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline dissipation(prob::AbstractProblem) = dissipation(prob.state, prob.vars, prob.params, prob.grid)

"""
    work(prob)
    work(s, v, p, g)

Returns the domain-averaged rate of work of energy by the forcing Fh.
"""
@inline function work(s, v::ForcedVars, g)
  @. v.uh = g.invKKrsq * s.sol * conj(v.Fh)
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline function work(s, v::StochasticForcedVars, g)
  @. v.uh = g.invKKrsq * (v.prevsol + s.sol)/2.0 * conj(v.Fh) # Stratonovich
  # @. v.uh = g.invKKrsq * v.prevsol * conj(v.Fh)             # Ito
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline work(prob::AbstractProblem) = work(prob.state, prob.vars, prob.grid)

"""
    drag(prob)
    drag(s, v, p, g)

Returns the extraction of domain-averaged energy by drag mu.
"""
@inline function drag(s, v, p, g)
  @. v.uh = g.KKrsq^(-1) * abs2(s.sol)
  v.uh[1, 1] = 0
  p.mu/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline drag(prob::AbstractProblem) = drag(prob.state, prob.vars, prob.params, prob.grid)


end # module
