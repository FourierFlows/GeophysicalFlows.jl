module BarotropicQGQL

export
  Problem,
  set_zeta!,
  updatevars!,

  energy,
  enstrophy,
  dissipation,
  work,
  drag

using
  FFTW,
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
  gr  = TwoDGrid(nx, Lx, ny, Ly; T=T)
  x, y = gridpoints(gr)

  # topographic PV
  if eta==nothing
    eta = 0*x
    etah = rfft(eta)
  end

  if typeof(eta)!=Array{T, 2} #this is true if eta was passes in Problem as a function
    pr = Params(gr, f0, beta, eta, mu, nu, nnu, calcF)
  else
    pr = Params{T}(f0, beta, eta, rfft(eta), mu, nu, nnu, calcF)
  end
  vs = calcF == nothingfunction ? Vars(gr) : (stochastic ? StochasticForcedVars(gr) : ForcedVars(gr))
  eq = BarotropicQGQL.Equation(pr, gr)
  FourierFlows.Problem(eq, stepper, dt, gr, vs, pr)
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
  eta::Array{T, 2}           # Topographic PV
  etah::Array{Complex{T}, 2} # FFT of Topographic PV
  mu::T                      # Linear drag
  nu::T                      # Viscosity coefficient
  nnu::Int                   # Hyperviscous order (nnu=1 is plain old viscosity)
  calcF!::Function           # Function that calculates the forcing on QGPV q
end

"""
    Params(g::TwoDGrid, f0, beta, eta::Function, mu, nu, nnu)

Constructor for Params that accepts a generating function for the topographic PV.
"""
function Params(g::AbstractGrid{T}, f0, beta, eta::Function, mu, nu, nnu, calcF) where T
  x, y = gridpoints(g)
  etagrid = eta(x, y)
  etah = rfft(etagrid)
  Params{T}(f0, beta, etagrid, etah, mu, nu, nnu, calcF)
end


# ---------
# Equations
# ---------

"""
    Equation(p, g)

Returns the equation for two-dimensional barotropic QG problem with params p and grid g.
"""
function Equation(p::Params, g::AbstractGrid{T}) where T
  L = @. -p.mu - p.nu*g.Krsq^p.nnu + im*p.beta*g.kr*g.invKrsq
  L[1, 1] = 0
  FourierFlows.Equation(L, calcN!, g)
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

function Vars(g::AbstractGrid{T}) where T
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

function ForcedVars(g::AbstractGrid{T}) where T
  v = Vars(g)
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

function StochasticForcedVars(g::AbstractGrid{T}) where T
  v = ForcedVars(g)
  prevsol = zeros(Complex{T}, (g.nkr, g.nl))
  StochasticForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., prevsol)
end


# -------
# Solvers
# -------

function calcN_advection!(N, sol, t, cl, v, p, g)
  # Note that U = sol[1, 1]. For all other elements ζ = sol
  Kr = [ g.kr[i] for i=1:g.nkr, j=1:g.nl]
  @. v.zetah = sol
  @. v.zetah[Kr .== 0] = 0
  @. v.Zetah = sol
  @. v.Zetah[abs.(Kr) .> 0] = 0

  @. v.uh =  im * g.l  * g.invKrsq * v.zetah
  @. v.vh = -im * g.kr * g.invKrsq * v.zetah
  @. v.Uh =  im * g.l  * g.invKrsq * v.Zetah

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
  @. v.NZ[abs.(Kr) .> 0] = 0

  @. v.U = v.U*v.zeta # U*ζ
  @. v.u = v.u*v.Zeta # u*Ζ
  @. v.v = v.v*v.Zeta # v*Ζ

  mul!(v.uh, g.rfftplan, v.U + v.u) # \hat{U*ζ + u*Ζ}
  @. v.Nz = -im*g.kr*v.uh # -∂[U*ζ + u*Ζ]/∂x
  mul!(v.vh, g.rfftplan, v.v) # \hat{v*Z}
  @. v.Nz += - im*g.l*v.vh # -∂[v*Z]/∂y
  @. v.Nz[abs.(Kr) .== 0] = 0

  @. N = v.NZ + v.Nz
end

function calcN!(N, sol, t, cl, v, p, g)
  calcN_advection!(N, sol, t, cl, v, p, g)
  addforcing!(N, sol, t, cl, v, p, g)
  nothing
end

addforcing!(N, sol, t, cl, v::Vars, p, g) = nothing

function addforcing!(N, sol, t, cl, v::ForcedVars, p, g)
  p.calcF!(v.Fh, sol, t, cl, v, p, g)
  @. N += v.Fh
  nothing
end

function addforcing!(N, sol, t, cl, v::StochasticForcedVars, p, g)
  if t == cl.t # not a substep
    @. v.prevsol = sol # sol at previous time-step is needed to compute budgets for stochastic forcing
    p.calcF!(v.Fh, sol, t, cl, v, p, g)
  end
  @. N += v.Fh
  nothing
end


# ----------------
# Helper functions
# ----------------

"""
    updatevars!(v, s, g)

Update the vars in v on the grid g with the solution in sol.
"""
function updatevars!(sol, v, p, g)
  Kr = [ g.kr[i] for i=1:g.nkr, j=1:g.nl]
  sol[1, 1] = 0
  @. v.zetah = sol
  @. v.zetah[Kr .== 0] = 0
  @. v.Zetah = sol
  @. v.Zetah[abs.(Kr) .> 0] = 0

  @. v.Psih = -v.Zetah * g.invKrsq
  @. v.psih = -v.zetah * g.invKrsq
  @. v.uh = -im * g.l  * v.psih
  @. v.vh =  im * g.kr * v.psih
  @. v.Uh =  im * g.l  * v.Zetah * g.invKrsq

  ldiv!(v.zeta, g.rfftplan, deepcopy(v.zetah))
  ldiv!(v.Zeta, g.rfftplan, deepcopy(v.Zetah))
  ldiv!(v.psi, g.rfftplan, v.psih)
  ldiv!(v.Psi, g.rfftplan, v.Psih)
  ldiv!(v.u, g.rfftplan, deepcopy(v.uh))
  ldiv!(v.v, g.rfftplan, deepcopy(v.vh))
  ldiv!(v.U, g.rfftplan, deepcopy(v.Uh))

  nothing
end

updatevars!(prob) = updatevars!(prob.sol, prob.vars, prob.params, prob.grid)

"""
    set_zeta!(prob, zeta)
    set_zeta!(sol, v, g, zeta)

Set the solution sol as the transform of zeta and update variables v
on the grid g.
"""
function set_zeta!(sol, v, p, g, zeta)
  mul!(v.zetah, g.rfftplan, zeta)
  v.zetah[1, 1] = 0.0
  @. sol = v.zetah

  updatevars!(sol, v, p, g)
  nothing
end

set_zeta!(prob, zeta) = set_zeta!(prob.sol, prob.vars, prob.params, prob.grid, zeta)


"""
Calculate the domain-averaged kinetic energy.
"""
function energy(prob)
  sol, g = prob.sol, prob.grid
  0.5*(parsevalsum2(g.kr.*g.invKrsq.*sol, g)
        + parsevalsum2(g.l.*g.invKrsq.*sol, g))/(g.Lx*g.Ly)
end


"""
Returns the domain-averaged enstrophy.
"""
function enstrophy(prob)
  sol, v, g = prob.sol, prob.vars, prob.grid
  @. v.uh = sol
  v.uh[1, 1] = 0
  0.5*parsevalsum2(v.uh, g)/(g.Lx*g.Ly)
end


"""
    dissipation(prob)
    dissipation(sol, v, p, g)

Returns the domain-averaged dissipation rate. nnu must be >= 1.
"""
@inline function dissipation(sol, v, p, g)
  @. v.uh = g.Krsq^(p.nnu-1) * abs2(sol)
  v.uh[1, 1] = 0
  p.nu/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline dissipation(prob) = dissipation(prob.sol, prob.vars, prob.params, prob.grid)

"""
    work(prob)
    work(sol, v, p, g)

Returns the domain-averaged rate of work of energy by the forcing Fh.
"""
@inline function work(sol, v::ForcedVars, g)
  @. v.uh = g.invKrsq * sol * conj(v.Fh)
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline function work(sol, v::StochasticForcedVars, g)
  @. v.uh = g.invKrsq * (v.prevsol + sol)/2.0 * conj(v.Fh) # Stratonovich
  # @. v.uh = g.invKrsq * v.prevsol * conj(v.Fh)             # Ito
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline work(prob) = work(prob.sol, prob.vars, prob.grid)

"""
    drag(prob)
    drag(sol, v, p, g)

Returns the extraction of domain-averaged energy by drag mu.
"""
@inline function drag(sol, v, p, g)
  @. v.uh = g.Krsq^(-1) * abs2(sol)
  v.uh[1, 1] = 0
  p.mu/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline drag(prob) = drag(prob.sol, prob.vars, prob.params, prob.grid)


end # module
