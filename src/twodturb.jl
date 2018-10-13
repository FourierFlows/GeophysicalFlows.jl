module TwoDTurb

export
  Problem,
  set_q!,
  updatevars!,

  energy,
  enstrophy,
  dissipation,
  work,
  drag

using
  FFTW,
  Requires,
  Reexport

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: getfieldspecs, structvarsexpr, parsevalsum, parsevalsum2

abstract type TwoDTurbVars <: AbstractVars end

const physicalvars = [:q, :U, :V]
const transformvars = [ Symbol(var, :h) for var in physicalvars ]
const forcedvars = [:Fh]
const stochforcedvars = [:prevsol]

nothingfunction(args...) = nothing

"""
    Problem(; parameters...)

Construct a 2D turbulence problem.
"""
function Problem(;
    # Numerical parameters
          nx = 256,
          Lx = 2π,
          ny = nx,
          Ly = Lx,
          dt = 0.01,
    # Drag and/or hyper-/hypo-viscosity
          nu = 0,
         nnu = 1,
          mu = 0,
         nmu = 0,
    # Timestepper and eqn options
     stepper = "RK4",
       calcF = nothingfunction,
  stochastic = false,
           T = Float64
)

  g  = TwoDGrid(nx, Lx, ny, Ly)
  pr = Params{T}(nu, nnu, mu, nmu, calcF)
  vs = calcF == nothingfunction ? Vars(g) : (stochastic ? StochasticForcedVars(g) : ForcedVars(g))
  eq = Equation(pr, g)
  ts = FourierFlows.autoconstructtimestepper(stepper, dt, eq.LC, g)

  FourierFlows.Problem(g, vs, pr, eq, ts)
end


# ----------
# Parameters
# ----------

"""
    Params(nu, nnu, mu, nmu, calcF!)

Returns the params for two-dimensional turbulence.
"""
struct Params{T} <: AbstractParams
  nu::T              # Vorticity viscosity
  nnu::Int           # Vorticity hyperviscous order
  mu::T              # Bottom drag or hypoviscosity
  nmu::Int           # Order of hypodrag
  calcF!::Function   # Function that calculates the forcing F
end
Params(nu, nnu) = Params(nu, nnu, typeof(nu)(0), 0, nothingfunction)


# ---------
# Equations
# ---------

"""
    Equation(p, g)

Returns the equation for two-dimensional turbulence with params p and grid g.
"""
function Equation(p::Params, g; T=typeof(g.Lx))
  LC = @. -p.nu*g.KKrsq^p.nnu - p.mu*g.KKrsq^p.nmu
  LC[1, 1] = 0
  FourierFlows.Equation{T,2}(LC, calcN!)
end


# --
# Vars
# --

varspecs = cat(
  getfieldspecs(physicalvars, :(Array{T,2})),
  getfieldspecs(transformvars, :(Array{Complex{T},2})),
  dims=1)

forcedvarspecs = cat(varspecs, getfieldspecs(forcedvars, :(Array{Complex{T},2})), dims=1)
stochforcedvarspecs = cat(forcedvarspecs, getfieldspecs(stochforcedvars, :(Array{Complex{T},2})), dims=1)

# Construct Vars types
eval(structvarsexpr(:Vars, varspecs; parent=:TwoDTurbVars))
eval(structvarsexpr(:ForcedVars, forcedvarspecs; parent=:TwoDTurbVars))
eval(structvarsexpr(:StochasticForcedVars, stochforcedvarspecs; parent=:TwoDTurbVars))

"""
    Vars(g)

Returns the vars for unforced two-dimensional turbulence with grid g.
"""
function Vars(g; T=typeof(g.Lx))
  @createarrays T (g.nx, g.ny) q U V
  @createarrays Complex{T} (g.nkr, g.nl) sol qh Uh Vh
  Vars(q, U, V, qh, Uh, Vh)
end

"""
    ForcedVars(g)

Returns the vars for forced two-dimensional turbulence with grid g.
"""
function ForcedVars(g; T=typeof(g.Lx))
  v = Vars(g; T=T)
  Fh = zeros(Complex{T}, (g.nkr, g.nl))
  ForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., Fh)
end

"""
    StochasticForcedVars(g; T)

Returns the vars for stochastically forced two-dimensional turbulence with grid
g.
"""
function StochasticForcedVars(g; T=typeof(g.Lx))
  v = ForcedVars(g; T=T)
  prevsol = zeros(Complex{T}, (g.nkr, g.nl))
  StochasticForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., prevsol)
end


# -------
# Solvers
# -------

"""
    calcN_advection(N, sol, t, s, v, p, g)

Calculates the advection term.
"""
function calcN_advection!(N, sol, t, s, v, p, g)
  @. v.Uh =  im * g.l  * g.invKKrsq * sol
  @. v.Vh = -im * g.kr * g.invKKrsq * sol
  @. v.qh = sol

  ldiv!(v.U, g.rfftplan, v.Uh)
  ldiv!(v.V, g.rfftplan, v.Vh)
  ldiv!(v.q, g.rfftplan, v.qh)

  @. v.U *= v.q # U*q
  @. v.V *= v.q # V*q

  mul!(v.Uh, g.rfftplan, v.U) # \hat{U*q}
  mul!(v.Vh, g.rfftplan, v.V) # \hat{U*q}

  @. N = -im*g.kr*v.Uh - im*g.l*v.Vh
  nothing
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
function updatevars!(v, s, g)
  v.qh .= s.sol
  @. v.Uh =  im * g.l  * g.invKKrsq * s.sol
  @. v.Vh = -im * g.kr * g.invKKrsq * s.sol
  ldiv!(v.q, g.rfftplan, deepcopy(v.qh))
  ldiv!(v.U, g.rfftplan, deepcopy(v.Uh))
  ldiv!(v.V, g.rfftplan, deepcopy(v.Vh))
  nothing
end

updatevars!(prob::AbstractProblem) = updatevars!(prob.vars, prob.state, prob.grid)

"""
    set_q!(prob, q)
    set_q!(s, v, g, q)

Set the solution s.sol as the transform of q and update variables v
on the grid g.
"""
function set_q!(s, v, g, q)
  mul!(s.sol, g.rfftplan, q)
  s.sol[1, 1] = 0 # zero domain average
  updatevars!(v, s, g)
  nothing
end
set_q!(prob, q) = set_q!(prob.state, prob.vars, prob.grid, q)

"""
    energy(prob)
    energy(s, v, g)

Returns the domain-averaged kinetic energy in the Fourier-transformed vorticity
solution s.sol.
"""
@inline function energy(s, v, g)
  @. v.Uh = g.invKKrsq * abs2(s.sol)
  1/(2*g.Lx*g.Ly)*parsevalsum(v.Uh, g)
end

@inline energy(prob) = energy(prob.state, prob.vars, prob.grid)

"""
    enstrophy(s, g)

Returns the domain-averaged enstrophy in the Fourier-transformed vorticity
solution `s.sol`.
"""
@inline enstrophy(s, g) = 1/(2*g.Lx*g.Ly)*parsevalsum2(s.sol, g)

@inline enstrophy(prob) = enstrophy(prob.state, prob.grid)

"""
    dissipation(prob)
    dissipation(s, v, p, g)

Returns the domain-averaged dissipation rate. nnu must be >= 1.
"""
@inline function dissipation(s, v, p, g)
  @. v.Uh = g.KKrsq^(p.nnu-1) * abs2(s.sol)
  v.Uh[1, 1] = 0
  p.nu/(g.Lx*g.Ly)*parsevalsum(v.Uh, g)
end

@inline dissipation(prob::AbstractProblem) = dissipation(prob.state, prob.vars, prob.params, prob.grid)

"""
    work(prob)
    work(s, v, p, g)

Returns the domain-averaged rate of work of energy by the forcing Fh.
"""
@inline function work(s, v::ForcedVars, g)
  @. v.Uh = g.invKKrsq * s.sol * conj(v.Fh)
  1/(g.Lx*g.Ly)*parsevalsum(v.Uh, g)
end

@inline function work(s, v::StochasticForcedVars, g)
  @. v.Uh = g.invKKrsq * (v.prevsol + s.sol)/2.0 * conj(v.Fh) # Stratonovich
  # @. v.Uh = g.invKKrsq * v.prevsol * conj(v.Fh)             # Ito
  1/(g.Lx*g.Ly)*parsevalsum(v.Uh, g)
end

@inline work(prob::AbstractProblem) = work(prob.state, prob.vars, prob.grid)

"""
    drag(prob)
    drag(s, v, p, g)

Returns the extraction of domain-averaged energy by drag/hypodrag mu.
"""
@inline function drag(s, v, p, g)
  @. v.Uh = g.KKrsq^(p.nmu-1) * abs2(s.sol)
  v.Uh[1, 1] = 0
  p.mu/(g.Lx*g.Ly)*parsevalsum(v.Uh, g)
end

@inline drag(prob::AbstractProblem) = drag(prob.state, prob.vars, prob.params, prob.grid)

end # module
