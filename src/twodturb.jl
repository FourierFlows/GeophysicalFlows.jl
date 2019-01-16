module TwoDTurb

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
using FourierFlows: getfieldspecs, structvarsexpr, parsevalsum, parsevalsum2

abstract type TwoDTurbVars <: AbstractVars end

const physicalvars = [:zeta, :u, :v]
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

  gr = TwoDGrid(nx, Lx, ny, Ly)
  pr = Params{T}(nu, nnu, mu, nmu, calcF)
  vs = calcF == nothingfunction ? Vars(gr) : (stochastic ? StochasticForcedVars(gr) : ForcedVars(gr))
  eq = Equation(pr, gr)
  FourierFlows.Problem(eq, stepper, dt, gr, vs, pr)
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
function Equation(p::Params, g::AbstractGrid{T}) where T
  L = @. - p.nu*g.Krsq^p.nnu - p.mu*g.Krsq^p.nmu
  L[1, 1] = 0
  FourierFlows.Equation(L, calcN!, g)
end


# ----
# Vars
# ----

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
  @createarrays T (g.nx, g.ny) zeta u v
  @createarrays Complex{T} (g.nkr, g.nl) sol zetah uh vh
  Vars(zeta, u, v, zetah, uh, vh)
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
    calcN_advection(N, sol, t, cl, v, p, g)

Calculates the advection term.
"""
function calcN_advection!(N, sol, t, cl, v, p, g)
  @. v.uh =  im * g.l  * g.invKrsq * sol
  @. v.vh = -im * g.kr * g.invKrsq * sol
  @. v.zetah = sol

  ldiv!(v.u, g.rfftplan, v.uh)
  ldiv!(v.v, g.rfftplan, v.vh)
  ldiv!(v.zeta, g.rfftplan, v.zetah)

  @. v.u *= v.zeta # u*zeta
  @. v.v *= v.zeta # v*zeta

  mul!(v.uh, g.rfftplan, v.u) # \hat{u*zeta}
  mul!(v.vh, g.rfftplan, v.v) # \hat{v*zeta}

  @. N = -im*g.kr*v.uh - im*g.l*v.vh
  nothing
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
    updatevars!(prob)

Update the vars in v on the grid g with the solution in sol.
"""
function updatevars!(prob)
  v, g, sol = prob.vars, prob.grid, prob.sol
  v.zetah .= sol
  @. v.uh =  im * g.l  * g.invKrsq * sol
  @. v.vh = -im * g.kr * g.invKrsq * sol
  ldiv!(v.zeta, g.rfftplan, deepcopy(v.zetah))
  ldiv!(v.u, g.rfftplan, deepcopy(v.uh))
  ldiv!(v.v, g.rfftplan, deepcopy(v.vh))
  nothing
end

"""
    set_zeta!(prob, zeta)

Set the solution sol as the transform of zeta and update variables v
on the grid g.
"""
function set_zeta!(prob, zeta)
  p, v, g, sol = prob.params, prob.vars, prob.grid, prob.sol
  mul!(sol, g.rfftplan, zeta)
  sol[1, 1] = 0 # zero domain average
  updatevars!(prob)
  nothing
end

"""
    energy(prob)

Returns the domain-averaged kinetic energy in the Fourier-transformed vorticity
solution `sol`.
"""
@inline function energy(prob)
  sol, v, g = prob.sol, prob.vars, prob.grid
  @. v.uh = g.invKrsq * abs2(sol)
  1/(2*g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

"""
    enstrophy(prob)

Returns the domain-averaged enstrophy in the Fourier-transformed vorticity
solution `sol`.
"""
@inline function enstrophy(prob)
  sol, g = prob.sol, prob.grid
  1/(2*g.Lx*g.Ly)*parsevalsum2(sol, g)
end

"""
    dissipation(prob)

Returns the domain-averaged dissipation rate. nnu must be >= 1.
"""
@inline function dissipation(prob)
  sol, v, p, g = prob.sol, prob.vars, prob.params, prob.grid
  @. v.uh = g.Krsq^(p.nnu-1) * abs2(sol)
  v.uh[1, 1] = 0
  p.nu/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

"""
    work(prob)
    work(sol, v, g)

Returns the domain-averaged rate of work of energy by the forcing Fh.
"""
@inline function work(sol, v::ForcedVars, g)
  @. v.uh = g.invKrsq * sol * conj(v.Fh)
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline function work(sol, v::StochasticForcedVars, g)
  @. v.uh = g.invKrsq * (v.prevsol + sol)/2.0 * conj(v.Fh) # Stratonovich
  # @. v.uh = g.invKrsq * v.prevsol * conj(v.Fh)           # Ito
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline work(prob) = work(prob.sol, prob.vars, prob.grid)

"""
    drag(prob)

Returns the extraction of domain-averaged energy by drag/hypodrag mu.
"""
@inline function drag(prob)
  sol, v, p, g = prob.sol, prob.vars, prob.params, prob.grid
  @. v.uh = g.Krsq^(p.nmu-1) * abs2(sol)
  v.uh[1, 1] = 0
  p.mu/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

end # module
