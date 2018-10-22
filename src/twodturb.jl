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
using FourierFlows: getfieldspecs, varsexpression, parsevalsum, parsevalsum2

abstract type TwoDTurbVars <: AbstractVars end

const physicalvars = [:q, :u, :v]
const fouriervars = [:qh, :uh, :vh]
const forcedfouriervars = cat(fouriervars, [:F], dims=1)
const stochfouriervars = cat(forcedfouriervars, [:prevsol], dims=1)

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
           T = Float64)

   grid  = TwoDGrid(nx, Lx, ny, Ly; T=T)
  params = Params{T}(nu, nnu, mu, nmu, calcF)
    vars = calcF == nothingfunction ? Vars(grid) : (stochastic ? StochasticForcedVars(grid) : ForcedVars(grid))
     eqn = Equation(params, grid)
  FourierFlows.Problem(eqn, stepper, dt, grid, vars, params)
end

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

"""
    Equation(p, g)

Returns the equation for two-dimensional turbulence with params p and grid g.
"""
function Equation(p, g)
  L = @. -p.nu*g.Krsq^p.nnu - p.mu*g.Krsq^p.nmu
  L[1, 1] = 0
  FourierFlows.Equation(L, calcN!, g)
end


# Construct Vars types
eval(varsexpression(:Vars, physicalvars, fouriervars))
eval(varsexpression(:ForcedVars, physicalvars, forcedfouriervars))
eval(varsexpression(:StochasticForcedVars, physicalvars, stochfouriervars))

"""
    Vars(g)

Returns the vars for unforced two-dimensional turbulence with grid g.
"""
function Vars(g::AbstractGrid{T}) where T
  @zeros T (g.nx, g.ny) q u v
  @zeros Complex{T} (g.nkr, g.nl) qh uh vh
  Vars(q, u, v, qh, uh, vh)
end

"""
    ForcedVars(g)

Returns the vars for forced two-dimensional turbulence with grid g.
"""
function ForcedVars(g::AbstractGrid{T}) where T
  v = Vars(g)
  Fh = zeros(Complex{T}, (g.nkr, g.nl))
  ForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., Fh)
end

"""
    StochasticForcedVars(g; T)

Returns the vars for stochastically forced two-dimensional turbulence with grid
g.
"""
function StochasticForcedVars(g::AbstractGrid{T}) where T
  v = ForcedVars(g)
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
function calcN_advection!(N, sol, t, cl, v, p, g)
  @. v.uh =  im * g.l  * g.invKrsq * sol
  @. v.vh = -im * g.kr * g.invKrsq * sol
  @. v.qh = sol

  ldiv!(v.u, g.rfftplan, v.uh)
  ldiv!(v.v, g.rfftplan, v.vh)
  ldiv!(v.q, g.rfftplan, v.qh)

  @. v.u *= v.q # u*q
  @. v.v *= v.q # v*q

  mul!(v.uh, g.rfftplan, v.u) # \hat{u*q}
  mul!(v.vh, g.rfftplan, v.v) # \hat{v*q}

  @. N = -im*g.kr*v.uh - im*g.l*v.vh
  nothing
end

function calcN!(N, sol, t, cl, v, p, g)
  calcN_advection!(N, sol, t, s, v, p, g)
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
    updatevars!(v, g, sol)

Update the vars in v on the grid g with the solution in s.sol.
"""
function updatevars!(v, g, sol)
  v.qh .= sol
  @. v.uh =  im * g.l  * g.invKrsq * sol
  @. v.vh = -im * g.kr * g.invKrsq * sol
  ldiv!(v.q, g.rfftplan, deepcopy(v.qh))
  ldiv!(v.u, g.rfftplan, deepcopy(v.uh))
  ldiv!(v.v, g.rfftplan, deepcopy(v.vh))
  nothing
end

updatevars!(prob::AbstractProblem) = updatevars!(prob.vars, prob.grid, prob.sol)

"""
    set_q!(prob, q)

Set the `prob.sol` as the transform of q and update variables.
"""
function set_q!(prob, q)
  mul!(prob.sol, prob.grid.rfftplan, q)
  prob.sol[1, 1] = 0 # zero domain average
  updatevars!(prob)
  nothing
end

"""
    energy(prob)

Returns the domain-averaged kinetic energy in `sol`.
"""
@inline function energy(sol, v, g)
  @. v.uh = g.invKrsq * abs2(sol)
  1/(2*g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline energy(prob) = energy(prob.sol, prob.vars, prob.grid)

"""
    enstrophy(prob)

Returns the domain-averaged enstrophy in `sol`.
"""
@inline enstrophy(sol, g) = 1/(2*g.Lx*g.Ly)*parsevalsum2(sol, g)

@inline enstrophy(prob) = enstrophy(prob.sol, prob.grid)

"""
    dissipation(prob)

Returns the domain-averaged dissipation rate. nnu must be >= 1.
"""
@inline function dissipation(sol, v, p, g)
  @. v.uh = g.Krsq^(p.nnu-1) * abs2(sol)
  v.uh[1, 1] = 0
  p.nu/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline dissipation(prob::AbstractProblem) = dissipation(prob.sol, prob.vars, prob.params, prob.grid)

"""
    work(prob)
    work(s, v, p, g)

Returns the domain-averaged rate of work of energy by the forcing Fh.
"""
@inline function work(sol, v::ForcedVars, g)
  @. v.uh = g.invKrsq * sol * conj(v.F)
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline function work(sol, v::StochasticForcedVars, g)
  @. v.uh = 0.5 * g.invKrsq * (v.prevsol + sol) * conj(v.F) # Stratonovich
  # @. v.uh = g.invKrsq * v.prevsol * conj(v.Fh)             # Ito
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline work(prob::AbstractProblem) = work(prob.sol, prob.vars, prob.grid)

"""
    drag(prob)
    drag(s, v, p, g)

Returns the extraction of domain-averaged energy by drag/hypodrag mu.
"""
@inline function drag(sol, v, p, g)
  @. v.uh = g.Krsq^(p.nmu-1) * abs2(sol)
  v.uh[1, 1] = 0
  p.mu/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline drag(prob::AbstractProblem) = drag(prob.sol, prob.vars, prob.params, prob.grid)

end # module
