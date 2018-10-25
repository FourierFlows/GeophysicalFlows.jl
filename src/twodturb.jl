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
using FourierFlows: getfieldspecs, varsexpression, parsevalsum, parsevalsum2

abstract type TwoDTurbVars <: AbstractVars end

const physicalvars = [:zeta, :u, :v]
const fouriervars = [:zetah, :uh, :vh]
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
  @zeros T (g.nx, g.ny) zeta u v
  @zeros Complex{T} (g.nkr, g.nl) zetah uh vh
  Vars(zeta, u, v, zetah, uh, vh)
end

"""
    ForcedVars(g)

Returns the vars for forced two-dimensional turbulence with grid g.
"""
function ForcedVars(g::AbstractGrid{T}) where T
  v = Vars(g)
  F = zeros(Complex{T}, (g.nkr, g.nl))
  ForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., F)
end

"""
    StochasticForcedVars(g)

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
  p.calcF!(v.F, sol, t, cl, v, p, g)
  @. N += v.F
  nothing
end

function addforcing!(N, sol, t, cl, v::StochasticForcedVars, p, g)
  if t == cl.t # not a substep
    @. v.prevsol = sol # sol at previous time-step is needed to compute budgets for stochastic forcing
    p.calcF!(v.F, sol, t, cl, v, p, g)
  end
  @. N += v.F
  nothing
end


# ----------------
# Helper functions
# ----------------

"""
    updatevars!(prob)

Update `prob.vars` using `prob.sol`.
"""
function updatevars!(v, g, sol)
  v.zetah .= sol
  @. v.uh =  im * g.l  * g.invKrsq * sol
  @. v.vh = -im * g.kr * g.invKrsq * sol
  ldiv!(v.zeta, g.rfftplan, deepcopy(v.zetah))
  ldiv!(v.u, g.rfftplan, deepcopy(v.uh))
  ldiv!(v.v, g.rfftplan, deepcopy(v.vh))
  nothing
end

updatevars!(prob) = updatevars!(prob.vars, prob.grid, prob.sol)

"""
    set_zeta!(prob, zeta)

Set the `prob.sol` as the transform of `zeta` and update variables.
"""
function set_zeta!(prob, zeta)
  mul!(prob.sol, prob.grid.rfftplan, zeta)
  prob.sol[1, 1] = 0 # zero domain average
  updatevars!(prob)
  nothing
end

"""
    energy(prob)

Returns the domain-averaged kinetic energy in `sol`.
"""
@inline function energy(sol, v, g)
  @. v.uh = g.invKrsq * abs2(sol) # |\hat{zeta}|^2/k^2
  1/(2*g.Lx*g.Ly)*parsevalsum(v.uh, g)
end
energy(prob) = energy(prob.sol, prob.vars, prob.grid)

"""
    enstrophy(prob)

Returns the domain-averaged enstrophy in `sol`.
"""
@inline enstrophy(sol, g) = 1/(2*g.Lx*g.Ly)*parsevalsum2(sol, g)
enstrophy(prob) = enstrophy(prob.sol, prob.grid)

"""
    dissipation(prob)

Returns the domain-averaged dissipation rate. nnu must be >= 1.
"""
@inline function dissipation(sol, v, p, g)
  @. v.uh = g.Krsq^(p.nnu-1) * abs2(sol)
  v.uh[1, 1] = 0
  p.nu/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

dissipation(prob) = dissipation(prob.sol, prob.vars, prob.params, prob.grid)

"""
    work(prob)

Returns the domain-averaged rate of work of energy by the forcing `F`.
"""
@inline function work(sol, v::ForcedVars, g)
  @. v.uh = g.invKrsq * sol * conj(v.F)
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline function work(sol, v::StochasticForcedVars, g)
  @. v.uh = 0.5 * g.invKrsq * (v.prevsol + sol) * conj(v.F) # Stratonovich
  # @. v.uh = g.invKrsq * v.prevsol * conj(v.Fh)            # Ito
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end
work(prob) = work(prob.sol, prob.vars, prob.grid)

"""
    drag(prob)

Returns the extraction of domain-averaged energy by drag/hypodrag `mu`.
"""
@inline function drag(sol, v, p, g)
  @. v.uh = g.Krsq^(p.nmu-1) * abs2(sol)
  v.uh[1, 1] = 0
  p.mu/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end
drag(prob) = drag(prob.sol, prob.vars, prob.params, prob.grid)

end # module
