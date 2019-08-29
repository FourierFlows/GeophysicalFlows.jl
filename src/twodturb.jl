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
           T = Float64,
         dev = CPU()
)

  gr = TwoDGrid(dev, nx, Lx, ny, Ly)
  pr = Params{T}(nu, nnu, mu, nmu, calcF)
  vs = calcF == nothingfunction ? Vars(dev, gr) : (stochastic ? StochasticForcedVars(dev, gr) : ForcedVars(dev, gr))
  eq = Equation(pr, gr)
  FourierFlows.Problem(eq, stepper, dt, gr, vs, pr, dev)
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

struct Vars{Aphys, Atrans} <: TwoDTurbVars
   zeta :: Aphys
      u :: Aphys
      v :: Aphys
  zetah :: Atrans
     uh :: Atrans
     vh :: Atrans
end

struct ForcedVars{Aphys, Atrans} <: TwoDTurbVars
   zeta :: Aphys
      u :: Aphys
      v :: Aphys
  zetah :: Atrans
     uh :: Atrans
     vh :: Atrans
     Fh :: Atrans
end

struct StochasticForcedVars{Aphys, Atrans} <: TwoDTurbVars
     zeta :: Aphys
        u :: Aphys
        v :: Aphys
    zetah :: Atrans
       uh :: Atrans
       vh :: Atrans
       Fh :: Atrans
  prevsol :: Atrans
end


"""
    Vars(dev, g)

Returns the vars for unforced two-dimensional turbulence on device dev and with 
  grid g.
"""
function Vars(::Dev, g::AbstractGrid{T}) where {Dev, T}
  @devzeros Dev T (g.nx, g.ny) zeta u v
  @devzeros Dev Complex{T} (g.nkr, g.nl) zetah uh vh
  Vars(zeta, u, v, zetah, uh, vh)
end

"""
    ForcedVars(dev, g)

Returns the vars for forced two-dimensional turbulence on device dev and with 
  grid g.
"""
function ForcedVars(dev::Dev, g::AbstractGrid{T}) where {Dev, T}
  v = Vars(dev, g)
  @devzeros Dev Complex{T} (g.nkr, g.nl) Fh
  ForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., Fh)
end

"""
    StochasticForcedVars(dev, g)

Returns the vars for stochastically forced two-dimensional turbulence on device
  dev and with grid g.
"""
function StochasticForcedVars(dev::Dev, g::AbstractGrid{T}) where {Dev, T}
  v = ForcedVars(dev, g)
  @devzeros Dev Complex{T} (g.nkr, g.nl) prevsol
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
  @. v.uh = g.invKrsq * (v.prevsol + sol)/2 * conj(v.Fh) # Stratonovich
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
