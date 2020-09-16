module SurfaceQG

export
  Problem,
  set_b!,
  updatevars!,

  kinetic_energy,
  buoyancy_variance,
  buoyancy_dissipation,
  buoyancy_work,
  buoyancy_advection,
  kinetic_energy_advection

  using
    CUDA,
    Reexport

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: parsevalsum

nothingfunction(args...) = nothing

"""
    Problem(; parameters...)

Construct a Surface QG turbulence problem.
"""
function Problem(dev::Device=CPU();
  # Numerical parameters
          nx = 256,
          Lx = 2π,
          ny = nx,
          Ly = Lx,
          dt = 0.01,
  # Hyper-viscosity parameters
           ν = 0,
          nν = 1,
  # Timestepper and equation options
     stepper = "RK4",
       calcF = nothingfunction,
  stochastic = false,
           T = Float64)

  grid = TwoDGrid(dev, nx, Lx, ny, Ly; T=T)

  params = Params{T}(ν, nν, calcF)

  vars = calcF == nothingfunction ? Vars(dev, grid) : (stochastic ? StochasticForcedVars(dev, grid) : ForcedVars(dev, grid))

  equation = Equation(params, grid)

  return FourierFlows.Problem(equation, stepper, dt, grid, vars, params, dev)
end


# ----------
# Parameters
# ----------

"""
    Params(ν, nν, calcF!)

Returns the params for Surface QG turbulence.
"""
struct Params{T} <: AbstractParams
       ν :: T         # Buoyancy viscosity coefficient
      nν :: Int       # Buoyancy hyperviscous order
  calcF! :: Function  # Calculates the Fourier transform of the buoyancy forcing `Fh`
end

Params(ν, nν) = Params(ν, nν, nothingfunction)


# ---------
# Equations
# ---------

"""
    Equation(params, grid)

Returns the equation for Surface QG turbulence with `params` and `grid`.
"""
function Equation(params::Params, grid::AbstractGrid)
  L = @. - params.ν * grid.Krsq^params.nν
  CUDA.@allowscalar L[1, 1] = 0
  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

abstract type SurfaceQGVars <: AbstractVars end

struct Vars{Aphys, Atrans, F, P} <: SurfaceQGVars
        b :: Aphys
        u :: Aphys
        v :: Aphys
       bh :: Atrans
       uh :: Atrans
       vh :: Atrans
       Fh :: F
  prevsol :: P
end

const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    Vars(dev, grid)

Returns the vars for unforced surface QG turbulence on device dev and with `grid`.
"""
function Vars(::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) b u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) bh uh vh
  return Vars(b, u, v, bh, uh, vh, nothing, nothing)
end

"""
    ForcedVars(dev, grid)

Returns the vars for forced surface QG turbulence on device `dev` and with
`grid`.
"""
function ForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) b u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) bh uh vh Fh
  return Vars(b, u, v, bh, uh, vh, Fh, nothing)
end

"""
    StochasticForcedVars(dev, grid)

Returns the `vars` for stochastically forced surface QG turbulence on device
`dev` and with `grid`.
"""
function StochasticForcedVars(dev::Dev, grid::AbstractGrid) where Dev
  T = eltype(grid)
  @devzeros Dev T (grid.nx, grid.ny) b u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) bh uh vh Fh prevsol
  return Vars(b, u, v, bh, uh, vh, Fh, prevsol)
end


# -------
# Solvers
# -------

"""
    calcN_advection(N, sol, t, clock, vars, params, grid)

Calculates the advection term.
"""
function calcN_advection!(N, sol, t, clock, vars, params, grid)
  @. vars.bh = sol
  @. vars.uh =   im * grid.l  * sqrt(grid.invKrsq) * sol
  @. vars.vh = - im * grid.kr * sqrt(grid.invKrsq) * sol

  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)
  ldiv!(vars.b, grid.rfftplan, vars.bh)

  @. vars.u *= vars.b # u*b
  @. vars.v *= vars.b # v*b

  mul!(vars.uh, grid.rfftplan, vars.u) # \hat{u*b}
  mul!(vars.vh, grid.rfftplan, vars.v) # \hat{v*b}

  @. N = - im * grid.kr * vars.uh - im * grid.l * vars.vh
  return nothing
end

function calcN!(N, sol, t, clock, vars, params, grid)
  calcN_advection!(N, sol, t, clock, vars, params, grid)
  addforcing!(N, sol, t, clock, vars, params, grid)
  return nothing
end

addforcing!(N, sol, t, clock, vars::Vars, params, grid) = nothing

function addforcing!(N, sol, t, clock, vars::ForcedVars, params, grid)
  params.calcF!(vars.Fh, sol, t, clock, vars, params, grid)
  @. N += vars.Fh
  return nothing
end

function addforcing!(N, sol, t, clock, vars::StochasticForcedVars, params, grid)
  if t == clock.t # not a substep
    @. vars.prevsol = sol # sol at previous time-step is needed to compute budgets for stochastic forcing
    params.calcF!(vars.Fh, sol, t, clock, vars, params, grid)
  end
  @. N += vars.Fh
  return nothing
end


# ----------------
# Helper functions
# ----------------

"""
    updatevars!(prob)

Update variables in `vars` with solution in `sol`.
"""
function updatevars!(prob)
  vars, grid, sol = prob.vars, prob.grid, prob.sol
  @. vars.bh = sol
  @. vars.uh =   im * grid.l  * sqrt(grid.invKrsq) * sol
  @. vars.vh = - im * grid.kr * sqrt(grid.invKrsq) * sol
  ldiv!(vars.b, grid.rfftplan, deepcopy(vars.bh))
  ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))
  ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh))
  return nothing
end

"""
    set_b!(prob, b)

Set the solution `sol` as the transform of b and update all variables.
"""
function set_b!(prob, b)
  mul!(prob.sol, prob.grid.rfftplan, b)
  CUDA.@allowscalar prob.sol[1, 1] = 0 # zero domain average
  updatevars!(prob)
  return nothing
end

"""
    kinetic_energy(prob)

Returns the domain-averaged surface kinetic energy. In SQG, this is
identical to half the domain-averaged surface buoyancy variance.
"""
@inline function kinetic_energy(prob)
  sol, vars, grid = prob.sol, prob.vars, prob.grid
  @. vars.uh =   im * grid.l  * sqrt(grid.invKrsq) * sol
  @. vars.vh = - im * grid.kr * sqrt(grid.invKrsq) * sol
  @. vars.bh = 0.5 * (abs2(vars.uh) + abs2(vars.vh)) # ½(|û|²+|v̂|²)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.bh, grid)
end

"""
    buoyancy_variance(prob)

Returns the domain-averaged buoyancy variance. In SQG flows this is identical to
the domain-averaged velocity variance (twice the kinetic energy)
"""
@inline function buoyancy_variance(prob)
  sol, grid = prob.sol, prob.grid
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(abs2.(sol), grid)
end

"""
    buoyancy_dissipation(prob)

Returns the domain-averaged dissipation rate of surface buoyancy variance due
to small scale diffusion/viscosity. nν must be >= 1.

In SQG, this is identical to twice the rate of kinetic energy dissipation
"""
@inline function buoyancy_dissipation(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  @. vars.uh = 2 * grid.Krsq^params.nν * abs2(sol)
  CUDA.@allowscalar vars.uh[1, 1] = 0
  return params.ν / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

"""
    buoyancy_work(prob)
    buoyancy_work(sol, vars, grid)

Returns the domain-averaged rate of work of buoyancy variance by the forcing Fh.
"""
@inline function buoyancy_work(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  @. vars.uh =  sol * conj(vars.Fh)    # b̂*conj(f̂)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline function buoyancy_work(sol, vars::StochasticForcedVars, grid)
  @. vars.uh =  (vars.prevsol + sol) / 2 * conj(vars.Fh) # Stratonovich
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.uh, grid)
end

@inline work(prob) = work(prob.sol, prob.vars, prob.grid)

"""
    buoyancy_advection(prob)

Returns the average of domain-averaged advection of buoyancy variance. This should
be zero if buoyancy variance is conserved.
"""
@inline function buoyancy_advection(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid

  # Diagnose all the neccessary variables from sol
  @. vars.bh = sol
  @. vars.uh =   im * grid.l  * sqrt(grid.invKrsq) * sol
  @. vars.vh = - im * grid.kr * sqrt(grid.invKrsq) * sol
  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)
  ldiv!(vars.b, grid.rfftplan, vars.bh)

  # Diagnose fluxes
  @. vars.u *= vars.b # u*b
  @. vars.v *= vars.b # v*b

  mul!(vars.uh, grid.rfftplan, vars.u) # \hat{u*b}
  mul!(vars.vh, grid.rfftplan, vars.v) # \hat{v*b}

  @. vars.bh = (- im * grid.kr * vars.uh - im * grid.l * vars.vh) * conj(vars.bh)
  # vars.bh is -conj(FFT[b])⋅FFT[u̲⋅∇(b)]

  return 1 / (grid.Lx * grid.Ly) * parsevalsum(vars.bh, grid)
end

"""
    kinetic_energy_advection(prob)

Returns the average of domain-averaged advection of kinetic energy by the
leading-order (geostrophic) flow.
"""
@inline function kinetic_energy_advection(prob)
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid

  # Diagnose all the neccessary variables from sol
  @. vars.uh =   im * grid.l  * sqrt(grid.invKrsq) * sol
  @. vars.vh = - im * grid.kr * sqrt(grid.invKrsq) * sol
  ldiv!(vars.u, grid.rfftplan, vars.uh)
  ldiv!(vars.v, grid.rfftplan, vars.vh)

  # The goal is to calculate -conj(FFT[u̲])⋅FFT[u̲⋅∇(u̲)] by summing its components

  # Diagnose Fluxes - initially vars.vh is used to store velocity correlations
  mul!(vars.vh, grid.rfftplan, vars.u .* vars.u)  # Transform adv. of u * u
  @. vars.bh = - im * grid.kr * vars.vh # -FFT(∂[uu]/∂x), bh will be advection
  @. vars.bh *= conj(vars.uh)         # -FFT(∂[uu]/∂x) * conj(FFT(u))
  mul!(vars.vh, grid.rfftplan, vars.u .* vars.v) # Transform adv. of u * v
  # add -FFT(∂[uv]/∂y) * conj(FFT(u))
  @. vars.bh -= im * grid.l  * vars.vh * conj(vars.uh)

  # Re-calculate current value of vars.vh, now vars.uh is storage variable
  @. vars.vh = - im * grid.kr * sqrt(grid.invKrsq) * sol
  mul!(vars.uh, grid.rfftplan, vars.u .* vars.v) # Transform adv. of u * v
  # add -FFT(∂[uv]/∂x) * conj(FFT(v))
  @. vars.bh -= im * grid.kr * vars.uh * conj(vars.vh)
  mul!(vars.uh, grid.rfftplan, vars.v .* vars.v) # Transform adv. of v * v
  # add -FFT(∂[vv]/∂y) * conj(FFT(v))
  @. vars.bh -= im * grid.l * vars.uh * conj(vars.vh)
  # vars.bh is now -conj(FFT[u̲])⋅FFT[u̲⋅∇(u̲)]

  return 1 / (grid.Lx * grid.Ly) * parsevalsum( vars.bh , grid)
end

end # module
