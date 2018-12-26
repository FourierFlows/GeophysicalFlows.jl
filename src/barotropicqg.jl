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
using FourierFlows: getfieldspecs, structvarsexpr, parsevalsum, parsevalsum2

abstract type BarotropicQGVars <: AbstractVars end
abstract type BarotropicQGForcedVars <: BarotropicQGVars end

const realscalarvars = [:U]
const physicalvars = [:q, :zeta, :psi, :u, :v]
const transformvars = [ Symbol(var, :h) for var in physicalvars ]
const forcedvars = [:Fqh]
const stochforcedvars = [:prevsol]

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
            f0 = 1.0,
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
  g  = BarotropicQG.TwoDGrid(nx, Lx, ny, Ly)

  # topographic PV
  if eta==nothing
    eta = 0*g.X
    etah = rfft(eta)
  end

  if typeof(eta)!=Array{Float64,2} #this is true if eta was passes in Problem as a function
    pr = Params(g, f0, beta, eta, mu, nu, nnu, calcFU, calcFq)
  else
    pr = Params(f0, beta, eta, rfft(eta), mu, nu, nnu, calcFU, calcFq)
  end

  if calcFq == nothingfunction && calcFU == nothingfunction
      vs = Vars(g)
  else
      if stochastic ==  false
          vs = ForcedVars(g)
      elseif stochastic == true
          vs = StochasticForcedVars(g)
      end
  end
  eq = BarotropicQG.Equation(pr, g)
  ts = FourierFlows.autoconstructtimestepper(stepper, dt, eq.LC, g)
  FourierFlows.Problem(g, vs, pr, eq, ts)
end

InitialValueProblem(; kwargs...) = Problem(; kwargs...)
ForcedProblem(; kwargs...) = Problem(; kwargs...)


# ----------
# Parameters
# ----------

"""
    Params(g::TwoDGrid, f0, beta, FU, eta, mu, nu, nnu, calcFU, calcFq)

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
  calcFU::Function    # Function that calculates the forcing F(t) on
                      # domain-averaged zonal flow U(t)
  calcFq!::Function   # Function that calculates the forcing on QGPV q
end

"""
    Params(g::TwoDGrid, f0, beta, eta::Function, mu, nu, nnu, calcFU, calcFq)

Constructor for Params that accepts a generating function for the topographic PV.
"""
function Params(g::TwoDGrid, f0, beta, eta::Function, mu, nu, nnu, calcFU, calcFq)
  etagrid = eta(g.X, g.Y)
  etah = rfft(etagrid)
  Params(f0, beta, etagrid, etah, mu, nu, nnu, calcFU, calcFq)
end


# ---------
# Equations
# ---------

"""
    Equation(p, g)

Returns the equation for two-dimensional barotropic QG problem with params p and grid g.
"""
function Equation(p::Params, g; T=typeof(g.Lx))
  LC = @. -p.mu - p.nu*g.Krsq^p.nnu + im*p.beta*g.kr*g.invKrsq
  LC[1, 1] = 0
  FourierFlows.Equation{Complex{T},2}(LC, calcN!)
end


# ----
# Vars
# ----

varspecs = cat(
  getfieldspecs(realscalarvars, :(Array{T,0})),
  getfieldspecs(physicalvars, :(Array{T,2})),
  getfieldspecs(transformvars, :(Array{Complex{T},2})),
  dims=1)

forcedvarspecs = cat(varspecs, getfieldspecs(forcedvars, :(Array{Complex{T},2})), dims=1)
stochforcedvarspecs = cat(forcedvarspecs, getfieldspecs(stochforcedvars, :(Array{Complex{T},2})), dims=1)

# Construct Vars types
eval(structvarsexpr(:Vars, varspecs; parent=:BarotropicQGVars))
eval(structvarsexpr(:ForcedVars, forcedvarspecs; parent=:BarotropicQGForcedVars))
eval(structvarsexpr(:StochasticForcedVars, stochforcedvarspecs; parent=:BarotropicQGForcedVars))

"""
    Vars(g)

Returns the vars for unforced two-dimensional barotropic QG problem with grid g.
"""
function Vars(g; T=typeof(g.Lx))
  U = Array{T,0}(undef, ); U[] = 0
  @createarrays T (g.nx, g.ny) q u v psi zeta
  @createarrays Complex{T} (g.nkr, g.nl) qh uh vh psih zetah
  Vars(U, q, zeta, psi, u, v, qh, zetah, psih, uh, vh)
end

"""
    ForcedVars(g)

Returns the vars for forced two-dimensional barotropic QG problem with grid g.
"""
function ForcedVars(g; T=typeof(g.Lx))
  v = Vars(g; T=T)
  Fqh = zeros(Complex{T}, (g.nkr, g.nl))
  ForcedVars(getfield.(Ref(v), fieldnames(typeof(v)))..., Fqh)
end

"""
    StochasticForcedVars(g)

Returns the vars for stochastically forced two-dimensional barotropic QG problem with grid g.
"""
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
  v.U[] = sol[1, 1].re
  @. v.zetah = sol
  v.zetah[1, 1] = 0

  @. v.uh =  im * g.l  * g.invKrsq * v.zetah
  @. v.vh = -im * g.kr * g.invKrsq * v.zetah

  ldiv!(v.zeta, g.rfftplan, v.zetah)
  ldiv!(v.u, g.rfftplan, v.uh)
  v.psih .= v.vh # FFTW's irfft destroys its input; v.vh is needed for N[1, 1]
  ldiv!(v.v, g.rfftplan, v.psih)

  @. v.q = v.zeta + p.eta
  @. v.u = (v.U[] + v.u)*v.q # (U+u)*q
  @. v.v = v.v*v.q # v*q

  mul!(v.uh, g.rfftplan, v.u) # \hat{(u+U)*q}
  # Nonlinear advection term for q (part 1)
  @. N = -im*g.kr*v.uh # -∂[(U+u)q]/∂x
  mul!(v.uh, g.rfftplan, v.v) # \hat{v*q}
  @. N += - im*g.l*v.uh # -∂[vq]/∂y
end

function calcN!(N, sol, t, s, v, p, g)
  calcN_advection!(N, sol, t, s, v, p, g)
  addforcing!(N, t, s, v, p, g)
  if p.calcFU != nothingfunction
    # 'Nonlinear' term for U with topographic correlation.
    # Note: < v*eta > = sum( conj(vh)*eta ) / (nx^2*ny^2) if fft is used
    # while < v*eta > = 2*sum( conj(vh)*eta ) / (nx^2*ny^2) if rfft is used
    N[1, 1] = p.calcFU(t) + 2*sum(conj(v.vh).*p.etah).re / (g.nx^2.0*g.ny^2.0)
  end
  nothing
end

addforcing!(N, t, s, v::Vars, p, g) = nothing

function addforcing!(N, t, s, v::ForcedVars, p, g)
  p.calcFq!(v.Fqh, t, s, v, p, g)
  @. N += v.Fqh
  nothing
end

function addforcing!(N, t, s, v::StochasticForcedVars, p, g)
  if t == s.t # not a substep
    @. v.prevsol = s.sol # sol at previous time-step is needed to compute budgets for stochastic forcing
    p.calcFq!(v.Fqh, t, s, v, p, g)
  end
  @. N += v.Fqh
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
  v.U[] = s.sol[1, 1].re
  @. v.zetah = s.sol
  v.zetah[1, 1] = 0.0

  @. v.psih = -v.zetah * g.invKrsq
  @. v.uh = -im * g.l  * v.psih
  @. v.vh =  im * g.kr * v.psih

  ldiv!(v.zeta, g.rfftplan, deepcopy(v.zetah))
  ldiv!(v.psi, g.rfftplan, deepcopy(v.psih))
  ldiv!(v.u, g.rfftplan, deepcopy(v.uh))
  ldiv!(v.v, g.rfftplan, deepcopy(v.vh))

  @. v.q = v.zeta + p.eta
  nothing
end

updatevars!(prob) = updatevars!(prob.state, prob.vars, prob.params, prob.grid)

"""
    set_zeta!(prob, zeta)
    set_zeta!(s, v, g, zeta)

Set the solution s.sol as the transform of zeta and update variables v
on the grid g.
"""
function set_zeta!(s, v::Vars, p, g, zeta)
  mul!(v.zetah, g.rfftplan, zeta)
  v.zetah[1, 1] = 0.0
  @. s.sol = v.zetah

  updatevars!(s, v, p, g)
  nothing
end

function set_zeta!(s, v::BarotropicQGForcedVars, p, g, zeta)
  v.U[] = deepcopy(s.sol[1, 1])
  mul!(v.zetah, g.rfftplan, zeta)
  v.zetah[1, 1] = 0.0
  @. s.sol = v.zetah
  s.sol[1, 1] = v.U[]

  updatevars!(s, v, p, g)
  nothing
end

set_zeta!(prob, zeta) = set_zeta!(prob.state, prob.vars, prob.params, prob.grid, zeta)

"""
    set_U!(prob, U)
    set_U!(s, v, g, U)

Set the (kx, ky)=(0, 0) part of solution s.sol as the domain-average zonal flow U.
"""
function set_U!(s, v, p, g, U::Float64)
  s.sol[1, 1] = U
  updatevars!(s, v, p, g)
  nothing
end

set_U!(prob, U::Float64) = set_U!(prob.state, prob.vars, prob.params, prob.grid, U)


"""
Calculate the domain-averaged kinetic energy.
"""
function energy(prob)
  s, g = prob.state, prob.grid
  0.5*(parsevalsum2(g.Kr.*g.invKrsq.*s.sol, g)
        + parsevalsum2(g.Lr.*g.invKrsq.*s.sol, g))/(g.Lx*g.Ly)
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
Returns the energy of the domain-averaged U.
"""
meanenergy(prob) = real(0.5*prob.state.sol[1, 1].^2)

"""
Returns the enstrophy of the domain-averaged U.
"""
meanenstrophy(prob) = real(prob.params.beta*prob.state.sol[1, 1])

"""
    dissipation(prob)
    dissipation(s, v, p, g)

Returns the domain-averaged dissipation rate. nnu must be >= 1.
"""
@inline function dissipation(s, v, p, g)
  @. v.uh = g.Krsq^(p.nnu-1) * abs2(s.sol)
  v.uh[1, 1] = 0
  p.nu/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline dissipation(prob) = dissipation(prob.state, prob.vars, prob.params, prob.grid)

"""
    work(prob)
    work(s, v, p, g)

Returns the domain-averaged rate of work of energy by the forcing Fqh.
"""
@inline function work(s, v::ForcedVars, g)
  @. v.uh = g.invKrsq * s.sol * conj(v.Fqh)
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline function work(s, v::StochasticForcedVars, g)
  @. v.uh = g.invKrsq * (v.prevsol + s.sol)/2.0 * conj(v.Fqh) # Stratonovich
  # @. v.uh = g.invKrsq * v.prevsol * conj(v.Fqh)             # Ito
  1/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline work(prob) = work(prob.state, prob.vars, prob.grid)

"""
    drag(prob)
    drag(s, v, p, g)

Returns the extraction of domain-averaged energy by drag mu.
"""
@inline function drag(s, v, p, g)
  @. v.uh = g.Krsq^(-1) * abs2(s.sol)
  v.uh[1, 1] = 0
  p.mu/(g.Lx*g.Ly)*FourierFlows.parsevalsum(v.uh, g)
end

@inline drag(prob) = drag(prob.state, prob.vars, prob.params, prob.grid)


end # module
