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
           β = 0.0,
         eta = nothing,
  # Drag and/or hyper-/hypo-viscosity
           ν = 0.0,
          nν = 1,
           μ = 0.0,
  # Timestepper and eqn options
     stepper = "RK4",
      calcFU = nothingfunction,
      calcFq = nothingfunction,
  stochastic = false,
           T = Float64,
         dev = CPU())

  # the grid
    gr = TwoDGrid(dev, nx, Lx, ny, Ly; T=T)
  x, y = gridpoints(gr)

  # topographic PV
  if eta==nothing
     eta = zeros(dev, T, (nx, ny))
  end

  pr = !(typeof(eta)<:ArrayType(dev)) ? Params(gr, f0, β, eta, μ, ν, nν, calcFU, calcFq) : Params(f0, β, eta, rfft(eta), μ, ν, nν, calcFU, calcFq)

  vs = (calcFq == nothingfunction && calcFU == nothingfunction) ? Vars(dev, gr) : (stochastic ? StochasticForcedVars(dev, gr) : ForcedVars(dev, gr))
  
  eq = Equation(pr, gr)
  FourierFlows.Problem(eq, stepper, dt, gr, vs, pr, dev)
end



# ----------
# Parameters
# ----------

"""
    Params(g::TwoDGrid, f0, β, FU, eta, μ, ν, nν, calcFU, calcFq)

Returns the params for an unforced two-dimensional barotropic QG problem.
"""
struct Params{T, Aphys, Atrans} <: AbstractParams
       f0 :: T            # Constant planetary vorticity
        β :: T            # Planetary vorticity y-gradient
      eta :: Aphys        # Topographic PV
     etah :: Atrans       # FFT of Topographic PV
        μ :: T            # Linear drag
        ν :: T            # Viscosity coefficient
       nν :: Int          # Hyperviscous order (nν=1 is plain old viscosity)
   calcFU :: Function     # Function that calculates the forcing F(t) on
                          # domain-averaged zonal flow U(t)
  calcFq! :: Function     # Function that calculates the forcing on QGPV q
end

"""
    Params(g::TwoDGrid, f0, β, eta::Function, μ, ν, nν, calcFU, calcFq)

Constructor for Params that accepts a generating function for the topographic PV.
"""
function Params(g::AbstractGrid{T, A}, f0, β, eta::Function, μ, ν, nν::Int, calcFU, calcFq) where {T, A}
  etagrid = A([eta(g.x[i], g.y[j]) for i=1:g.nx, j=1:g.ny])
     etah = rfft(etagrid)
  Params(f0, β, etagrid, etah, μ, ν, nν, calcFU, calcFq)
end


# ---------
# Equations
# ---------

"""
    Equation(p, g)

Returns the equation for two-dimensional barotropic QG problem with params p and grid g.
"""
function Equation(p::Params, g::AbstractGrid)
  L = @. - p.μ - p.ν*g.Krsq^p.nν + im*p.β*g.kr*g.invKrsq
  L[1, 1] = 0
  FourierFlows.Equation(L, calcN!, g)
end


# ----
# Vars
# ----

struct Vars{Ascalar, Aphys, Atrans, F, P} <: BarotropicQGVars
        U :: Ascalar
        q :: Aphys
     zeta :: Aphys
      psi :: Aphys
        u :: Aphys
        v :: Aphys
       qh :: Atrans
    zetah :: Atrans
     psih :: Atrans
       uh :: Atrans
       vh :: Atrans
      Fqh :: F
  prevsol :: P
end

const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    Vars(dev, g)

Returns the vars for unforced two-dimensional barotropic QG problem on device dev and with grid g.
"""
function Vars(dev::Dev, g::AbstractGrid{T}) where {Dev, T}
  U = ArrayType(dev, T, 0)(undef, ); U[] = 0
  @devzeros Dev T (g.nx, g.ny) q u v psi zeta
  @devzeros Dev Complex{T} (g.nkr, g.nl) qh uh vh psih zetah
  Vars(U, q, zeta, psi, u, v, qh, zetah, psih, uh, vh, nothing, nothing)
end

"""
    ForcedVars(dev, g)

Returns the vars for forced two-dimensional barotropic QG problem on device dev and with grid g.
"""
function ForcedVars(dev::Dev, g::AbstractGrid{T}) where {Dev, T}
  U = ArrayType(dev, T, 0)(undef, ); U[] = 0
  @devzeros Dev T (g.nx, g.ny) q u v psi zeta
  @devzeros Dev Complex{T} (g.nkr, g.nl) qh uh vh psih zetah Fqh
  Vars(U, q, zeta, psi, u, v, qh, zetah, psih, uh, vh, Fqh, nothing)
end

"""
    StochasticForcedVars(dev, g)

Returns the vars for stochastically forced two-dimensional barotropic QG problem on device dev and with grid g.
"""
function StochasticForcedVars(dev::Dev, g::AbstractGrid{T}) where {Dev, T}
  U = ArrayType(dev, T, 0)(undef, ); U[] = 0
  @devzeros Dev T (g.nx, g.ny) q u v psi zeta
  @devzeros Dev Complex{T} (g.nkr, g.nl) qh uh vh psih zetah Fqh prevsol
  Vars(U, q, zeta, psi, u, v, qh, zetah, psih, uh, vh, Fqh, prevsol)
end


# -------
# Solvers
# -------

function calcN_advection!(N, sol, t, cl, v, p, g)
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

function calcN!(N, sol, t, cl, v, p, g)
  calcN_advection!(N, sol, t, cl, v, p, g)
  addforcing!(N, sol, t, cl, v, p, g)
  if p.calcFU != nothingfunction
    # 'Nonlinear' term for U with topographic correlation.
    # Note: < v*eta > = sum( conj(vh)*eta ) / (nx^2*ny^2) if fft is used
    # while < v*eta > = 2*sum( conj(vh)*eta ) / (nx^2*ny^2) if rfft is used
    N[1, 1] = p.calcFU(t) + 2*sum(conj(v.vh).*p.etah).re / (g.nx^2.0*g.ny^2.0)
  end
  nothing
end

addforcing!(N, sol, t, cl, v::Vars, p, g) = nothing

function addforcing!(N, sol, t, cl, v::ForcedVars, p, g)
  p.calcFq!(v.Fqh, sol, t, cl, v, p, g)
  @. N += v.Fqh
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


# ----------------
# Helper functions
# ----------------

"""
    updatevars!(v, s, g)

Update the vars in v on the grid g with the solution in sol.
"""
function updatevars!(sol, v, p, g)
  v.U[] = sol[1, 1].re
  @. v.zetah = sol
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

updatevars!(prob) = updatevars!(prob.sol, prob.vars, prob.params, prob.grid)

"""
    set_zeta!(prob, zeta)
    set_zeta!(s, v, g, zeta)

Set the solution sol as the transform of zeta and update variables v
on the grid g.
"""
function set_zeta!(sol, v::Vars, p, g, zeta)
  mul!(v.zetah, g.rfftplan, zeta)
  v.zetah[1, 1] = 0.0
  @. sol = v.zetah

  updatevars!(sol, v, p, g)
  nothing
end

function set_zeta!(sol, v::Union{ForcedVars, StochasticForcedVars}, p, g, zeta)
  v.U[] = deepcopy(sol[1, 1])
  mul!(v.zetah, g.rfftplan, zeta)
  v.zetah[1, 1] = 0.0
  @. sol = v.zetah
  sol[1, 1] = v.U[]

  updatevars!(sol, v, p, g)
  nothing
end

set_zeta!(prob, zeta) = set_zeta!(prob.sol, prob.vars, prob.params, prob.grid, zeta)

"""
    set_U!(prob, U)
    set_U!(sol, v, g, U)

Set the (kx, ky)=(0, 0) part of solution sol as the domain-average zonal flow U.
"""
function set_U!(sol, v, p, g, U::Float64)
  sol[1, 1] = U
  updatevars!(sol, v, p, g)
  nothing
end

set_U!(prob, U::Float64) = set_U!(prob.sol, prob.vars, prob.params, prob.grid, U)


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
Returns the energy of the domain-averaged U.
"""
meanenergy(prob) = real(0.5*prob.sol[1, 1].^2)

"""
Returns the enstrophy of the domain-averaged U.
"""
meanenstrophy(prob) = real(prob.params.β*prob.sol[1, 1])

"""
    dissipation(prob)
    dissipation(s, v, p, g)

Returns the domain-averaged dissipation rate. nν must be >= 1.
"""
@inline function dissipation(sol, v, p, g)
  @. v.uh = g.Krsq^(p.nν-1) * abs2(sol)
  v.uh[1, 1] = 0
  p.ν/(g.Lx*g.Ly)*parsevalsum(v.uh, g)
end

@inline dissipation(prob) = dissipation(prob.sol, prob.vars, prob.params, prob.grid)

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

Returns the extraction of domain-averaged energy by drag μ.
"""
@inline function drag(sol, v, p, g)
  @. v.uh = g.Krsq^(-1) * abs2(sol)
  v.uh[1, 1] = 0
  p.μ/(g.Lx*g.Ly)*FourierFlows.parsevalsum(v.uh, g)
end

@inline drag(prob) = drag(prob.sol, prob.vars, prob.params, prob.grid)


end # module
