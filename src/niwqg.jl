module NIWQG

export
  Problem,
  set_q!,
  set_phi!,
  set_planewave!,
  updatevars!,

  waveaction,
  qgke,
  wavepe,
  coupledenergy
  
using 
  Reexport,
  FFTW

@reexport using FourierFlows

using LinearAlgebra: mul!, ldiv!
using FourierFlows: getfieldspecs, structvarsexpr, parsevalsum, parsevalsum2

abstract type NIWQGVars <: AbstractVars end

const physicalvarsr = [:q, :U, :V, :zeta, :psi, :Uq, :Vq, :phijac, :phisq]
const physicalvarsc = [:phi, :Uphi, :Vphi, :zetaphi, :phix, :phiy] 
const transformvarsr = [ Symbol(var, :h) for var in physicalvarsr ]
const transformvarsc = [ Symbol(var, :h) for var in physicalvarsc ]
const forcedvars = [:Fr, :Fc]
const stochforcedvars = [:prevsolr, :prevsolc]

nothingfunction(args...) = nothing
onefunction(args...) = 1

# --
# Problem
# --

"""
    Problem(; parameters...)

Construct a VerticallyFourierBoussinesq initial value problem.
"""
function Problem(;
  # Numerical parameters
          nx = 64,
          Lx = 2π,
          ny = nx,
          Ly = Lx,
          dt = 1e-9,
  # Drag and/or hyper-/hypo-viscosity
          nu = 0, # wave (hyper-)viscosity
         nnu = 1, # wave (hyper-)viscous order (1=Laplacian)
         kap = 0, # PV (hyper-)visosity
        nkap = 1, # PV (hyper-)viscous order
         muq = 0, # drag/arbitrary-order dissipation for q
        nmuq = 0, # order of 2nd dissipation term for q
         muw = 0, # drag/arbitrary-order dissipation for w
        nmuw = 0, # order of 2nd dissipation term for w
  # Physical parameters
         eta = 0, # dispersivity: eta = N^2 / f m^2
           f = 1, # inertial frequency
  # Optional uniform and steady background flow
          Ub = 0,
          Vb = 0,
  # Timestepper and eqn options
     stepper = "RK4",
       calcF = nothingfunction,
  stochastic = false,
    nthreads = Sys.CPU_THREADS,
           T = Float64,
     dealias = false
  )

  gd = TwoDGrid(nx, Lx, ny, Ly; T=T)
  pr = Params{T}(nu, nnu, kap, nkap, muw, nmuw, muq, nmuq, eta, f, 1/f, Ub, Vb, calcF)
  vs = calcF == nothingfunction ? Vars(gd) : (stochastic ? StochasticForcedVars(gd) : ForcedVars(gd))
  eq = Equation(pr, gd; dealias=dealias)
  ts = TimeStepper(stepper, dt, eq.LCc, eq.LCr, gd)

  FourierFlows.Problem(gd, vs, pr, eq, ts)
end


# --
# Params
# --

"""
    Params(nu, nnu, kap, nkap, muq, nmuq, muw, nmuw, eta, f, invf, Ub, Vb, calcF!)

Construct parameters for an NIWQG problem.
"""
struct Params{T} <: AbstractParams
  nu::T             # Wave viscosity
  nnu::Int          # Wave hyperviscous order
  kap::T            # PV 'diffusivity'
  nkap::Int         # PV hyperdiffusive order
  muw::T            # Wave drag / hypoviscosityty
  nmuw::Int         # Wave drag / hypoviscous orderer
  muq::T            # PV drag / hypoviscosity
  nmuq::Int         # PV drag / hypoviscous order
  eta::T            # Dispersivity
  f::T              # Planetary vorticity
  invf::T           # 1/f
  Ub::T             # Steady barotropic background ('mean') x-velocity
  Vb::T             # Steady barotropic background ('mean') y-velocity
  calcF!::Function  # Forcing
end

# ---------
# Equations
# ---------

function Equation(p, g; T=typeof(g.Lx), dealias=false)
  LCc = @. - p.nu  * g.KKsq  ^ p.nnu   -  p.muw * g.KKsq  ^ p.nmuw  -  0.5*im*p.eta*g.KKsq
  LCr = @. - p.kap * g.KKrsq ^ p.nkap  -  p.muq * g.KKrsq ^ p.nmuq
  LCr[1, 1] = 0
  LCc[1 ,1] = 0

  (dealias ? 
    FourierFlows.DualEquation{Complex{T},2,2}(LCc, LCr, calcN_dealiased!) :
    FourierFlows.DualEquation{Complex{T},2,2}(LCc, LCr, calcN!)
   )
end


# --
# Vars
# --

varspecs = cat(
  getfieldspecs(physicalvarsr, :(Array{T,2})),
  getfieldspecs(physicalvarsc, :(Array{Complex{T},2})),
  getfieldspecs(transformvarsr, :(Array{Complex{T},2})),
  getfieldspecs(transformvarsc, :(Array{Complex{T},2})),
  dims=1)

forcedvarspecs = cat(varspecs, getfieldspecs(forcedvars, :(Array{Complex{T},2})), dims=1)
stochforcedvarspecs = cat(forcedvarspecs, getfieldspecs(stochforcedvars, :(Array{Complex{T},2})), dims=1)

eval(structvarsexpr(:Vars, varspecs; parent=:NIWQGVars))
eval(structvarsexpr(:ForcedVars, forcedvarspecs; parent=:NIWQGVars))
eval(structvarsexpr(:StochasticForcedVars, stochforcedvarspecs; parent=:NIWQGVars))

function Vars(g; T=typeof(g.Lx))
  @createarrays T (g.nx, g.ny) q U V zeta psi Uq Vq phijac phisq
  @createarrays Complex{T} (g.nkr, g.nl) qh Uh Vh zetah psih Uqh Vqh phijach phisqh
  @createarrays Complex{T} (g.nx, g.ny) phi Uphi Vphi zetaphi phix phiy
  @createarrays Complex{T} (g.nk, g.nl) phih Uphih Vphih zetaphih phixh phiyh
  Vars{T}(
    q, U, V, zeta, psi, Uq, Vq, phijac, phisq,
    phi, Uphi, Vphi, zetaphi, phix, phiy,
    qh, Uh, Vh, zetah, psih, Uqh, Vqh, phijach, phisqh,
    phih, Uphih, Vphih, zetaphih, phixh, phiyh
  )
end

function ForcedVars(g; T=typeof(g.Lx))
  v = Vars(g; T=T)
  Fr = zeros(Complex{T}, (g.nkr, g.nl))
  Fc = zeros(Complex{T}, (g.nk, g.nl))
  ForcedVars{T}(getfield.(Ref(v), fieldnames(typeof(v)))..., Fr, Fc)
end

function StochasticForcedVars(g; T=typeof(g.Lx))
  v = ForcedVars(g; T=T)
  prevsolr = zeros(Complex{T}, (g.nkr, g.nl))
  prevsolc = zeros(Complex{T}, (g.nk, g.nl))
  ForcedVars{T}(getfield.(Ref(v), fieldnames(typeof(v)))..., prevsolr, prevsolc)
end


# --
# Solver routines
# --

function calczetah!(zetah, qh, phih, phi, v, p, g)
  # Calc qw
  @. v.phixh = im*g.k*phih
  @. v.phiyh = im*g.l*phih

  ldiv!(v.phix, g.fftplan, v.phixh)
  ldiv!(v.phiy, g.fftplan, v.phiyh)

  @. v.phijac = real(im*(conj(v.phix)*v.phiy - conj(v.phiy)*v.phix))
  @. v.phisq = abs2(phi)

  mul!(v.phijach, g.rfftplan, v.phijac)
  mul!(v.phisqh, g.rfftplan, v.phisq)

  #   zeta = q  -               *** q^w ***
  @. zetah = qh - p.invf*(-0.25*g.KKrsq*v.phisqh + 0.5*v.phijach)
  zetah[1, 1] = 0

  nothing
end

function calcUhVh!(Uh, Vh, psih, Ub, Vb, g)
  @. Uh = - im * g.l  * psih # U = - ∂y ψ
  @. Vh =   im * g.kr * psih # V = + ∂x ψ
  Uh[1, 1] += Ub*g.nx*g.ny
  Vh[1, 1] += Vb*g.nx*g.ny
  nothing
end

function calcadvectionterms!(Uq, Vq, Uphi, Vphi, U, V, q, phi)
  @. Uq = U*q
  @. Vq = V*q
  @. Uphi = U*phi
  @. Vphi = V*phi
  nothing
end

function calcN!(Nc, Nr, solc, solr, t, s, v, p, g)
  @. v.qh = solr # inverse transforms may destroy solr
  @. v.phih = solc # this copy may not be necessary, but clarifies code

  ldiv!(v.phi, g.fftplan, solc)
  calczetah!(v.zetah, solr, solc, v.phi, v, p, g)

  @. v.psih = -g.invKKrsq*v.zetah
  calcUhVh!(v.Uh, v.Vh, v.psih, p.Ub, p.Vb, g)

  # inverse transforms may destroy v.Uh, v.Vh, v.zetah, v.qh
  ldiv!(v.U, g.rfftplan, v.Uh)
  ldiv!(v.V, g.rfftplan, v.Vh)
  ldiv!(v.zeta, g.rfftplan, v.zetah)
  ldiv!(v.q, g.rfftplan, v.qh)

  calcadvectionterms!(v.Uq, v.Vq, v.Uphi, v.Vphi, v.U, v.V, v.q, v.phi)
  @. v.zetaphi = v.zeta*v.phi

  mul!(v.Uphih, g.fftplan, v.Uphi)
  mul!(v.Vphih, g.fftplan, v.Vphi)
  mul!(v.Uqh, g.rfftplan, v.Uq)
  mul!(v.Vqh, g.rfftplan, v.Vq)
  mul!(v.zetaphih, g.fftplan, v.zetaphi)

  @. Nc = - im*g.k*v.Uphih - im*g.l*v.Vphih - 0.5*im*v.zetaphih
  @. Nr = - im*g.kr*v.Uqh - im*g.l*v.Vqh

  addforcing!(Nc, Nr, t, s, v, p, g)

  nothing
end

function calcN_dealiased!(Nc, Nr, solc, solr, t, s, v, p, g)
  calcN!(Nc, Nr, solc, solr, t, s, v, p, g)
  dealias!(Nc, g, g.ialias)
  dealias!(Nr, g, g.iralias)
  nothing
end

addforcing!(Nc, Nr, t, s, v::Vars, p, g) = nothing

function addforcing!(Nc, Nr, t, s, v::ForcedVars, p, g)
  p.calcF!(v.Fc, v.Fr, t, s, v, p, g)
  @. Nc += v.Fc
  @. Nr += v.Fr
  nothing
end

function addforcing!(Nc, Nr, t, s, v::StochasticForcedVars, p, g)
  if t == s.t # not a substep
    @. v.prevsolc = s.solc
    @. v.prevsolr = s.solr
    p.calcF!(v.Fc, v.Fr, t, s, v, p, g)
  end
  @. Nc += v.Fc
  @. Nr += v.Fr
  nothing
end

# --
# Helper functions
# --

"""
    updatevars!(prob)

Update variables in `prob.vars` using the solution in `prob.state.solr` and prob.state.solc`.
"""
function updatevars!(v, s, p, g)
  @. v.qh .= s.solr
  @. v.phih .= s.solc

  ldiv!(v.phi, g.fftplan, v.phih)
  calczetah!(v.zetah, s.solr, s.solc, v.phi, v, p, g)

  @. v.psih = -g.invKKrsq*v.zetah
  calcUhVh!(v.Uh, v.Vh, v.psih, p.Ub, p.Vb, g)

  # Copy variables that are destroyed by inverse transforms
  psih1 = deepcopy(v.psih)
    qh1 = deepcopy(v.qh)
    Uh1 = deepcopy(v.Uh)
    Vh1 = deepcopy(v.Vh)

  ldiv!(v.psi, g.rfftplan, psih1)
  ldiv!(v.q, g.rfftplan, qh1)
  ldiv!(v.U, g.rfftplan, Uh1)
  ldiv!(v.V, g.rfftplan, Vh1)

  ldiv!(v.phi, g.fftplan, v.phih)

  nothing
end
updatevars!(prob) = updatevars!(prob.vars, prob.state, prob.params, prob.grid)

"""
    set_q!(prob, q)

Set potential vorticity and update variables.
"""
function set_q!(s, v, p, g, q)
  @. v.q = q
  mul!(s.solr, g.rfftplan, v.q)
  updatevars!(v, s, p, g)
  nothing
end
set_q!(prob, q) = set_q!(prob.state, prob.vars, prob.params, prob.grid, q)

"""
    set_phi!(prob, phi)

Set the wave field amplitude, phi.
"""
function set_phi!(s, vs, pr, g, phi)
  @. vs.phi = phi
  mul!(s.solc, g.fftplan, vs.phi)
  updatevars!(vs, s, pr, g)
  nothing
end
set_phi!(prob, phi) = set_phi!(prob.state, prob.vars, prob.params, prob.grid, phi)

"""
    set_planewave!(prob, uw, nkw, θ=0; kwargs...)

Set plane wave solution in `prob` with initial speed `uw` and non-dimensional wave
number `nkw`. Keyword argument `envelope=env(x, y)` multiplies the plane wave by `env(x, y)`
"""
function set_planewave!(s, vs, pr, g, uw, nkw, θ=0; envelope=onefunction)
  k = 2π/g.Lx*round(Int, nkw*cos(θ))
  l = 2π/g.Lx*round(Int, nkw*sin(θ))
  x, y = g.X, g.Y

  Φ = @. k*x + l*y
  phi = @. uw * exp(im*Φ) * envelope(x, y)
  set_phi!(s, vs, pr, g, phi)
  nothing
end

set_planewave!(prob, uw, nkw, args...; kwargs...) = set_planewave!(
  prob.state, prob.vars, prob.params, prob.grid, uw, nkw, args...; kwargs...)

# -----------
# Diagnostics
# -----------

"""
    waveaction(prob)

Returns the domain-averaged near-inertial action.
"""
waveaction(prob) = waveaction(prob.state, prob.params, prob.grid)

waveaction(s, p, g) = 0.5*p.invf*parsevalsum2(s.solc, g) / (g.Lx*g.Ly)

"""
    qgke(prob)

Returns the QG kinetic energy.
"""
qgke(prob) = qgke(prob.state, prob.vars, prob.params, prob.grid)

function qgke(s, v, p, g)
  @. v.qh = s.solr
  ldiv!(v.phi, g.fftplan, s.solc)
  calczetah!(v.zetah, v.qh, s.solc, v.phi, v, p, g)
  @. v.Uh = g.invKKrsq * abs2(v.zetah)
  1/(g.Lx*g.Ly) * 0.5 * parsevalsum(v.Uh, g)
end

"""
    wavepe(prob)

Returns the potential energ of the near-inertial waves.
"""
wavepe(prob) = wavepe(prob.state.solc, prob.vars, prob.params, prob.grid)

function wavepe(phih, v, p, g)
  @. v.phixh = g.k*phih
  @. v.phiyh = g.l*phih
  1/(g.Lx*g.Ly) * 0.25*p.eta*p.invf * (parsevalsum2(v.phixh, g) + parsevalsum2(v.phiyh, g))
end

"""
    coupledenergy(prob)

Returns the 'coupled energy' in the NIW-QG flow.
"""
coupledenergy(prob) = qgke(prob) + wavepe(prob)

#=
"""
    mode1ke(prob)

Returns the domain-averaged kinetic energy in the first mode.
"""

"""
    mode1pe(prob)

Returns the domain-averaged potential energy in the first mode.
"""
@inline mode1pe(s, p, g) = @views p.m^2/(g.Lx*g.Ly*p.N^2)*parsevalsum2(s.solc[:, :, 3], g)
@inline mode1pe(prob) = mode1pe(prob.state, prob.params, prob.grid)

"""
    mode1energy(prob)

Returns the domain-averaged total energy in the first mode.
"""
mode1energy(s, p, g) = mode1ke(s, g) + mode1pe(s, p, g)
mode1energy(prob) = mode1energy(prob.state, prob.params, prob.grid)

"""
    mode0dissipation(prob)

Returns the domain-averaged kinetic energy dissipation of the zeroth mode.
"""
@inline function mode0dissipation(s, v, p, g)
  @. v.Uh = g.KKrsq^(p.nnu0-1) * abs2(s.solr)
  p.nu0/(g.Lx*g.Ly)*parsevalsum(v.Uh, g)
end
@inline mode0dissipation(prob) = mode0dissipation(prob.state, prob.vars, prob.params, prob.grid)

"""
    mode0drag(prob)

Returns the extraction of domain-averaged energy extraction by the drag μ.
"""
@inline function mode0drag(s, v, p, g)
  @. v.Uh = g.KKrsq^(p.nmu0-1) * abs2(s.solr)
  @. v.Uh[1, 1] = 0
  p.mu0/(g.Lx*g.Ly)*parsevalsum(v.Uh, g)
end
@inline mode0drag(prob) = mode0drag(prob.state, prob.vars, prob.params, prob.grid)

"""
    mode1dissipation(prob)

Returns the domain-averaged kinetic energy dissipation of the first mode
by horizontal viscosity.
"""
@inline function mode1dissipation(s, v, p, g)
  @views @. v.Uuh = g.k^p.nnu1*s.solc[:, :, 1]
  @views @. v.Vuh = g.l^p.nnu1*s.solc[:, :, 2]
  2*p.nu1/(g.Lx*g.Ly)*(parsevalsum2(v.Uuh, g) + parsevalsum2(v.Vuh, g))
end
@inline mode1dissipation(prob) = mode1dissipation(prob.state, prob.vars, prob.params, prob.grid)

"""
    mode1drag(prob)

Returns the domain-averaged kinetic energy dissipation of the first mode
by horizontal viscosity.
"""
@inline function mode1drag(s, v, p, g)
  @views @. v.Uuh = g.k^p.nmu1*s.solc[:, :, 1]
  @views @. v.Vuh = g.l^p.nmu1*s.solc[:, :, 2]
  if p.nmu1 != 0 # zero out zeroth mode
    @views @. v.Uuh[1, :] = 0
    @views @. v.Vuh[:, 1] = 0
  end
  2*p.mu1/(g.Lx*g.Ly)*(parsevalsum2(v.Uuh, g) + parsevalsum2(v.Vuh, g))
end
@inline mode1drag(prob) = mode1drag(prob.state, prob.vars, prob.params,
                                    prob.grid)

"""
    totalenergy(prob)

Returns the total energy projected onto the zeroth mode.
"""
@inline totalenergy(s, v, p, g) = mode0energy(s, v, g) + mode1energy(s, p, g)
@inline totalenergy(prob) = totalenergy(prob.state, prob.vars, prob.params, prob.grid)

"""
    shearp(prob)

Returns the domain-integrated shear production.
"""
function shearp(s, v, p, g)
  v.Zh .= s.solr
  @. v.Psih = -g.invKKrsq*v.Zh
  @. v.Uh  = -im*g.l  * v.Psih
  @. v.Vh  =  im*g.kr * v.Psih
  @. v.Uxh =  im*g.kr * v.Uh
  @. v.Uyh =  im*g.l  * v.Uh
  @. v.Vxh =  im*g.kr * v.Vh
  @. v.Vyh =  im*g.l  * v.Vh

  ldiv!(v.Ux, g.rfftplan, v.Uxh)
  ldiv!(v.Uy, g.rfftplan, v.Uyh)
  ldiv!(v.Vx, g.rfftplan, v.Vxh)
  ldiv!(v.Vy, g.rfftplan, v.Vyh)

  @views ldiv!(v.u, g.fftplan, s.solc[:, :, 1])
  @views ldiv!(v.v, g.fftplan, s.solc[:, :, 2])

  @. v.UZuzvw = real( 2.0*abs2(v.u)*v.Ux + 2.0*abs2(v.v)*v.Vy
                       + (conj(v.u)*v.v + v.u*conj(v.v))*(v.Uy + v.Vx) )

  g.dx*g.dy*sum(v.UZuzvw)
end
shearp(prob) = shearp(prob.state, prob.vars, prob.params, prob.grid)


"""
    conversion(prob)

Return the domain-integrated conversion from potential to kinetic energy.
"""
function conversion(s, v, p, g)
  @views @. v.wh = -(g.k*s.solc[:, :, 1] + g.l*s.solc[:, :, 2]) / p.m
  @views ldiv!(v.p, g.fftplan, s.solc[:, :, 3])
  ldiv!(v.w, g.fftplan, v.wh)
  # b = i*m*p
  @. v.UZuzvw = real(im*p.m*conj(v.w)*v.p - im*p.m*v.w*conj(v.p))
  g.dx*g.dy*sum(v.UZuzvw)
end
conversion(prob) = conversion(prob.state, prob.vars, prob.params, prob.grid)


"""
    mode0apv(prob)

Returns the projection of available potential vorticity onto the
zeroth mode.
"""
function mode0apv(Z, u, v, p, pr::TwoModeParams, g::TwoDGrid)
  Z .+ irfft( im*pr.m^2/pr.N^2 * (
      g.l .*rfft( @. real(u*conj(p) + conj(u)*p) )
    - g.kr.*rfft( @. real(v*conj(p) + conj(v)*p) )), g.nx)
end

function mode0apv(s, v::Vars, p::TwoModeParams, g::TwoDGrid)
  v.Z = irfft(s.solr, g.nx)
  @views ldiv!(v.u, g.fftplan, s.solc[:, :, 1])
  @views ldiv!(v.v, g.fftplan, s.solc[:, :, 2])
  @views ldiv!(v.p, g.fftplan, s.solc[:, :, 3])
  mode0apv(v.Z, v.u, v.v, v.p, p, g)
end
mode0apv(prob) = mode0apv(prob.state, prob.vars, prob.params, prob.grid)


"""
    mode1apv(prob)

Returns the projection of available potential energy onto the first mode.
"""
mode1apv(Z, zeta, p, pr, g) = @. zeta - pr.m^2/pr.N^2*(pr.f + Z)*p

function mode1apv(Z, s::DualState, v, p, g)
  @views @. v.ph = s.solc[:, :, 3]
  @views @. v.zetah = im*g.k*s.solc[:, :, 2] - im*g.l*s.solc[:, :, 1]

  ldiv!(v.p,  g.fftplan, v.ph)
  ldiv!(v.zeta,  g.fftplan, v.zetah)

  mode1apv(Z, v.zeta, v.p, p, g)
end

function mode1apv(s::DualState, v, p, g)
  v.Zh .= s.solr
  ldiv!(v.Z, g.rfftplan, v.Zh)
  mode1apv(v.Z, v, p, g)
end
mode1apv(prob) = mode1apv(prob.state, prob.vars, prob.params, prob.grid)


"""
    mode1u(prob)

Return the x-velocity associated with mode-1 at z=0.
"""
mode1u(v) = @. real(v.u + conj.(v.u))
mode1u(prob::AbstractProblem) = mode1u(prob.vars)

"""
    mode1v(prob)

Return the y-velocity associated with mode-1 at z=0.
"""
mode1v(v) = @. real(v.v + conj(v.v))
mode1v(prob::AbstractProblem) = mode1v(prob.vars)

"""
    mode1w(prob)

Return the z-velocity associated with mode-1 at z=0.
"""
mode1w(v) = @. real(v.w + conj(v.w))
mode1w(prob::AbstractProblem) = mode1w(prob.vars)

"""
    mode1p(prob)

Return the pressure associated with mode-1 at z=0.
"""
mode1p(v) = @. real(v.p + conj(v.p))
mode1p(prob::AbstractProblem) = mode1p(prob.vars)

"""
    mode1buoyancy(prob)
Return the buoyancy associated with mode-1 at z=0.
"""
mode1buoyancy(v, p) = @. real(im*p.m*v.p - im*p.m*conj(v.p))
mode1buoyancy(prob) = mode1buoyancy(prob.vars, prob.params)

"""
    mode1speed(prob)

Return the speed associated with mode-1 at z=0.
"""
mode1speed(v) = @. sqrt($mode1u(v)^2 + $mode1v(v)^2)
mode1speed(prob::AbstractProblem) = mode1speed(prob.vars)

"""
    mode0speed(prob)

Return the speed associated with mode-0.
"""
mode0speed(prob) = @. sqrt(prob.vars.U^2 + prob.vars.V^2)
=#

end # module

