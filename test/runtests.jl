#!/usr/bin/env julia

using 
  FourierFlows,
  Test,
  Statistics,
  Random,
  FFTW

import # use 'import' rather than 'using' for submodules to keep namespace clean
  GeophysicalFlows.TwoDTurb,
  GeophysicalFlows.BarotropicQG,
  GeophysicalFlows.VerticallyCosineBoussinesq,
  GeophysicalFlows.VerticallyFourierBoussinesq,
  GeophysicalFlows.NIWQG

using FourierFlows: lambdipole, parsevalsum, xmoment, ymoment
using GeophysicalFlows.VerticallyFourierBoussinesq: mode1u

const rtol_lambdipole = 1e-2 # tolerance for lamb dipole tests
const rtol_niwqg = 1e-13 # tolerance for niwqg forcing tests
const rtol_twodturb = 1e-13 # tolerance for niwqg forcing tests

"Get the CFL number, assuming a uniform grid with `dx=dy`."
cfl(U, V, dt, dx) = maximum([maximum(abs.(U)), maximum(abs.(V))]*dt/dx)
cfl(prob) = cfl(prob.vars.U, prob.vars.V, prob.ts.dt, prob.grid.dx)

"Returns the energy in vertically Fourier mode 1 in the Boussinesq equations."
e1_fourier(u, v, p, m, N) = @. abs2(u) + abs2(v) + m^2*abs2(p)/N^2
e1_fourier(prob) = e1_fourier(prob.vars.u, prob.vars.v, prob.vars.p, prob.params.m, prob.params.N)

"Returns the `x,y` centroid of a cosine mode 1 internal wave in the Boussinesq equations."
wavecentroid_fourier(prob) = (xmoment(e1_fourier(prob), prob.grid), ymoment(e1_fourier(prob), prob.grid))
                             
"Returns the energy in vertically cosine mode 1 in the Boussinesq equations."
e1_cosine(u, v, p, m, N) = @. ( u^2 + v^2 + m^2*p^2/N^2 )/2
e1_cosine(prob) = e1_cosine(prob.vars.u, prob.vars.v, prob.vars.p, prob.params.m, prob.params.N)

"Returns the x,y centroid of a cosine mode 1 internal wave in the Boussinesq equations."
wavecentroid_cosine(prob) = (xmoment(e1_cosine(prob), prob.grid), ymoment(e1_cosine(prob), prob.grid))

"Returns the wave kinetic energy in NIWQG."
ke_niwqg(phi) = @. abs2(phi)
ke_niwqg(prob::AbstractProblem) = ke_niwqg(prob.vars.phi)

"Returns the `x,y` centroid of the wave field kinetic energy in NIWQG."
wavecentroid_niwqg(prob) = (xmoment(ke_niwqg(prob), prob.grid), ymoment(ke_niwqg(prob), prob.grid))

# Run tests
testtime = @elapsed begin

@testset "TwoDTurb" begin
  include("test_twodturb.jl")
end

@testset "BarotropicQG" begin
  include("test_barotropicqg.jl")
end

@testset "Vertically Cosine Boussinesq" begin
  include("test_verticallycosineboussinesq.jl")
end

@testset "Vertically Fourier Boussinesq" begin
  include("test_verticallyfourierboussinesq.jl")
end

@testset "NIWQG" begin
  include("test_niwqg.jl")

  @test test_set_q()
  @test test_set_phi()
  @test test_niwqg_lambdipole(nx=256, dt=1e-3)
  @test test_niwqg_groupvelocity(nkw=16)
  @test test_niwqg_forcing_q(dt=0.01, stepper="RK4", nsteps=100)
  @test test_niwqg_forcing_phi(dt=0.01, stepper="RK4", nsteps=100, muw=0, nu=0.01, nnu=0)
  @test test_niwqg_forcing_phi(dt=0.01, stepper="RK4", nsteps=100, nu=0, muw=0.02, nmuw=1)
  @test test_niwqg_nonlinear1(dt=0.01, stepper="RK4", nsteps=100)
  @test test_niwqg_nonlinear2(dt=0.01, stepper="RK4", nsteps=10)
  @test test_niwqg_wavepv(dt=0.01, stepper="RK4", nsteps=100)
end

end

println("Total test time: ", testtime)
