#!/usr/bin/env julia

using
  FourierFlows,
  Test,
  Statistics,
  Random,
  FFTW

import # use 'import' rather than 'using' for submodules to keep namespace clean
  # GeophysicalFlows.TwoDTurb,
  # GeophysicalFlows.BarotropicQG,
  # GeophysicalFlows.BarotropicQGQL,
  GeophysicalFlows.MultilayerQG


using GeophysicalFlows: lambdipole
using FourierFlows: parsevalsum, xmoment, ymoment

#using GeophysicalFlows.VerticallyFourierBoussinesq: mode1u

const rtol_lambdipole = 1e-2 # tolerance for lamb dipole tests
const rtol_multilayerqg = 1e-13 # tolerance for multilayerqg forcing tests
const rtol_twodturb = 1e-13 # tolerance for twodturb forcing tests

"Get the CFL number, assuming a uniform grid with `dx=dy`."
cfl(U, V, dt, dx) = maximum([maximum(abs.(U)), maximum(abs.(V))]*dt/dx)
cfl(prob) = cfl(prob.vars.u, prob.vars.v, prob.cl.dt, prob.grid.dx)


# Run tests
testtime = @elapsed begin

# @testset "TwoDTurb" begin
#   include("test_twodturb.jl")
#
#   @test test_twodturb_advection(0.0005, "ForwardEuler")
#   @test test_twodturb_lambdipole(256, 1e-3)
#   @test test_twodturb_stochasticforcingbudgets()
#   @test test_twodturb_energyenstrophy()
# end

# @testset "BarotropicQG" begin
#   include("test_barotropicqg.jl")
#
#   @test test_bqg_rossbywave("ETDRK4", 1e-2, 20)
#   @test test_bqg_rossbywave("FilteredETDRK4", 1e-2, 20)
#   @test test_bqg_rossbywave("RK4", 1e-2, 20)
#   @test test_bqg_rossbywave("FilteredRK4", 1e-2, 20)
#   @test test_bqg_rossbywave("AB3", 1e-3, 200)
#   @test test_bqg_rossbywave("FilteredAB3", 1e-3, 200)
#   @test test_bqg_rossbywave("ForwardEuler", 1e-4, 2000)
#   @test test_bqg_rossbywave("FilteredForwardEuler", 1e-4, 2000)
#   @test test_bqg_advection(0.0005, "ForwardEuler")
#   @test test_bqg_formstress(0.01, "ForwardEuler")
#   @test test_bqg_energyenstrophy()
#   @test test_bqg_meanenergyenstrophy()
#   @test test_bqg_deterministicforcingbudgets()
#   @test test_bqg_stochasticforcingbudgets()
# end

@testset "MultilayerQG" begin
  include("test_multilayerqg.jl")

  @test test_pvtofromstreamfunction()
  @test test_mqg_nonlinearadvection(0.001, "ForwardEuler")
  @test test_mqg_linearadvection(0.001, "ForwardEuler")
  @test test_mqg_energies()
end


#=
@testset "BarotropicQGQL" begin
  include("test_barotropicqgql.jl")
end

@testset "Vertically Cosine Boussinesq" begin
  include("test_verticallycosineboussinesq.jl")

  @test test_cosine_nonlinearterms(0.0005, "ForwardEuler")
  @test test_cosine_lambdipole(256, 1e-3)
  @test test_cosine_groupvelocity(16)
end

@testset "Vertically Fourier Boussinesq" begin
  include("test_verticallyfourierboussinesq.jl")

  @test test_fourier_lambdipole(256, 1e-3)
  @test test_fourier_groupvelocity(16)
end

@testset "NIWQG" begin
  include("test_niwqg.jl")

  @test test_set_q()
  @test test_set_phi()

  @test test_niwqg_lambdipole(nx=256, dt=1e-3, stepper="FilteredRK4")
  @test test_niwqg_lambdipole(nx=256, dt=1e-3, stepper="FilteredETDRK4")

  @test test_niwqg_groupvelocity(nkw=16, stepper="FilteredRK4")
  @test test_niwqg_groupvelocity(nkw=16, stepper="FilteredETDRK4")

  @test test_niwqg_forcing_q(dt=0.01, stepper="RK4", nsteps=100)
  @test test_niwqg_forcing_q(dt=0.01, stepper="FilteredRK4", nsteps=100)
  @test test_niwqg_forcing_phi(dt=0.01, stepper="RK4", nsteps=100, muw=0, nu=0.01, nnu=0)
  @test test_niwqg_forcing_phi(dt=0.01, stepper="FilteredRK4", nsteps=100, muw=0, nu=0.01, nnu=0)
  @test test_niwqg_forcing_phi(dt=0.01, stepper="FilteredRK4", nsteps=100, nu=0, muw=0.02, nmuw=1)

  @test test_niwqg_nonlinear1(dt=0.01, stepper="FilteredRK4", nsteps=100)
  @test test_niwqg_nonlinear2(dt=0.01, stepper="FilteredRK4", nsteps=100)
  @test test_niwqg_wavepv(dt=0.01, stepper="FilteredRK4", nsteps=100)

  @test test_niwqg_calczetah()
  @test test_niwqg_energetics()

  @test   test_niwqg_hyperdissipation_q(nsteps=50, nkap=1, stepper="ETDRK4")
  @test   test_niwqg_hyperdissipation_q(nsteps=50, nkap=2, stepper="ETDRK4")
  @test test_niwqg_hyperdissipation_phi(nsteps=50, nnu=1,  stepper="ETDRK4")
  @test test_niwqg_hyperdissipation_phi(nsteps=50, nnu=2,  stepper="ETDRK4")

  @test test_niwqg_exactnonlinearsolution(nsteps=50)
end
=#

end # time

println("Total test time: ", testtime)
