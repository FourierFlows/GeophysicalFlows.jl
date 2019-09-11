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
  GeophysicalFlows.BarotropicQGQL,
  GeophysicalFlows.MultilayerQG

using FourierFlows: parsevalsum, xmoment, ymoment
using GeophysicalFlows: lambdipole, peakedisotropicspectrum

const rtol_lambdipole = 1e-2 # tolerance for lamb dipole tests
const rtol_multilayerqg = 1e-13 # tolerance for multilayerqg forcing tests
const rtol_twodturb = 1e-13 # tolerance for twodturb forcing tests

"Get the CFL number, assuming a uniform grid with `dx=dy`."
cfl(u, v, dt, dx) = maximum([maximum(abs.(u)), maximum(abs.(v))]*dt/dx)
cfl(prob) = cfl(prob.vars.u, prob.vars.v, prob.clock.dt, prob.grid.dx)


# Run tests
testtime = @elapsed begin

@testset "Utils" begin
  include("test_utils.jl")

  @test testpeakedisotropicspectrum()
  @test_throws ErrorException("the domain is not square") testpeakedisotropicspectrum_rectangledomain()
end

@testset "TwoDTurb" begin
  include("test_twodturb.jl")

  @test test_twodturb_advection(0.0005, "ForwardEuler")
  @test test_twodturb_lambdipole(256, 1e-3)
  @test test_twodturb_stochasticforcingbudgets()
  @test test_twodturb_deterministicforcingbudgets()
  @test test_twodturb_energyenstrophy()
  @test test_twodturb_problemtype(Float32)
  @test TwoDTurb.nothingfunction() == nothing
end

@testset "BarotropicQG" begin
  include("test_barotropicqg.jl")

  @test test_bqg_rossbywave("ETDRK4", 1e-2, 20)
  @test test_bqg_rossbywave("FilteredETDRK4", 1e-2, 20)
  @test test_bqg_rossbywave("RK4", 1e-2, 20)
  @test test_bqg_rossbywave("FilteredRK4", 1e-2, 20)
  @test test_bqg_rossbywave("AB3", 1e-3, 200)
  @test test_bqg_rossbywave("FilteredAB3", 1e-3, 200)
  @test test_bqg_rossbywave("ForwardEuler", 1e-4, 2000)
  @test test_bqg_rossbywave("FilteredForwardEuler", 1e-4, 2000)
  @test test_bqg_stochasticforcingbudgets()
  @test test_bqg_deterministicforcingbudgets()
  @test test_bqg_advection(0.0005, "ForwardEuler")
  @test test_bqg_formstress(0.01, "ForwardEuler")
  @test test_bqg_energyenstrophy()
  @test test_bqg_meanenergyenstrophy()
  @test test_bqg_problemtype(Float32)
  @test BarotropicQG.nothingfunction() == nothing
end

@testset "BarotropicQGQL" begin
  include("test_barotropicqgql.jl")

  @test test_bqgql_rossbywave("ETDRK4", 1e-2, 20)
  @test test_bqgql_rossbywave("FilteredETDRK4", 1e-2, 20)
  @test test_bqgql_rossbywave("RK4", 1e-2, 20)
  @test test_bqgql_rossbywave("FilteredRK4", 1e-2, 20)
  @test test_bqgql_rossbywave("AB3", 1e-3, 200)
  @test test_bqgql_rossbywave("FilteredAB3", 1e-3, 200)
  @test test_bqgql_rossbywave("ForwardEuler", 1e-4, 2000)
  @test test_bqgql_rossbywave("FilteredForwardEuler", 1e-4, 2000)
  @test test_bqgql_deterministicforcingbudgets()
  @test test_bqgql_stochasticforcingbudgets()
  @test test_bqgql_advection(0.0005, "ForwardEuler")
  @test test_bqgql_energyenstrophy()
  @test test_bqgql_problemtype(Float32)
  @test BarotropicQGQL.nothingfunction() == nothing
end

@testset "MultilayerQG" begin
  include("test_multilayerqg.jl")

  @test test_pvtofromstreamfunction_2layer()
  @test test_pvtofromstreamfunction_3layer()
  @test test_mqg_nonlinearadvection(0.001, "ForwardEuler")
  @test test_mqg_linearadvection(0.001, "ForwardEuler")
  @test test_mqg_energies()
  @test test_mqg_fluxes()
  @test test_mqg_setqsetpsi()
  @test test_mqg_paramsconstructor()
  @test test_mqg_problemtype(Float32)
  @test MultilayerQG.nothingfunction() == nothing
end

end # time

println("Total test time: ", testtime)
