using
  CUDA,
  FourierFlows,
  Statistics,
  Random,
  FFTW,
  Test

import # use 'import' rather than 'using' for submodules to keep namespace clean
  GeophysicalFlows.TwoDNavierStokes,
  GeophysicalFlows.SingleLayerQG,
  GeophysicalFlows.BarotropicQGQL,
  GeophysicalFlows.MultilayerQG,
  GeophysicalFlows.SurfaceQG

using FourierFlows: parsevalsum
using GeophysicalFlows: lambdipole, peakedisotropicspectrum

# the devices on which tests will run
devices = (CPU(),)
@has_cuda devices = (CPU(), GPU())

const rtol_lambdipole = 1e-2 # tolerance for lamb dipole tests
const rtol_twodnavierstokes = 1e-13 # tolerance for twodnavierstokes forcing tests
const rtol_singlelayerqg = 1e-13 # tolerance for singlelayerqg forcing tests
const rtol_multilayerqg = 1e-13 # tolerance for multilayerqg forcing tests
const rtol_surfaceqg = 1e-13 # tolerance for surfaceqg forcing tests


# Run tests
testtime = @elapsed begin
for dev in devices
  
  println("testing on "*string(typeof(dev)))
  
  @testset "Utils" begin
    include("test_utils.jl")

    @test testpeakedisotropicspectrum(dev)
    @test_throws ErrorException("the domain is not square") testpeakedisotropicspectrum_rectangledomain()
  end
  
  @testset "TwoDNavierStokes" begin
    include("test_twodnavierstokes.jl")

    @test test_twodnavierstokes_advection(0.0005, "ForwardEuler", dev)
    @test test_twodnavierstokes_lambdipole(256, 1e-3, dev)
    @test test_twodnavierstokes_deterministicforcing_energybudget(dev)
    @test test_twodnavierstokes_stochasticforcing_energybudget(dev)
    @test test_twodnavierstokes_deterministicforcing_enstrophybudget(dev)
    @test test_twodnavierstokes_stochasticforcing_enstrophybudget(dev)
    @test test_twodnavierstokes_energyenstrophy(dev)
    @test test_twodnavierstokes_problemtype(dev, Float32)
    @test TwoDNavierStokes.nothingfunction() == nothing
  end
  
  @testset "SingleLayerQG" begin
    include("test_singlelayerqg.jl")

    @test test_1layerqg_rossbywave("ETDRK4", 1e-2, 20, dev)
    @test test_1layerqg_rossbywave("FilteredETDRK4", 1e-2, 20, dev)
    @test test_1layerqg_rossbywave("RK4", 1e-2, 20, dev)
    @test test_1layerqg_rossbywave("FilteredRK4", 1e-2, 20, dev)
    @test test_1layerqg_rossbywave("AB3", 1e-3, 200, dev)
    @test test_1layerqg_rossbywave("FilteredAB3", 1e-3, 200, dev)
    @test test_1layerqg_rossbywave("ForwardEuler", 1e-4, 2000, dev)
    @test test_1layerqg_rossbywave("FilteredForwardEuler", 1e-4, 2000, dev)
    @test test_1layerqg_deterministicforcing_energybudget(dev)
    @test test_1layerqg_stochasticforcing_energybudget(dev)
    @test test_1layerqg_deterministicforcing_enstrophybudget(dev)
    @test test_1layerqg_stochasticforcing_enstrophybudget(dev)
    @test test_1layerqg_advection(0.0005, "ForwardEuler", dev)
    @test test_1layerqg_energyenstrophy(dev)
    @test test_1layerqg_problemtype(dev, Float32)
    @test SingleLayerQG.nothingfunction() == nothing
  end
   
  @testset "BarotropicQGQL" begin
    include("test_barotropicqgql.jl")

    @test test_bqgql_rossbywave("ETDRK4", 1e-2, 20, dev)
    @test test_bqgql_rossbywave("FilteredETDRK4", 1e-2, 20, dev)
    @test test_bqgql_rossbywave("RK4", 1e-2, 20, dev)
    @test test_bqgql_rossbywave("FilteredRK4", 1e-2, 20, dev)
    @test test_bqgql_rossbywave("AB3", 1e-3, 200, dev)
    @test test_bqgql_rossbywave("FilteredAB3", 1e-3, 200, dev)
    @test test_bqgql_rossbywave("ForwardEuler", 1e-4, 2000, dev)
    @test test_bqgql_rossbywave("FilteredForwardEuler", 1e-4, 2000, dev)
    @test test_bqgql_deterministicforcingbudgets(dev)
    @test test_bqgql_stochasticforcingbudgets(dev)
    @test test_bqgql_advection(0.0005, "ForwardEuler", dev)
    @test test_bqgql_energyenstrophy(dev)
    @test test_bqgql_problemtype(dev, Float32)
    @test BarotropicQGQL.nothingfunction() == nothing
  end
  
  @testset "SurfaceQG" begin
    include("test_surfaceqg.jl")
      
    @test test_sqg_kineticenergy_buoyancyvariance(dev)
    @test test_sqg_advection(0.0005, "ForwardEuler", dev)
    @test test_sqg_deterministicforcing_buoyancy_variance_budget(dev)
    @test test_sqg_stochasticforcing_buoyancy_variance_budget(dev)
    @test test_sqg_stochasticforcedproblemconstructor(dev)
    @test test_sqg_problemtype(dev, Float32)
    @test test_sqg_paramsconstructor(dev)
    @test test_sqg_noforcing(dev)
    @test SurfaceQG.nothingfunction() == nothing
  end
  
  @testset "MultilayerQG" begin
    include("test_multilayerqg.jl")
    
    @test test_pvtofromstreamfunction_2layer(dev)
    @test test_pvtofromstreamfunction_3layer(dev)
    @test test_mqg_rossbywave("RK4", 1e-2, 20, dev)
    @test test_mqg_nonlinearadvection(0.005, "ForwardEuler", dev)
    @test test_mqg_linearadvection(0.005, "ForwardEuler", dev)
    @test test_mqg_energies(dev)
    @test test_mqg_energysinglelayer(dev)
    @test test_mqg_fluxes(dev)
    @test test_mqg_fluxessinglelayer(dev)
    @test test_mqg_setqsetψ(dev)
    @test test_mqg_paramsconstructor(dev)
    @test test_mqg_stochasticforcedproblemconstructor(dev)
    @test test_mqg_problemtype(dev, Float32)
    @test MultilayerQG.nothingfunction() == nothing
  end
end
end # time

println("Total test time: ", testtime)
