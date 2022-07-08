"""
Main module for `GeophysicalFlows.jl` -- a collection of solvers for geophysical fluid dynamics
problems in periodic domains on CPUs and GPUs. All modules use Fourier-based pseudospectral 
methods and leverage the functionality of `FourierFlows.jl` ecosystem.
"""
module GeophysicalFlows

using
  CUDA,
  Statistics,
  SpecialFunctions,
  Reexport,
  DocStringExtensions

@reexport using FourierFlows

include("utils.jl")
include("twodnavierstokes.jl")
include("twodnavierstokeswithtracer.jl")
include("singlelayerqg.jl")
include("multilayerqg.jl")
include("surfaceqg.jl")
include("barotropicqgql.jl")

@reexport using GeophysicalFlows.TwoDNavierStokes
@reexport using GeophysicalFlows.TwoDNavierStokesTracer
@reexport using GeophysicalFlows.SingleLayerQG
@reexport using GeophysicalFlows.MultiLayerQG
@reexport using GeophysicalFlows.SurfaceQG
@reexport using GeophysicalFlows.BarotropicQGQL

end # module
