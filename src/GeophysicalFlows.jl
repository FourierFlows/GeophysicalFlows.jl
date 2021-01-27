"""
Main module for `GeophysicalFlows.jl` -- a collection of solvers for geophysical fluid dynamics
problems in periodic domains on CPUs and GPUs. All modules use Fourier-based pseudospectral 
methods and leverage the functionality of `FourierFlows.jl` ecosystem.
"""
module GeophysicalFlows

using
  CUDA,
  FourierFlows,
  Statistics,
  SpecialFunctions,
  DocStringExtensions

using FFTW: irfft

include("utils.jl")
include("twodnavierstokes.jl")
include("singlelayerqg.jl")
include("barotropicqgql.jl")
include("multilayerqg.jl")
include("surfaceqg.jl")

end # module
