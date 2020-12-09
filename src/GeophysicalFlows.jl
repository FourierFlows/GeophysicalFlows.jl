module GeophysicalFlows

using
  CUDA,
  FourierFlows,
  Statistics,
  SpecialFunctions

using FFTW: irfft

include("utils.jl")
include("twodnavierstokes.jl")
include("barotropicqg.jl")
include("barotropicqgql.jl")
include("multilayerqg.jl")
include("surfaceqg.jl")
include("shallowater.jl")

end # module
