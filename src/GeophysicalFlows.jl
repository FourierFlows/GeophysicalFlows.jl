module GeophysicalFlows

using
  CUDA,
  FourierFlows,
  Statistics,
  SpecialFunctions

using FFTW: irfft

include("utils.jl")
include("twodnavierstokes.jl")
include("singlelayerqg.jl")
include("barotropicqgql.jl")
include("multilayerqg.jl")
include("surfaceqg.jl")
include("shallowwater.jl")

end # module
