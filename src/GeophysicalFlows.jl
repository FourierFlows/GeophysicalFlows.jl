module GeophysicalFlows

using
  FourierFlows,
  Statistics,
  SpecialFunctions

using FFTW: irfft

include("utils.jl")
include("twodturb.jl")
include("barotropicqg.jl")
include("multilayerqg.jl")

end # module
