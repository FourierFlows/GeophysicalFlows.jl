module GeophysicalFlows

using 
  FourierFlows,
  Statistics,
  SpecialFunctions

include("utils.jl")
include("twodturb.jl")
include("barotropicqg.jl")
include("barotropicqgql.jl")
include("verticallycosineboussinesq.jl")
include("verticallyfourierboussinesq.jl")
include("niwqg.jl")

end # module
