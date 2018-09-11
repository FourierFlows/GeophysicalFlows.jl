#!/usr/bin/env julia

#using CuArrays
using Requires, Test, FourierFlows

# Run tests
testtime = @elapsed begin

@testset "Physics: TwoDTurb" begin
  include("test_twodturb.jl")
end

@testset "Physics: BarotropicQG" begin
  include("test_barotropicqg.jl")
end

end

println("Total test time: ", testtime)
