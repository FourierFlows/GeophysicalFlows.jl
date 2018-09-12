#!/usr/bin/env julia

#using CuArrays
using Requires, Test, FourierFlows

# Run tests
testtime = @elapsed begin

@testset "TwoDTurb" begin
  include("test_twodturb.jl")
end

@testset "BarotropicQG" begin
  include("test_barotropicqg.jl")
end

@testset "Vertically Cosine Boussinesq" begin
  include("test_verticallycosineboussinesq.jl")
end

@testset "Vertically Fourier Boussinesq" begin
  include("test_verticallyfourierboussinesq.jl")
end

end

println("Total test time: ", testtime)
