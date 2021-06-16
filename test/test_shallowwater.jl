function test_swe_problemtype(dev, T)
  prob = ShallowWater.Problem(dev; T=T)

  A = ArrayType(dev)

  (typeof(prob.sol)<:A{Complex{T},3} && typeof(prob.grid.Lx)==T && eltype(prob.grid.x)==T && typeof(prob.vars.u)<:A{T,2})
end

