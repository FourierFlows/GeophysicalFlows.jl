function test_sqg_problemtype(dev, T)
  prob = SurfaceQG.Problem(dev; T=T)

  A = ArrayType(dev)

  (typeof(prob.sol)<:A{Complex{T},2} && typeof(prob.grid.Lx)==T && eltype(prob.grid.x)==T && typeof(prob.vars.u)<:A{T,2})
end
