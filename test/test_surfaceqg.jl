function test_sqg_problemtype(dev, T)
  prob = SurfaceQG.Problem(dev; T=T)

  A = ArrayType(dev)

  (typeof(prob.sol)<:A{Complex{T},2} && typeof(prob.grid.Lx)==T && eltype(prob.grid.x)==T && typeof(prob.vars.u)<:A{T,2})
end

function test_sqg_stochasticforcedproblemconstructor(dev::Device=CPU())
  
  function calcF!(Fqh, sol, t, clock, vars, params, grid)
    Fqh .= Ffh
    return nothing
  end
       
  prob = SurfaceQG.Problem(dev; calcF=calcF!, stochastic=true)
  
  return typeof(prob.vars.prevsol) == typeof(prob.sol)
end

