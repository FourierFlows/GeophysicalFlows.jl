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

"""
    test_sqg_advection(dt, stepper; kwargs...)

Tests the advection term in the SurfaceQG module by timestepping a
test problem with timestep dt and timestepper identified by the string stepper.
The test problem is derived by picking a solution bf (with associated
streamfunction ψf) for which the advection term J(ψf, bf) is non-zero. Next, a
forcing Ff is derived according to Ff = ∂bf/∂t + J(ψf, bf) - ν∇²bf. One solution
to the buoyancy equation forced by this Ff is then bf. (This solution may not
be realized, at least at long times, if it is unstable.)
"""
function test_sqg_advection(dt, stepper, dev::Device=CPU(); n=128, L=2π, ν=1e-2, nν=1)
  n, L  = 128, 2π
  ν, nν = 1e-2, 1
  tf = 0.5  # SQG piles up energy at small energy so running for longer brings up instability
  nt = round(Int, tf/dt)

  grid = TwoDGrid(dev, n, L)
  x, y = gridpoints(grid)

  ψf = @.             sin(2x)*cos(2y) +            2sin(x)*cos(3y)
  bf = @. - sqrt(8) * sin(2x)*cos(2y) - sqrt(10) * 2sin(x)*cos(3y)

  Ff = @. (
    - ν*(16*sqrt(2) * sin(2x)*cos(2y) + 10*sqrt(10) * 2sin(x)*cos(3y) )
    - 4*sqrt(2)*(sqrt(5)-2) * (cos(x)*cos(3y)*sin(2x)*sin(2y) 
      - 3cos(2x)*cos(2y)*sin(x)*sin(3y))
  )

  Ffh = rfft(Ff)

  # Forcing
  function calcF!(Fh, sol, t, clock, vars, params, grid)
    Fh .= Ffh
    return nothing
  end

  prob = SurfaceQG.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, dt=dt, stepper=stepper, calcF=calcF!, stochastic=false)
  
  SurfaceQG.set_b!(prob, bf)

  stepforward!(prob, nt)
  
  SurfaceQG.updatevars!(prob)

  isapprox(prob.vars.b, bf, rtol=rtol_surfaceqg)
end
