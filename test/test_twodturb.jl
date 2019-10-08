function test_twodturb_lambdipole(n, dt, dev::Device=CPU(); L=2π, Ue=1, Re=L/20, ν=0.0, nν=1, ti=L/Ue*0.01, nm=3)
  nt = round(Int, ti/dt)
  prob = TwoDTurb.Problem(nx=n, Lx=L, ν=ν, nν=nν, dt=dt, stepper="FilteredRK4", dev=dev)
  zeta₀ = lambdipole(Ue, Re, prob.grid)
  TwoDTurb.set_zeta!(prob, zeta₀)

  xzeta = zeros(nm) # centroid of abs(zeta)
  Ue_m = zeros(nm)  # measured dipole speed
  x, y = gridpoints(prob.grid)
  zeta = prob.vars.zeta

  for i = 1:nm # step forward
    stepforward!(prob, nt)
    TwoDTurb.updatevars!(prob)
    xzeta[i] = mean(@. abs(zeta)*x) / mean(abs.(zeta))
    if i > 1
      Ue_m[i] = (xzeta[i]-xzeta[i-1]) / ((nt-1)*dt)
    end
  end
  isapprox(Ue, mean(Ue_m[2:end]), rtol=rtol_lambdipole)
end

function test_twodturb_stochasticforcingbudgets(dev::Device=CPU(); n=256, L=2π, dt=0.005, ν=1e-7, nν=2, μ=1e-1, nμ=0, tf=0.1/μ)
  nt = round(Int, tf/dt)

  # Forcing parameters
  kf, dkf = 12.0, 2.0
  ε = 0.1

  gr  = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

  Kr = ArrayType(dev)([ gr.kr[i] for i=1:gr.nkr, j=1:gr.nl])

  force2k = ArrayType(dev)(zero(gr.Krsq))
  @. force2k = exp(-(sqrt(gr.Krsq)-kf)^2/(2*dkf^2))
  @. force2k[gr.Krsq .< 2.0^2 ] = 0
  @. force2k[gr.Krsq .> 20.0^2 ] = 0
  @. force2k[Kr .< 2π/L] = 0
  ε0 = parsevalsum(force2k.*gr.invKrsq/2.0, gr)/(gr.Lx*gr.Ly)
  force2k .= ε/ε0 * force2k

  Random.seed!(1234)

  function calcF!(Fh, sol, t, cl, v, p, g)
    eta = ArrayType(dev)(exp.(2π*im*rand(Float64, size(sol)))/sqrt(cl.dt))
    eta[1, 1] = 0.0
    @. Fh = eta * sqrt(force2k)
    nothing
  end

  prob = TwoDTurb.Problem(nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=true, dev=dev)

  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid;

  TwoDTurb.set_zeta!(prob, 0*x)
  E = Diagnostic(TwoDTurb.energy,      prob, nsteps=nt)
  D = Diagnostic(TwoDTurb.dissipation, prob, nsteps=nt)
  R = Diagnostic(TwoDTurb.drag,        prob, nsteps=nt)
  W = Diagnostic(TwoDTurb.work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  stepforward!(prob, diags, nt)
  TwoDTurb.updatevars!(prob)

  E, D, W, R = diags
  t = round(μ*cl.t, digits=2)

  i₀ = 1
  dEdt = (E[(i₀+1):E.i] - E[i₀:E.i-1])/cl.dt
  ii = (i₀):E.i-1
  ii2 = (i₀+1):E.i

  # dEdt = W - D - R?
  # If the Ito interpretation was used for the work
  # then we need to add the drift term
  # total = W[ii2]+ε - D[ii] - R[ii]      # Ito
  total = W[ii2] - D[ii] - R[ii]        # Stratonovich

  residual = dEdt - total
  isapprox(mean(abs.(residual)), 0, atol=1e-4)
end


function test_twodturb_deterministicforcingbudgets(dev::Device=CPU(); n=256, dt=0.01, L=2π, ν=1e-7, nν=2, μ=1e-1, nμ=0)
  n, L  = 256, 2π
  ν, nν = 1e-7, 2
  μ, nμ = 1e-1, 0
  dt, tf = 0.005, 0.1/μ
  nt = round(Int, tf/dt)
  
  gr  = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

  # Forcing = 0.01cos(4x)cos(5y)cos(2t)
  f = @. 0.01cos(4x)*cos(5y)
  fh = rfft(f)
  function calcF!(Fh, sol, t, cl, v, p, g::AbstractGrid{T, A}) where {T, A}
    Fh = fh*cos(2t)
    nothing
  end

  prob = TwoDTurb.Problem(nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt,
   stepper="RK4", calcF=calcF!, stochastic=false, dev=dev)

  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  TwoDTurb.set_zeta!(prob, 0*x)

  E = Diagnostic(TwoDTurb.energy,      prob, nsteps=nt)
  D = Diagnostic(TwoDTurb.dissipation, prob, nsteps=nt)
  R = Diagnostic(TwoDTurb.drag,        prob, nsteps=nt)
  W = Diagnostic(TwoDTurb.work,        prob, nsteps=nt)
  diags = [E, D, W, R]

  # Step forward
  stepforward!(prob, diags, nt)
  TwoDTurb.updatevars!(prob)

  E, D, W, R = diags
  t = round(μ*cl.t, digits=2)

  i₀ = 1
  dEdt = (E[(i₀+1):E.i] - E[i₀:E.i-1])/cl.dt
  ii = (i₀):E.i-1
  ii2 = (i₀+1):E.i

  # dEdt = W - D - R?
  total = W[ii2] - D[ii] - R[ii]
  residual = dEdt - total

  isapprox(mean(abs.(residual)), 0, atol=1e-8)
end

"""
    testnonlinearterms(dt, stepper; kwargs...)

Tests the advection term in the twodturb module by timestepping a
test problem with timestep dt and timestepper identified by the string stepper.
The test problem is derived by picking a solution ζf (with associated
streamfunction ψf) for which the advection term J(ψf, ζf) is non-zero. Next, a
forcing Ff is derived according to Ff = ∂ζf/∂t + J(ψf, ζf) - νΔζf. One solution
to the vorticity equation forced by this Ff is then ζf. (This solution may not
be realized, at least at long times, if it is unstable.)
"""
function test_twodturb_advection(dt, stepper, dev::Device=CPU(); n=128, L=2π, ν=1e-2, nν=1, μ=0.0, nμ=0)
  n, L  = 128, 2π
  ν, nν = 1e-2, 1
  μ, nμ = 0.0, 0
  tf = 1.0
  nt = round(Int, tf/dt)

  gr   = TwoDGrid(dev, n, L)
  x, y = gridpoints(gr)

   psif = @.   sin(2x)*cos(2y) +  2sin(x)*cos(3y)
  zetaf = @. -8sin(2x)*cos(2y) - 20sin(x)*cos(3y)

  Ff = @. -(
    ν*( 64sin(2x)*cos(2y) + 200sin(x)*cos(3y) )
    + 8*( cos(x)*cos(3y)*sin(2x)*sin(2y) - 3cos(2x)*cos(2y)*sin(x)*sin(3y) )
  )

  Ffh = rfft(Ff)

  # Forcing
  function calcF!(Fh, sol, t, cl, v, p, g)
    Fh .= Ffh
    nothing
  end

  prob = TwoDTurb.Problem(nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt, stepper=stepper, calcF=calcF!, stochastic=false, dev=dev)
  sol, cl, p, v, g = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
  TwoDTurb.set_zeta!(prob, zetaf)

  stepforward!(prob, nt)
  TwoDTurb.updatevars!(prob)

  isapprox(prob.vars.zeta, zetaf, rtol=rtol_twodturb)
end

function test_twodturb_energyenstrophy(dev::Device=CPU())
  nx, Lx  = 128, 2π
  ny, Ly  = 128, 3π
  gr = TwoDGrid(dev, nx, Lx, ny, Ly)
  x, y = gridpoints(gr)

  k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers
    psi0 = @. sin(2k0*x)*cos(2l0*y) + 2sin(k0*x)*cos(3l0*y)
   zeta0 = @. -((2k0)^2+(2l0)^2)*sin(2k0*x)*cos(2l0*y) - (k0^2+(3l0)^2)*2sin(k0*x)*cos(3l0*y)

  energy_calc = 29/9
  enstrophy_calc = 2701/162

  prob = TwoDTurb.Problem(nx=nx, Lx=Lx, ny=ny, Ly=Ly, stepper="ForwardEuler", dev=dev)
  
  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid;

  TwoDTurb.set_zeta!(prob, zeta0)
  TwoDTurb.updatevars!(prob)

  energyzeta0 = TwoDTurb.energy(prob)
  enstrophyzeta0 = TwoDTurb.enstrophy(prob)
  
  params = TwoDTurb.Params(p.ν, p.nν)

  (isapprox(energyzeta0, energy_calc, rtol=rtol_twodturb) &&
   isapprox(enstrophyzeta0, enstrophy_calc, rtol=rtol_twodturb) &&
   TwoDTurb.addforcing!(prob.timestepper.N, sol, cl.t, cl, v, p, g)==nothing && p == params)
end

function test_twodturb_problemtype(T=Float32, dev::Device=CPU())
  prob = TwoDTurb.Problem(T=T, dev=dev)

  (typeof(prob.sol)==Array{Complex{T},2} && typeof(prob.grid.Lx)==T && typeof(prob.grid.x)==Array{T,2} && typeof(prob.vars.u)==Array{T,2})
end
