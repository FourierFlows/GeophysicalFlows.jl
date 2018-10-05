#=
function test_niwqg_lambdipole(n, dt; L=2π, Ue=1, Re=L/20, kap=0, nkap=1, ti=L/Ue*0.01, nm=3, message=false)
  nt = round(Int, ti/dt)
  prob = NIWQG.Problem(nx=n, Lx=L, kap=kap, nkap=nkap, dt=dt, stepper="FilteredRK4")
  x, y, q = prob.grid.X, prob.grid.Y, prob.vars.q # nicknames

  q0 = lambdipole(Ue, Re, prob.grid)
  NIWQG.set_q!(prob, q0)

  xq = zeros(nm)   # centroid of abs(Z)
  Ue_m = zeros(nm) # measured dipole speed

  # Step forward
  for i = 1:nm
    stepforward!(prob, nt)
    NIWQG.updatevars!(prob)
    xq[i] = mean(abs.(q).*x) / mean(abs.(q))

    if i > 1
      Ue_m[i] = (xq[i]-xq[i-1]) / ((nt-1)*dt)
    else
      Ue_m[i] = 0
    end
  end

  isapprox(Ue, mean(Ue_m[2:end]), atol=0.02)
end
=#

function test_niwqg_lambdipole(n, dt; L=2π, Ue=1, Re=L/20, ti=L/Ue*0.01, nm=3)
  nt = round(Int, ti/dt)
  prob = NIWQG.Problem(nx=n, Lx=L, dt=dt, stepper="FilteredRK4")
  q0 = lambdipole(Ue, Re, prob.grid)
  NIWQG.set_q!(prob, q0)

  xq = zeros(nm)   # centroid of abs(q)
  Ue_m = zeros(nm) # measured dipole speed
  x, y, q = prob.grid.X, prob.grid.Y, prob.vars.q # nicknames

  for i = 1:nm # step forward
    stepforward!(prob, nt)
    NIWQG.updatevars!(prob)
    xq[i] = mean(abs.(q).*x) / mean(abs.(q))
    if i > 1
      Ue_m[i] = (xq[i]-xq[i-1]) / ((nt-1)*dt)
    end
  end

  println("Ue: $Ue, Ue_m: $(mean(Ue_m[2:end]))")

  isapprox(Ue, mean(Ue_m[2:end]), rtol=1e-2)
end



function test_niwqg_groupvelocity(nkw; n=128, L=2π, f=1, eta=1/64, uw=1e-2, rtol=1e-3, del=L/10)
  kw = nkw*2π/L
   σ = eta*kw^2/2
  tσ = 2π/σ
  dt = tσ/100
  nt = round(Int, 3tσ/(2dt)) # 3/2 a wave period
  cga = eta*kw # analytical group velocity

  prob = NIWQG.Problem(f=f, eta=eta, nx=n, Lx=L, dt=dt, stepper="FilteredRK4")
  envelope(x, y) = exp(-x^2/(2*del^2))
  NIWQG.set_planewave!(prob, uw, nkw; envelope=envelope)

  t₋₁ = prob.t
  xw₋₁, yw₋₁ = wavecentroid_niwqg(prob)

  stepforward!(prob, nt)
  NIWQG.updatevars!(prob)
  xw, yw = wavecentroid_niwqg(prob)
  cgn = (xw-xw₋₁) / (prob.t-t₋₁)

  isapprox(cga, cgn, rtol=rtol)
end

function test_niwqg_nonlinear1(nk=2; nx=32, Lx=2π, dt=0.1, nsteps=100, eta=0.2)
  k = 2π/Lx * nk
  g = TwoDGrid(nx, Lx)
  phi0 = fill(1, (g.nx, g.ny))
  q0 = @. sin(k*g.X)

  forcing = @. im/2*sin(k*g.X)
  forcingh = fft(forcing)

  function calcF!(Fc, Fr, t, s, v, p, g)
    Fc .= forcing
    nothing
  end

  prob = NIWQG.Problem(nx=nx, Lx=Lx, dt=dt, eta=eta, calcF=calcF!)
  NIWQG.set_q!(prob, q0)
  NIWQG.set_phi!(prob, phi0)

  stepforward!(prob, nsteps)
  NIWQG.updatevars!(prob)

  rtol = 1e-13
  isapprox(prob.vars.q, q0, rtol=rtol) && isapprox(prob.vars.phi, phi0, rtol=rtol)
end

function test_niwqg_nonlinear2(nk=2, nl=3; nx=32, Lx=2π, dt=0.1, nsteps=100, eta=0.2)
  k = 2π/Lx * nk
  l = 2π/Lx * nl
  g = TwoDGrid(nx, Lx)
  x, y = g.X, g.Y
  phi0 = @. exp(im*l*y)
  q0 = @. sin(k*x)

  forcing = @. im*exp(im*l*y) * (cos(k*x)*(0.5-l/k) + l^2*eta/2)
  forcingh = fft(forcing)

  function calcF!(Fc, Fr, t, s, v, p, g)
    Fc .= forcing
    nothing
  end

  prob = NIWQG.Problem(nx=nx, Lx=Lx, dt=dt, eta=eta, calcF=calcF!)
  NIWQG.set_q!(prob, q0)
  NIWQG.set_phi!(prob, phi0)

  stepforward!(prob, nsteps)
  NIWQG.updatevars!(prob)

  rtol = 1e-13
  isapprox(prob.vars.q, q0, rtol=rtol) && isapprox(prob.vars.phi, phi0, rtol=rtol)
end

@test test_niwqg_lambdipole(256, 1e-3)
@test test_niwqg_groupvelocity(16)
@test test_niwqg_nonlinear1()
@test test_niwqg_nonlinear2()
