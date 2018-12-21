using PyPlot

"""
    constructtestfields()

Constructs flow fields for a 2-layer problem with parameters such that
    q1 = Δψ1 + 25*(ψ2-ψ1),
    q2 = Δψ2 + 25/4*(ψ1-ψ2).
"""
function constructtestfields(gr)
  x, y = gridpoints(gr)
  k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers

  # a set of streafunctions ψ1 and ψ2, ...
  ψ1 = @. 1e-3 * ( 1/4*cos(2k0*x)*cos(5l0*y) + 1/3*cos(3k0*x)*cos(3l0*y) )
  ψ2 = @. 1e-3 * (     cos(3k0*x)*cos(4l0*y) + 1/2*cos(4k0*x)*cos(2l0*y) )

  # ... their corresponding PVs q1, q2,
  q1 = @. 1e-3 * ( 1/6 *(   75cos(4k0*x)*cos(2l0*y) +   2cos(3k0*x)*( -43cos(3l0*y) + 75cos(4l0*y) ) - 81cos(2k0*x)*cos(5l0*y) ) )
  q2 = @. 1e-3 * ( 1/48*( -630cos(4k0*x)*cos(2l0*y) + 100cos(3k0*x)*(    cos(3l0*y) - 15cos(4l0*y) ) + 75cos(2k0*x)*cos(5l0*y) ) )

  # ... and various derived fields, e.g., ∂ψ1/∂x,
  ψ1x = @. 1e-3 * (     -k0/2*sin(2k0*x)*cos(5l0*y) -       k0*sin(3k0*x)*cos(3l0*y) )
  ψ2x = @. 1e-3 * (      -3k0*sin(3k0*x)*cos(4l0*y) -      2k0*sin(4k0*x)*cos(2l0*y) )
  Δψ2 = @. 1e-3 * ( -25*k0*l0*cos(3k0*x)*cos(4l0*y) - 10*k0*l0*cos(4k0*x)*cos(2l0*y) )

  q1x = @. 1e-3 * ( 1/6 *( -4k0*75sin(4k0*x)*cos(2l0*y) - 3k0*2sin(3k0*x)*( -43cos(3l0*y) + 75cos(4l0*y) ) + 2k0*81sin(2k0*x)*cos(5l0*y) ) )
  q2x = @. 1e-3 * ( 1/48*( 4k0*630sin(4k0*x)*cos(2l0*y) - 3k0*100sin(3k0*x)*(    cos(3l0*y) - 15cos(4l0*y) ) - 2k0*75sin(2k0*x)*cos(5l0*y) ) )

  Δq1 = @. 1e-3 *(k0*l0)*( 1/6 *( -20* 75cos(4k0*x)*cos(2l0*y) +   2cos(3k0*x)*( +18*43cos(3l0*y) - 25*75cos(4l0*y) ) +29*81cos(2k0*x)*cos(5l0*y) ) )
  Δq2 = @. 1e-3 *(k0*l0)*( 1/48*( +20*630cos(4k0*x)*cos(2l0*y) + 100cos(3k0*x)*(    -18cos(3l0*y) + 25*15cos(4l0*y) ) -29*75cos(2k0*x)*cos(5l0*y) ) )

  return ψ1, ψ2, q1, q2, ψ1x, ψ2x, q1x, q2x, Δψ2, Δq1, Δq2
end




"""
    test_pvtofromstreamfunction()

Tests the pvfromstreamfunction function that gives qh from ψh. To do so, it
constructs a 2-layer problem with parameters such that
    q1 = Δψ1 + 25*(ψ2-ψ1), q2 = Δψ2 + 25/4*(ψ1-ψ2).
Then given ψ1 and ψ2 it tests if pvfromstreamfunction gives the expected
q1 and q2. Similarly, that streamfunctionfrompv gives ψ1 and ψ2 from q1 and q2.
"""
function test_pvtofromstreamfunction()
   n, L = 128, 2π
   gr = TwoDGrid(n, L)

   nlayers = 2       # these choice of params
   f0, g = 1, 1      # gives the desired
    H = [0.2, 0.8]   # q1 = Δψ1 + 25*(ψ2-ψ1), and
  rho = [5.0, 6.0]   # q2 = Δψ2 + 25/4*(ψ1-ψ2).

  beta = 0.0

   U = zeros(nlayers)
   u = zeros(gr.ny, nlayers)

  mu, nu, nnu = 0.0, 0.0, 1

  x, y = gridpoints(gr)
  k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers

  η = @. 0*x

  pr = MultilayerQG.Params(nlayers, g, f0, beta, rho, H, U, u, η, mu, nu, nnu, gr)
  eq = MultilayerQG.Equation(pr, gr)
  vs = MultilayerQG.Vars(gr, pr)

  ψ1, ψ2, q1, q2, ψ1x, ψ2x, q1x, q2x, Δψ2, Δq1, Δq2 = constructtestfields(gr)

  vs.ψh[:, :, 1] .= rfft(ψ1)
  vs.ψh[:, :, 2] .= rfft(ψ2)

  MultilayerQG.pvfromstreamfunction!(vs.qh, vs.ψh, pr.S, gr)
  MultilayerQG.invtransform!(vs.q, vs.qh, pr)

  vs.qh[:, :, 1] .= rfft(q1)
  vs.qh[:, :, 2] .= rfft(q2)

  MultilayerQG.streamfunctionfrompv!(vs.ψh, vs.qh, pr.invS, gr)
  MultilayerQG.invtransform!(vs.ψ, vs.ψh, pr)

  isapprox(q1, vs.q[:, :, 1], rtol=rtol_multilayerqg) && isapprox(q2, vs.q[:, :, 2], rtol=rtol_multilayerqg) && isapprox(ψ1, vs.ψ[:, :, 1], rtol=rtol_multilayerqg) && isapprox(ψ2, vs.ψ[:, :, 2], rtol=rtol_multilayerqg)
end


"""
    test_bqg_nonlinearadvection(dt, stepper; kwargs...)

Tests the advection term by timestepping a test problem with timestep dt and
timestepper identified by the string stepper. The test 2-layer problem is
derived by picking a solution q1f, q2f (with streamfunctions ψ1f, ψ2f) for which
the advection terms J(ψj, qj) are non-zero. Next, a forcing Ff is derived such
that a solution to the problem forced by this Ff is then qf.
(This solution may not be realized, at least at long times, if it is unstable.)
"""
function test_bqg_advection(dt, stepper; n=128, L=2π, nlayers=2, mu=0.0, nu=0.0, nnu=1, message=false)

  tf = 1*dt
  nt = round(Int, tf/dt)

  nlayers = 2
  n, L = 64, 2π
  gr = TwoDGrid(n, L)

  x, y = gridpoints(gr)
  k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers

  beta = 0.35
    f0 = 1
     g = 1
     U = [0.1, 0.05]
     u = zeros(gr.ny, nlayers)
     ampl = 1.0
     # u[:, 1] = ampl*0.10sin.(2l0*gr.y)
     # u[:, 2] = ampl*0.02cos.(3l0*gr.y)
     # u[:, 1] = 0.4*exp.(0*gr.y)
     # u[:, 2] = 0.3*exp.(0*gr.y)
     H = [0.2, 0.8]
   rho = [5.0, 6.0]

  mu, nu, nnu = 0.1, 0.0, 1

  uyy = zeros(gr.ny, nlayers)
  # uyy[:, 1] = 4*exp.(0*gr.y)
  # uyy[:, 2] = 2*exp.(0*gr.y)
  # uyy[:, 1] = -ampl*(2l0)^2*u[:, 1]
  # uyy[:, 2] = -ampl*(3l0)^2*u[:, 2]
  uyy[:, 1] = real.(ifft( -transpose(gr.l.^2) .* fft(u[:, 1]) ))
  uyy[:, 2] = real.(ifft( -transpose(gr.l.^2) .* fft(u[:, 2]) ))

  η0, σx, σy = 1.0, L/25, L/20
  η = @. η0*exp( -(x+L/8)^2/(2σx^2) -(y-L/8)^2/(2σy^2) )
  ηx = @. -(x+L/8)/(σx^2) * η
  η = @. η0*sin(2k0*x)*sin(2l0*y) + 2*η0*sin(2k0*x) + η0
  ηx = @. 2k0*η0*cos(2k0*x)*sin(2l0*y) + 2k0*2η0*cos(2k0*x)

  ψ1, ψ2, q1, q2, ψ1x, ψ2x, q1x, q2x, Δψ2, Δq1, Δq2 = constructtestfields(gr)

  #Ff = J(ψ, q)

  Ff1 = FourierFlows.jacobian(ψ1, q1, gr)     + beta*ψ1x + ( -uyy[:, 1] .-   25*(U[2].+u[:, 2].-U[1].-u[:, 1]) ).*ψ1x + (U[1].+u[:, 1]).*q1x - nu*Δq1
  Ff2 = FourierFlows.jacobian(ψ2, q2 + η, gr) + beta*ψ2x + ( -uyy[:, 2] .- 25/4*(U[1].+u[:, 1].-U[2].-u[:, 2]) ).*ψ2x + (U[2].+u[:, 2]).*(q2x + ηx) + mu*Δψ2 - nu*Δq2

  Ff = zeros(gr.nx, gr.ny, nlayers)
  Ff[:, :, 1] .= Ff1
  Ff[:, :, 2] .= Ff2

  Ffh = zeros(Complex{Float64}, gr.nkr, gr.nl, nlayers)
  Ffh[:, :, 1] .= rfft(Ff1)
  Ffh[:, :, 2] .= rfft(Ff2)

  function calcFq!(Fqh, sol, t, cl, v, p, g)
    Fqh .= Ffh
    nothing
  end

  pr = MultilayerQG.Params(nlayers, g, f0, beta, rho, H, U, u, η, mu, nu, nnu, gr; calcFq=calcFq!)
  eq = MultilayerQG.Equation(pr, gr)
  vs = MultilayerQG.ForcedVars(gr, pr)

  prob = FourierFlows.Problem(eq, stepper, dt, gr, vs, pr)

  qf = zeros(gr.nx, gr.ny, nlayers)
  qf[:, :, 1] .= q1
  qf[:, :, 2] .= q2

  ψf = zeros(gr.nx, gr.ny, nlayers)
  ψf[:, :, 1] .= ψ1
  ψf[:, :, 2] .= ψ2

  MultilayerQG.set_q!(prob, qf)

  stepforward!(prob, round(Int, nt))
  MultilayerQG.updatevars!(prob)

  figure(1)
  pcolormesh(vs.q[:, :, 1] - qf[:, :, 1])
  colorbar()
  figure(2)
  pcolormesh(vs.q[:, :, 2] - qf[:, :, 2])
  colorbar()
  println(vs.q[1:4, 1:4, 2])
  println(qf[1:4, 1:4, 2])
  isapprox(vs.q, qf, rtol=rtol_multilayerqg) && isapprox(vs.psi, ψf, rtol=rtol_multilayerqg)
end
