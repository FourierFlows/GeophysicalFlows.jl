"""
    constructtestfields_2layer(gr)

Constructs flow fields for a 2-layer problem with parameters such that
    q1 = Δψ1 + 25*(ψ2-ψ1),
    q2 = Δψ2 + 25/4*(ψ1-ψ2).
"""
function constructtestfields_2layer(gr)
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

  Δq1 = @. 1e-3 * (k0*l0)*( 1/6 *( -20* 75cos(4k0*x)*cos(2l0*y) +   2cos(3k0*x)*( +18*43cos(3l0*y) - 25*75cos(4l0*y) ) +29*81cos(2k0*x)*cos(5l0*y) ) )
  Δq2 = @. 1e-3 * (k0*l0)*( 1/48*( +20*630cos(4k0*x)*cos(2l0*y) + 100cos(3k0*x)*(    -18cos(3l0*y) + 25*15cos(4l0*y) ) -29*75cos(2k0*x)*cos(5l0*y) ) )

  return ψ1, ψ2, q1, q2, ψ1x, ψ2x, q1x, q2x, Δψ2, Δq1, Δq2
end


"""
    constructtestfields_3layer(gr)

Constructs flow fields for a 3-layer problem with parameters such that
    q1 = Δψ1 + 20ψ2 - 20ψ1,
    q2 = Δψ2 + 20ψ1 - 44ψ2 + 24ψ3,
    q2 = Δψ2 + 12ψ2 - 12ψ3.
"""
function constructtestfields_3layer(gr)
  x, y = gridpoints(gr)
  k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers

  # a set of streafunctions ψ1, ψ2, ψ3, ...
  ψ1 = @. 1e-3 * ( 1/4*cos(2k0*x)*cos(5l0*y) + 1/3*cos(3k0*x)*cos(3l0*y) )
  ψ2 = @. 1e-3 * (     cos(3k0*x)*cos(4l0*y) + 1/2*cos(4k0*x)*cos(2l0*y) )
  ψ3 = @. 1e-3 * (     cos(1k0*x)*cos(3l0*y) + 1/2*cos(2k0*x)*cos(2l0*y) )
  
  Δψ1 = @. -1e-3 * ( 1/4*((2k0)^2+(5l0)^2)*cos(2k0*x)*cos(5l0*y) + 1/3*((3k0)^2+(3l0)^2)*cos(3k0*x)*cos(3l0*y) )
  Δψ2 = @. -1e-3 * (     ((3k0)^2+(4l0)^2)*cos(3k0*x)*cos(4l0*y) + 1/2*((4k0)^2+(2l0)^2)*cos(4k0*x)*cos(2l0*y) )
  Δψ3 = @. -1e-3 * (     ((1k0)^2+(3l0)^2)*cos(1k0*x)*cos(3l0*y) + 1/2*((2k0)^2+(2l0)^2)*cos(2k0*x)*cos(2l0*y) )

  # ... their corresponding PVs q1, q2, q3,
  q1 = @. Δψ1 + 20ψ2 - 20ψ1
  q2 = @. Δψ2 + 20ψ1 - 44ψ2 + 24ψ3
  q3 = @. Δψ3        + 12ψ2 - 12ψ3

  return ψ1, ψ2, ψ3, q1, q2, q3
end


"""
    test_pvtofromstreamfunction_2layer()

Tests the pvfromstreamfunction function that gives qh from ψh. To do so, it
constructs a 2-layer problem with parameters such that
    q1 = Δψ1 + 25*(ψ2-ψ1), q2 = Δψ2 + 25/4*(ψ1-ψ2).
Then given ψ1 and ψ2 it tests if pvfromstreamfunction gives the expected
q1 and q2. Similarly, that streamfunctionfrompv gives ψ1 and ψ2 from q1 and q2.
"""
function test_pvtofromstreamfunction_2layer()
  n, L = 128, 2π
  gr = TwoDGrid(n, L)

  nlayers = 2      # these choice of parameters give the
  f0, g = 1, 1     # desired PV-streamfunction relations
  H = [0.2, 0.8]   # q1 = Δψ1 + 25*(ψ2-ψ1), and
  ρ = [4.0, 5.0]   # q2 = Δψ2 + 25/4*(ψ1-ψ2).

  prob = MultilayerQG.Problem(nlayers=nlayers, nx=n, Lx=L, f0=f0, g=g, H=H, ρ=ρ)
  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  ψ1, ψ2, q1, q2, ψ1x, ψ2x, q1x, q2x, Δψ2, Δq1, Δq2 = constructtestfields_2layer(gr)

  vs.psih[:, :, 1] .= rfft(ψ1)
  vs.psih[:, :, 2] .= rfft(ψ2)

  MultilayerQG.pvfromstreamfunction!(vs.qh, vs.psih, pr.S, gr)
  MultilayerQG.invtransform!(vs.q, vs.qh, pr)

  vs.qh[:, :, 1] .= rfft(q1)
  vs.qh[:, :, 2] .= rfft(q2)

  MultilayerQG.streamfunctionfrompv!(vs.psih, vs.qh, pr.invS, gr)
  MultilayerQG.invtransform!(vs.psi, vs.psih, pr)

  isapprox(q1, vs.q[:, :, 1], rtol=rtol_multilayerqg) && isapprox(q2, vs.q[:, :, 2], rtol=rtol_multilayerqg) && isapprox(ψ1, vs.psi[:, :, 1], rtol=rtol_multilayerqg) && isapprox(ψ2, vs.psi[:, :, 2], rtol=rtol_multilayerqg)
end


"""
    test_pvtofromstreamfunction_3layer()

Tests the pvfromstreamfunction function that gives qh from ψh. To do so, it
constructs a 3-layer problem with parameters such that
  q1 = Δψ1 + 20ψ2 - 20ψ1, q2 = Δψ2 + 20ψ1 - 44ψ2 + 24ψ3, q3 = Δψ3 + 12ψ2 - 12ψ3.
Then given ψ1, ψ2, and ψ2 it tests if pvfromstreamfunction gives the expected
q1, q2, and q3. Similarly, that streamfunctionfrompv gives ψ1, ψ2, and ψ2 from 
q1, q2, and q3.
"""
function test_pvtofromstreamfunction_3layer()
  n, L = 128, 2π
  gr = TwoDGrid(n, L)

  nlayers = 3      # these choice of parameters give the
  f0, g = 1, 1     # desired PV-streamfunction relations
  H = [0.25, 0.25, 0.5]  # q1 = Δψ1 + 20ψ2 - 20ψ1,
  ρ = [4.0, 5.0, 6.0]    # q2 = Δψ2 + 20ψ1 - 44ψ2 + 24ψ3,
                         # q3 = Δψ3        + 12ψ2 - 12ψ3.

  prob = MultilayerQG.Problem(nlayers=nlayers, nx=n, Lx=L, f0=f0, g=g, H=H, ρ=ρ)
  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  ψ1, ψ2, ψ3, q1, q2, q3 = constructtestfields_3layer(gr)

  vs.psih[:, :, 1] .= rfft(ψ1)
  vs.psih[:, :, 2] .= rfft(ψ2)
  vs.psih[:, :, 3] .= rfft(ψ3)

  MultilayerQG.pvfromstreamfunction!(vs.qh, vs.psih, pr.S, gr)
  MultilayerQG.invtransform!(vs.q, vs.qh, pr)

  vs.qh[:, :, 1] .= rfft(q1)
  vs.qh[:, :, 2] .= rfft(q2)
  vs.qh[:, :, 3] .= rfft(q3)

  MultilayerQG.streamfunctionfrompv!(vs.psih, vs.qh, pr.invS, gr)
  MultilayerQG.invtransform!(vs.psi, vs.psih, pr)

  isapprox(q1, vs.q[:, :, 1], rtol=rtol_multilayerqg) && isapprox(q2, vs.q[:, :, 2], rtol=rtol_multilayerqg) && isapprox(q3, vs.q[:, :, 3], rtol=rtol_multilayerqg) && isapprox(ψ1, vs.psi[:, :, 1], rtol=rtol_multilayerqg) && isapprox(ψ2, vs.psi[:, :, 2], rtol=rtol_multilayerqg) && isapprox(ψ3, vs.psi[:, :, 3], rtol=rtol_multilayerqg)
end


"""
    test_mqg_nonlinearadvection(dt, stepper; kwargs...)

Tests the advection term by timestepping a test problem with timestep dt and
timestepper identified by the string stepper. The test 2-layer problem is
derived by picking a solution q1f, q2f (with streamfunctions ψ1f, ψ2f) for which
the advection terms J(ψn, qn) are non-zero. Next, a forcing Ff is derived such
that a solution to the problem forced by this Ff is then qf.
(This solution may not be realized, at least at long times, if it is unstable.)
"""
function test_mqg_nonlinearadvection(dt, stepper; n=128, L=2π, nlayers=2, μ=0.0, ν=0.0, nν=1)
  tf = 0.5
  nt = round(Int, tf/dt)

  nx, ny = 64, 66
  Lx, Ly = 2π, 2π
  gr = TwoDGrid(nx, Lx, ny, Ly)

  x, y = gridpoints(gr)
  k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers

  nlayers = 2       # these choice of parameters give the
  f0, g = 1, 1      # desired PV-streamfunction relations
  H = [0.2, 0.8]   # q1 = Δψ1 + 25*(ψ2-ψ1), and
  ρ = [4.0, 5.0]   # q2 = Δψ2 + 25/4*(ψ1-ψ2).

  β = 0.35

  U1, U2 = 0.1, 0.05
  u1 = @. 0.5sech(gr.y/(Ly/15))^2
  u2 = @. 0.02cos(3l0*gr.y)
  uyy1 = real.(ifft( -gr.l.^2 .* fft(u1) ))
  uyy2 = real.(ifft( -gr.l.^2 .* fft(u2) ))

  U = zeros(ny, nlayers)
  U[:, 1] = u1 .+ U1
  U[:, 2] = u2 .+ U2

  μ, ν, nν = 0.1, 0.05, 1

  η0, σx, σy = 1.0, Lx/25, Ly/20
  η = @. η0*exp( -(x+Lx/8)^2/(2σx^2) -(y-Ly/8)^2/(2σy^2) )
  ηx = @. -(x+Lx/8)/(σx^2) * η

  ψ1, ψ2, q1, q2, ψ1x, ψ2x, q1x, q2x, Δψ2, Δq1, Δq2 = constructtestfields_2layer(gr)

  Ff1 = FourierFlows.jacobian(ψ1, q1, gr)     + (β .- uyy1 .-   25*(U2.+u2.-U1.-u1) ).*ψ1x + (U1.+u1).*q1x - ν*Δq1
  Ff2 = FourierFlows.jacobian(ψ2, q2 + η, gr) + (β .- uyy2 .- 25/4*(U1.+u1.-U2.-u2) ).*ψ2x + (U2.+u2).*(q2x + ηx) + μ*Δψ2 - ν*Δq2

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

  prob = MultilayerQG.Problem(nlayers=nlayers, nx=nx, ny=ny, Lx=Lx, Ly=Ly, f0=f0,
          g=g, H=H, ρ=ρ, U=U, eta=η, β=β, μ=μ, ν=ν, nν=nν, calcFq=calcFq!)
  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  qf = zeros(gr.nx, gr.ny, nlayers)
  qf[:, :, 1] .= q1
  qf[:, :, 2] .= q2

  ψf = zeros(gr.nx, gr.ny, nlayers)
  ψf[:, :, 1] .= ψ1
  ψf[:, :, 2] .= ψ2

  MultilayerQG.set_q!(prob, qf)

  stepforward!(prob, round(Int, nt))
  MultilayerQG.updatevars!(prob)

  isapprox(vs.q, qf, rtol=rtol_multilayerqg) && isapprox(vs.psi, ψf, rtol=rtol_multilayerqg)
end

"""
    test_mqg_linearadvection(dt, stepper; kwargs...)

Tests the advection term of the linearized equations by timestepping a test
problem with timestep dt and timestepper identified by the string stepper.
The test 2-layer problem is derived by picking a solution q1f, q2f (with
streamfunctions ψ1f, ψ2f) for which the advection terms J(ψn, qn) are non-zero.
Next, a forcing Ff is derived such that a solution to the problem forced by this
Ff is then qf. (This solution may not be realized, at least at long times, if it
is unstable.)
"""
function test_mqg_linearadvection(dt, stepper; n=128, L=2π, nlayers=2, μ=0.0, ν=0.0, nν=1)
  tf = 0.5
  nt = round(Int, tf/dt)

  nx, ny = 64, 66
  Lx, Ly = 2π, 2π
  gr = TwoDGrid(nx, Lx, ny, Ly)

  x, y = gridpoints(gr)
  k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers

  nlayers = 2       # these choice of parameters give the
  f0, g = 1, 1      # desired PV-streamfunction relations
  H = [0.2, 0.8]   # q1 = Δψ1 + 25*(ψ2-ψ1), and
  ρ = [4.0, 5.0]   # q2 = Δψ2 + 25/4*(ψ1-ψ2).

  β = 0.35

  U1, U2 = 0.1, 0.05
  u1 = @. 0.5sech(gr.y/(Ly/15))^2
  u2 = @. 0.02cos(3l0*gr.y)
  uyy1 = real.(ifft( -gr.l.^2 .* fft(u1) ))
  uyy2 = real.(ifft( -gr.l.^2 .* fft(u2) ))

  U = zeros(ny, nlayers)
  U[:, 1] = u1 .+ U1
  U[:, 2] = u2 .+ U2

  μ, ν, nν = 0.1, 0.05, 1

  η0, σx, σy = 1.0, Lx/25, Ly/20
  η = @. η0*exp( -(x+Lx/8)^2/(2σx^2) -(y-Ly/8)^2/(2σy^2) )
  ηx = @. -(x+Lx/8)/(σx^2) * η

  ψ1, ψ2, q1, q2, ψ1x, ψ2x, q1x, q2x, Δψ2, Δq1, Δq2 = constructtestfields_2layer(gr)

  Ff1 = (β .- uyy1 .-   25*(U2.+u2.-U1.-u1) ).*ψ1x + (U1.+u1).*q1x - ν*Δq1
  Ff2 = FourierFlows.jacobian(ψ2, η, gr) + (β .- uyy2 .- 25/4*(U1.+u1.-U2.-u2) ).*ψ2x + (U2.+u2).*(q2x + ηx) + μ*Δψ2 - ν*Δq2

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

  prob = MultilayerQG.Problem(nlayers=nlayers, nx=nx, ny=ny, Lx=Lx, Ly=Ly, f0=f0,
          g=g, H=H, ρ=ρ, U=U, eta=η, β=β, μ=μ, ν=ν, nν=nν, calcFq=calcFq!, linear=true)
  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid


  qf = zeros(gr.nx, gr.ny, nlayers)
  qf[:, :, 1] .= q1
  qf[:, :, 2] .= q2

  ψf = zeros(gr.nx, gr.ny, nlayers)
  ψf[:, :, 1] .= ψ1
  ψf[:, :, 2] .= ψ2

  MultilayerQG.set_q!(prob, qf)

  stepforward!(prob, round(Int, nt))
  MultilayerQG.updatevars!(prob)

  isapprox(vs.q, qf, rtol=rtol_multilayerqg) && isapprox(vs.psi, ψf, rtol=rtol_multilayerqg)
end

"""
    test_mqg_energies(dt, stepper; kwargs...)

Tests the kinetic (KE) and potential (PE) energies function by constructing a
2-layer problem and initializing it with a flow field whose KE and PE are known.
"""
function test_mqg_energies(; dt=0.001, stepper="ForwardEuler", n=128, L=2π, nlayers=2, μ=0.0, ν=0.0, nν=1)
  nx, ny = 64, 66
  Lx, Ly = 2π, 2π
  gr = TwoDGrid(nx, Lx, ny, Ly)

  x, y = gridpoints(gr)
  k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers

  nlayers = 2      # these choice of parameters give the
  f0, g = 1, 1     # desired PV-streamfunction relations
  H = [0.2, 0.8]   # q1 = Δψ1 + 25*(ψ2-ψ1), and
  ρ = [4.0, 5.0]   # q2 = Δψ2 + 25/4*(ψ1-ψ2).

  prob = MultilayerQG.Problem(nlayers=nlayers, nx=nx, ny=ny, Lx=Lx, Ly=Ly, f0=f0, g=g, H=H, ρ=ρ)
  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  ψ1, ψ2, q1, q2, ψ1x, ψ2x, q1x, q2x, Δψ2, Δq1, Δq2 = constructtestfields_2layer(gr)

  qf = zeros(gr.nx, gr.ny, nlayers)
  qf[:, :, 1] .= q1
  qf[:, :, 2] .= q2

  MultilayerQG.set_q!(prob, qf)

  KE, PE = MultilayerQG.energies(prob)

  isapprox(KE[1], 61/640*1e-6, rtol=rtol_multilayerqg) && isapprox(KE[2], 3*1e-6, rtol=rtol_multilayerqg) && isapprox(PE[1], 1025/1152*1e-6, rtol=rtol_multilayerqg) && MultilayerQG.addforcing!(prob.timestepper.RHS₁, sol, cl.t, cl, vs, pr, gr)==nothing
end

"""
    test_mqg_fluxes(dt, stepper; kwargs...)

Tests the lateral and vertical eddy fluxes by constructing a 2-layer problem and
initializing it with a flow field whose fluxes are known.
"""
function test_mqg_fluxes(; dt=0.001, stepper="ForwardEuler", n=128, L=2π, nlayers=2, μ=0.0, ν=0.0, nν=1)
  nx, ny = 128, 126
  Lx, Ly = 2π, 2π
  gr = TwoDGrid(nx, Lx, ny, Ly)

  x, y = gridpoints(gr)
  k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers

  nlayers = 2      # these choice of parameters give the
  f0, g = 1, 1     # desired PV-streamfunction relations
  H = [0.2, 0.8]   # q1 = Δψ1 + 25*(ψ2-ψ1), and
  ρ = [4.0, 5.0]   # q2 = Δψ2 + 25/4*(ψ1-ψ2).
  U = zeros(ny, nlayers)
  U[:, 1] = @. sech(gr.y/0.2)^2

  prob = MultilayerQG.Problem(nlayers=nlayers, nx=nx, ny=ny, Lx=Lx, Ly=Ly, f0=f0, g=g, H=H, ρ=ρ, U=U)
  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  ψ1 = @. cos(k0*x)*cos(l0*y)
  ψ2 = @. cos(k0*x+π/10)*cos(l0*y)
  ψ = zeros(gr.nx, gr.ny, nlayers)
  ψ[:, :, 1] .= ψ1
  ψ[:, :, 2] .= ψ2
  MultilayerQG.set_psi!(prob, ψ)
  lateralfluxes, verticalfluxes = MultilayerQG.fluxes(prob)

  isapprox(lateralfluxes[1], 0, atol=1e-12) && isapprox(lateralfluxes[2], 0, atol=1e-12) && isapprox(verticalfluxes[1], -0.04763511558, rtol=1e-6)
end

"""
    test_setqsetpsi(dt, stepper; kwargs...)

Tests the set_q!() and set_psi!() functions that initialize sol with a flow with
given `q` or `psi` respectively.
"""
function test_mqg_setqsetpsi(; dt=0.001, stepper="ForwardEuler", n=64, L=2π, nlayers=2, μ=0.0, ν=0.0, nν=1)
  nx, ny = 32, 34
  L = 2π
  gr = TwoDGrid(nx, L, ny, L)

  x, y = gridpoints(gr)
  k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers

  nlayers = 2      # these choice of parameters give the
  f0, g = 1, 1     # desired PV-streamfunction relations
  H = [0.2, 0.8]   # q1 = Δψ1 + 25*(ψ2-ψ1), and
  ρ = [4.0, 5.0]   # q2 = Δψ2 + 25/4*(ψ1-ψ2).

  prob = MultilayerQG.Problem(nlayers=nlayers, nx=nx, ny=ny, Lx=L, f0=f0, g=g, H=H, ρ=ρ)
  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  f1 = @. 2cos(k0*x)*cos(l0*y)
  f2 = @.  cos(k0*x+π/10)*cos(2l0*y)
  f = zeros(gr.nx, gr.ny, nlayers)
  f[:, :, 1] .= f1
  f[:, :, 2] .= f2

  ψtest = zeros(size(f))
  MultilayerQG.set_psi!(prob, f)
  @. vs.qh = sol
  MultilayerQG.streamfunctionfrompv!(vs.psih, vs.qh, pr.invS, gr)
  MultilayerQG.invtransform!(ψtest, vs.psih, pr)

  qtest = zeros(size(f))
  MultilayerQG.set_q!(prob, f)
  @. vs.qh = sol
  MultilayerQG.invtransform!(qtest, vs.qh, pr)

  isapprox(ψtest, f, rtol=rtol_multilayerqg) && isapprox(qtest, f, rtol=rtol_multilayerqg)
end

"""
    test_paramsconstructor(; kwargs...)

Tests that `Params` constructor works with both mean flow `U` being a floats
(i.e., constant `U` in each layer) or vectors (i.e., `U(y)` in each layer).
"""
function test_mqg_paramsconstructor(; dt=0.001, stepper="ForwardEuler")
  nx, ny = 32, 34
  L = 2π
  gr = TwoDGrid(nx, L, ny, L)

  nlayers = 2      # these choice of parameters give the
  f0, g = 1, 1     # desired PV-streamfunction relations
  H = [0.2, 0.8]   # q1 = Δψ1 + 25*(ψ2-ψ1), and
  ρ = [4.0, 5.0]   # q2 = Δψ2 + 25/4*(ψ1-ψ2).
  
  U1, U2 = 0.1, 0.05

  Uvectors = zeros(ny, nlayers)
  Uvectors[:, 1] .= U1
  Uvectors[:, 2] .= U2

  Ufloats = zeros(nlayers)
  Ufloats[1] = U1
  Ufloats[2] = U2

  probUvectors = MultilayerQG.Problem(nlayers=nlayers, nx=nx, ny=ny, Lx=L, f0=f0, g=g, H=H, ρ=ρ, U=Uvectors)
  probUfloats = MultilayerQG.Problem(nlayers=nlayers, nx=nx, ny=ny, Lx=L, f0=f0, g=g, H=H, ρ=ρ, U=Ufloats)

  isapprox(probUfloats.params.U, probUvectors.params.U, rtol=rtol_multilayerqg)
end

function test_mqg_problemtype(T=Float32)
  prob = MultilayerQG.Problem(nlayers=2, T=T)

  (typeof(prob.sol)==Array{Complex{T},3} && typeof(prob.grid.Lx)==T && typeof(prob.grid.x)==Array{T,2} && typeof(prob.vars.u)==Array{T,3})
end
