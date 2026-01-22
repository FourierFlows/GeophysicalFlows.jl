"""
    constructtestfields_2layer(gr)

Constructs flow fields for a 2-layer problem with parameters such that
    q1 = Δψ1 + 25*(ψ2-ψ1),
    q2 = Δψ2 + 25/4*(ψ1-ψ2).
"""
function constructtestfields_2layer(gr)
  x, y = gridpoints(gr)
  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly # fundamental wavenumbers

  # a set of streafunctions ψ1 and ψ2, ...
  ψ1 = @. 1e-3 * ( 1/4 * cos(2k₀*x) * cos(5l₀*y) +
                   1/3 * cos(3k₀*x) * cos(3l₀*y) )
  ψ2 = @. 1e-3 * (       cos(3k₀*x) * cos(4l₀*y) +
                   1/2 * cos(4k₀*x) * cos(2l₀*y) )

  Δψ1 = @. 1e-3 * ( - 1/4 * ((2k₀)^2 + (5l₀)^2) * cos(2k₀*x) * cos(5l₀*y) +
                    - 1/3 * ((3k₀)^2 + (3l₀)^2) * cos(3k₀*x) * cos(3l₀*y) )
  Δψ2 = @. 1e-3 * (       - ((3k₀)^2 + (4l₀)^2) * cos(3k₀*x) * cos(4l₀*y) +
                    - 1/2 * ((4k₀)^2 + (2l₀)^2) * cos(4k₀*x) * cos(2l₀*y) )

  Δ²ψ1 = @. 1e-3 * ( + 1/4 * ((2k₀)^2 + (5l₀)^2)^2 * cos(2k₀*x) * cos(5l₀*y) +
                     + 1/3 * ((3k₀)^2 + (3l₀)^2)^2 * cos(3k₀*x) * cos(3l₀*y) )
  Δ²ψ2 = @. 1e-3 * (       + ((3k₀)^2 + (4l₀)^2)^2 * cos(3k₀*x) * cos(4l₀*y) +
                     + 1/2 * ((4k₀)^2 + (2l₀)^2)^2 * cos(4k₀*x) * cos(2l₀*y) )

  # ... their corresponding PVs q1, q2,
  q1 = @. Δψ1 +   25 * ψ2 -   25 * ψ1
  q2 = @. Δψ2 + 25/4 * ψ1 - 25/4 * ψ2

  # ... and various derived fields, e.g., ∂ψ1/∂x,s
  ψ1x = @. 1e-3 * ( - 1/4 * 2k₀ * sin(2k₀*x) * cos(5l₀*y) +
                    - 1/3 * 3k₀ * sin(3k₀*x) * cos(3l₀*y) )
  ψ2x = @. 1e-3 * (       - 3k₀ * sin(3k₀*x) * cos(4l₀*y) +
                    - 1/2 * 4k₀ * sin(4k₀*x) * cos(2l₀*y) )

  Δψ1x = @. 1e-3 * ( + 1/4 * 2k₀ * ((2k₀)^2 + (5l₀)^2) * sin(2k₀*x) * cos(5l₀*y) +
                     + 1/3 * 3k₀ * ((3k₀)^2 + (3l₀)^2) * sin(3k₀*x) * cos(3l₀*y) )
  Δψ2x = @. 1e-3 * (       + 3k₀ * ((3k₀)^2 + (4l₀)^2) * sin(3k₀*x) * cos(4l₀*y) +
                     + 1/2 * 4k₀ * ((4k₀)^2 + (2l₀)^2) * sin(4k₀*x) * cos(2l₀*y) )

  # ... their corresponding PVs q1, q2, q3,
  Δq1 = @. Δ²ψ1 +   25 * Δψ2 -   25 * Δψ1
  Δq2 = @. Δ²ψ2 + 25/4 * Δψ1 - 25/4 * Δψ2

  q1x = @. Δψ1x +   25 * ψ2x -   25 * ψ1x
  q2x = @. Δψ2x + 25/4 * ψ1x - 25/4 * ψ2x

  return ψ1, ψ2, q1, q2, ψ1x, ψ2x, q1x, q2x, Δψ2, Δq1, Δq2
end

"""
    constructtestfields_3layer(gr)

Constructs flow fields for a 3-layer problem with parameters such that
    q1 = Δψ1 + 20ψ2 - 20ψ1,
    q2 = Δψ2 + 20ψ1 - 44ψ2 + 24ψ3,
    q3 = Δψ3 + 12ψ2 - 12ψ3.
"""
function constructtestfields_3layer(gr)
  x, y = gridpoints(gr)
  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly # fundamental wavenumbers

  # a set of streafunctions ψ1, ψ2, ψ3, ...
  ψ1 = @. 1e-3 * ( 1/4 * cos(2k₀*x) * cos(5l₀*y) +
                   1/3 * cos(3k₀*x) * cos(3l₀*y) )
  ψ2 = @. 1e-3 * (       cos(3k₀*x) * cos(4l₀*y) +
                   1/2 * cos(4k₀*x) * cos(2l₀*y) )
  ψ3 = @. 1e-3 * (       cos(1k₀*x) * cos(3l₀*y) +
                   1/2 * cos(2k₀*x) * cos(2l₀*y) )

  Δψ1 = @. 1e-3 * ( - 1/4 * ((2k₀)^2 + (5l₀)^2) * cos(2k₀*x) * cos(5l₀*y) +
                    - 1/3 * ((3k₀)^2 + (3l₀)^2) * cos(3k₀*x) * cos(3l₀*y) )
  Δψ2 = @. 1e-3 * (       - ((3k₀)^2 + (4l₀)^2) * cos(3k₀*x) * cos(4l₀*y) +
                    - 1/2 * ((4k₀)^2 + (2l₀)^2) * cos(4k₀*x) * cos(2l₀*y) )
  Δψ3 = @. 1e-3 * (       - ((1k₀)^2 + (3l₀)^2) * cos(1k₀*x) * cos(3l₀*y) +
                    - 1/2 * ((2k₀)^2 + (2l₀)^2) * cos(2k₀*x) * cos(2l₀*y) )

  Δ²ψ1 = @. 1e-3 * ( + 1/4 * ((2k₀)^2 + (5l₀)^2)^2 * cos(2k₀*x) * cos(5l₀*y) +
                     + 1/3 * ((3k₀)^2 + (3l₀)^2)^2 * cos(3k₀*x) * cos(3l₀*y) )
  Δ²ψ2 = @. 1e-3 * (       + ((3k₀)^2 + (4l₀)^2)^2 * cos(3k₀*x) * cos(4l₀*y) +
                     + 1/2 * ((4k₀)^2 + (2l₀)^2)^2 * cos(4k₀*x) * cos(2l₀*y) )
  Δ²ψ3 = @. 1e-3 * (       + ((1k₀)^2 + (3l₀)^2)^2 * cos(1k₀*x) * cos(3l₀*y) +
                     + 1/2 * ((2k₀)^2 + (2l₀)^2)^2 * cos(2k₀*x) * cos(2l₀*y) )

  # ... their corresponding PVs q1, q2, q3,
  q1 = @. Δψ1  + 20ψ2  - 20ψ1
  q2 = @. Δψ2  + 20ψ1  - 44ψ2  + 24ψ3
  q3 = @. Δψ3          + 12ψ2  - 12ψ3

  # ... and various derived fields, e.g., ∂ψ1/∂x,
  ψ1x = @. 1e-3 * ( - 1/4 * 2k₀ * sin(2k₀*x) * cos(5l₀*y) +
                    - 1/3 * 3k₀ * sin(3k₀*x) * cos(3l₀*y) )
  ψ2x = @. 1e-3 * (       - 3k₀ * sin(3k₀*x) * cos(4l₀*y) +
                    - 1/2 * 4k₀ * sin(4k₀*x) * cos(2l₀*y) )
  ψ3x = @. 1e-3 * (       - 1k₀ * sin(1k₀*x) * cos(3l₀*y) +
                    - 1/2 * 2k₀ * sin(2k₀*x) * cos(2l₀*y) )

  Δψ1x = @. 1e-3 * ( + 1/4 * 2k₀ * ((2k₀)^2 + (5l₀)^2) * sin(2k₀*x) * cos(5l₀*y) +
                     + 1/3 * 3k₀ * ((3k₀)^2 + (3l₀)^2) * sin(3k₀*x) * cos(3l₀*y) )
  Δψ2x = @. 1e-3 * (       + 3k₀ * ((3k₀)^2 + (4l₀)^2) * sin(3k₀*x) * cos(4l₀*y) +
                     + 1/2 * 4k₀ * ((4k₀)^2 + (2l₀)^2) * sin(4k₀*x) * cos(2l₀*y) )
  Δψ3x = @. 1e-3 * (       + 1k₀ * ((1k₀)^2 + (3l₀)^2) * sin(1k₀*x) * cos(3l₀*y) +
                     + 1/2 * 2k₀ * ((2k₀)^2 + (2l₀)^2) * sin(2k₀*x) * cos(2l₀*y) )

  q1x = @. Δψ1x + 20ψ2x - 20ψ1x
  q2x = @. Δψ2x + 20ψ1x - 44ψ2x + 24ψ3x
  q3x = @. Δψ3x         + 12ψ2x - 12ψ3x

  Δq1 = @. Δ²ψ1 + 20Δψ2 - 20Δψ1
  Δq2 = @. Δ²ψ2 + 20Δψ1 - 44Δψ2 + 24Δψ3
  Δq3 = @. Δ²ψ3         + 12Δψ2 - 12Δψ3

  return ψ1, ψ2, ψ3, q1, q2, q3, ψ1x, ψ2x, ψ3x, q1x, q2x, q3x, Δq1, Δq2, Δq3, Δψ3
end


"""
    test_pvtofromstreamfunction_2layer(dev)

Tests the pvfromstreamfunction function that gives qh from ψh. To do so, it
constructs a 2-layer problem with parameters such that
    q1 = Δψ1 + 25*(ψ2-ψ1), q2 = Δψ2 + 25/4*(ψ1-ψ2).
Then given ψ1 and ψ2 it tests if pvfromstreamfunction gives the expected
q1 and q2. Similarly, that streamfunctionfrompv gives ψ1 and ψ2 from q1 and q2.
"""
function test_pvtofromstreamfunction_2layer(dev::Device=CPU())
  n, L = 128, 2π
  gr = TwoDGrid(dev; nx=n, Lx=L)

  nlayers = 2      # these choice of parameters give the
  f₀ = 1           # desired PV-streamfunction relations
  H = [0.2, 0.8]   # q1 = Δψ1 + 25*(ψ2-ψ1), and
  b = [-1.0, -1.2] # q2 = Δψ2 + 25/4*(ψ1-ψ2).

  prob = MultiLayerQG.Problem(nlayers, dev; nx=n, Lx=L, f₀, H, b)
  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  ψ1, ψ2, q1, q2, ψ1x, ψ2x, q1x, q2x, Δψ2, Δq1, Δq2 = constructtestfields_2layer(gr)

  vs.ψh[:, :, 1] .= rfft(ψ1)
  vs.ψh[:, :, 2] .= rfft(ψ2)

  MultiLayerQG.pvfromstreamfunction!(vs.qh, vs.ψh, pr, gr)
  MultiLayerQG.invtransform!(vs.q, vs.qh, pr)

  vs.qh[:, :, 1] .= rfft(q1)
  vs.qh[:, :, 2] .= rfft(q2)

  MultiLayerQG.streamfunctionfrompv!(vs.ψh, vs.qh, pr, gr)
  MultiLayerQG.invtransform!(vs.ψ, vs.ψh, pr)

  return isapprox(q1, vs.q[:, :, 1], rtol=rtol_multilayerqg) &&
         isapprox(q2, vs.q[:, :, 2], rtol=rtol_multilayerqg) &&
         isapprox(ψ1, vs.ψ[:, :, 1], rtol=rtol_multilayerqg) &&
         isapprox(ψ2, vs.ψ[:, :, 2], rtol=rtol_multilayerqg)
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
function test_pvtofromstreamfunction_3layer(dev::Device=CPU())
  n, L = 128, 2π
  gr = TwoDGrid(dev; nx=n, Lx=L)

  nlayers = 3                 # these choice of parameters give the
  f₀ = 1                      # desired PV-streamfunction relations
  H = [0.25, 0.25, 0.5]       # q1 = Δψ1 + 20ψ2 - 20ψ1,
  b = [-1.0, -1.2, -1.2-1/6]  # q2 = Δψ2 + 20ψ1 - 44ψ2 + 24ψ3,
                              # q3 = Δψ3        + 12ψ2 - 12ψ3.

  prob = MultiLayerQG.Problem(nlayers, dev; nx=n, Lx=L, f₀, H, b)
  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  ψ1, ψ2, ψ3, q1, q2, q3, ψ1x, ψ2x, ψ3x, q1x, q2x, q3x, Δq1, Δq2, Δq3, Δψ3 = constructtestfields_3layer(gr)

  vs.ψh[:, :, 1] .= rfft(ψ1)
  vs.ψh[:, :, 2] .= rfft(ψ2)
  vs.ψh[:, :, 3] .= rfft(ψ3)

  MultiLayerQG.pvfromstreamfunction!(vs.qh, vs.ψh, pr, gr)
  MultiLayerQG.invtransform!(vs.q, vs.qh, pr)

  vs.qh[:, :, 1] .= rfft(q1)
  vs.qh[:, :, 2] .= rfft(q2)
  vs.qh[:, :, 3] .= rfft(q3)

  MultiLayerQG.streamfunctionfrompv!(vs.ψh, vs.qh, pr, gr)
  MultiLayerQG.invtransform!(vs.ψ, vs.ψh, pr)

  MultiLayerQG.set_q!(prob, vs.q)

  KE, PE = MultiLayerQG.energies(prob)

  return isapprox(q1, vs.q[:, :, 1], rtol=rtol_multilayerqg) &&
         isapprox(q2, vs.q[:, :, 2], rtol=rtol_multilayerqg) &&
         isapprox(q3, vs.q[:, :, 3], rtol=rtol_multilayerqg) &&
         isapprox(ψ1, vs.ψ[:, :, 1], rtol=rtol_multilayerqg) &&
         isapprox(ψ2, vs.ψ[:, :, 2], rtol=rtol_multilayerqg) &&
         isapprox(ψ3, vs.ψ[:, :, 3], rtol=rtol_multilayerqg) &&
         KE isa Vector{Float64} && length(KE) == nlayers &&
         PE isa Vector{Float64} && length(PE) == nlayers-1
end


"""
    test_mqg_nonlinearadvection_2layers(dt, stepper, dev::Device=CPU())

Tests the advection term by timestepping a test problem with timestep dt and
timestepper identified by the string stepper. The test 2-layer problem is
derived by picking a solution q1f, q2f (with streamfunctions ψ1f, ψ2f) for which
the advection terms J(ψn, qn) are non-zero. Next, a forcing Ff is derived such
that a solution to the problem forced by this Ff is then qf.
(This solution may not be realized, at least at long times, if it is unstable.)
"""
function test_mqg_nonlinearadvection_2layers(dt, stepper, dev::Device=CPU())

  A = device_array(dev)

  tf = 200dt
  nt = round(Int, tf/dt)

  nx, ny = 64, 66
  Lx, Ly = 2π, 2π
  gr = TwoDGrid(dev; nx, Lx, ny, Ly)

  x, y = gridpoints(gr)
  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly # fundamental wavenumbers

  nlayers = 2       # these choice of parameters give the
  f₀ = 1            # desired PV-streamfunction relations
  H = [0.2, 0.8]    # q1 = Δψ1 + 25*(ψ2-ψ1), and
  b = [-1.0, -1.2]  # q2 = Δψ2 + 25/4*(ψ1-ψ2).

  β = 0.35

  U1, U2 = 0.02, 0.025

  u1 = @. 0.05sech(gr.y / (Ly/15))^2; u1 = A(reshape(u1, (1, gr.ny)))
  u2 = @. 0.02cos(3l₀*gr.y);          u2 = A(reshape(u2, (1, gr.ny)))

  uyy1 = real.(ifft( -gr.l.^2 .* fft(u1) ))
  uyy2 = real.(ifft( -gr.l.^2 .* fft(u2) ))

  U = zeros(ny, nlayers)
  CUDA.@allowscalar U[:, 1] = u1 .+ U1
  CUDA.@allowscalar U[:, 2] = u2 .+ U2

  μ, ν, nν = 0.1, 0.05, 1

  η0, σx, σy = 1.0, Lx/25, Ly/20
   η = @. η0 * exp( -(x + Lx/8)^2 / 2σx^2 - (y - Ly/8)^2 / 2σy^2)
  ηx = @. - (x + Lx/8) / σx^2 * η

  ψ1, ψ2, q1, q2, ψ1x, ψ2x, q1x, q2x, Δψ2, Δq1, Δq2 = constructtestfields_2layer(gr)

  Ff1 = FourierFlows.jacobian(ψ1, q1, gr) + @. (β - uyy1 -   25 * ((U2+u2) - (U1+u1))) * ψ1x + (U1+u1) * q1x - ν * Δq1
  Ff2 = FourierFlows.jacobian(ψ2, q2, gr) + @. (β - uyy2 - 25/4 * ((U1+u1) - (U2+u2))) * ψ2x + (U2+u2) * q2x - ν * Δq2
  Ff2 .+= FourierFlows.jacobian(ψ2, η, gr) + @. (U2+u2) * ηx + μ * Δψ2

  T = eltype(gr)

  Ff = zeros(dev, T, (gr.nx, gr.ny, nlayers))
  @views Ff[:, :, 1] = Ff1
  @views Ff[:, :, 2] = Ff2

  Ffh = zeros(dev, Complex{T}, (gr.nkr, gr.nl, nlayers))
  @views Ffh[:, :, 1] = rfft(Ff1)
  @views Ffh[:, :, 2] = rfft(Ff2)

  function calcFq!(Fqh, sol, t, cl, v, p, g)
    Fqh .= Ffh
    return nothing
  end

  prob = MultiLayerQG.Problem(nlayers, dev; nx, ny, Lx, Ly, f₀, H, b, U,
                              eta=η, β, μ, ν, nν, calcFq=calcFq!, stepper, dt)

  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  qf = zeros(dev, T, (gr.nx, gr.ny, nlayers))
  @views qf[:, :, 1] = q1
  @views qf[:, :, 2] = q2

  ψf = zeros(dev, T, (gr.nx, gr.ny, nlayers))
  @views ψf[:, :, 1] = ψ1
  @views ψf[:, :, 2] = ψ2

  MultiLayerQG.set_q!(prob, qf)

  stepforward!(prob, nt)
  MultiLayerQG.updatevars!(prob)

  return isapprox(vs.q, qf, rtol=rtol_multilayerqg) &&
         isapprox(vs.ψ, ψf, rtol=rtol_multilayerqg)
end

"""
    test_mqg_nonlinearadvection_3layers(dt, stepper, dev::Device=CPU()

Same as `test_mqg_nonlinearadvection_2layers` but for 3 layers.
"""
function test_mqg_nonlinearadvection_3layers(dt, stepper, dev::Device=CPU())

  A = device_array(dev)

  tf = 200*dt
  nt = round(Int, tf/dt)

  nx, ny = 64, 66
  Lx, Ly = 2π, 5π
  gr = TwoDGrid(dev; nx, Lx, ny, Ly)

  x, y = gridpoints(gr)
  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly # fundamental wavenumbers

  nlayers = 3                # these choice of parameters give the
  f₀ = 1                     # desired PV-streamfunction relations
  H = [0.25, 0.25, 0.5]      # q1 = Δψ1 + 20ψ2 - 20ψ1,
  b = [-1.0, -1.2, -1.2-1/6] # q2 = Δψ2 + 20ψ1 - 44ψ2 + 24ψ3,
                             # q3 = Δψ3 + 12ψ2 - 12ψ3.

  β = 0.35

  U1, U2, U3 = 0.02, 0.025, 0.01
  u1 = @. 0.05sech(gr.y / (Ly/15))^2; u1 = A(reshape(u1, (1, gr.ny)))
  u2 = @. 0.02cos(3l₀ * gr.y);        u2 = A(reshape(u2, (1, gr.ny)))
  u3 = @. 0.01cos(3l₀ * gr.y);        u3 = A(reshape(u3, (1, gr.ny)))

  uyy1 = real.(ifft( -gr.l.^2 .* fft(u1) ))
  uyy2 = real.(ifft( -gr.l.^2 .* fft(u2) ))
  uyy3 = real.(ifft( -gr.l.^2 .* fft(u3) ))

  U = zeros(ny, nlayers)
  CUDA.@allowscalar U[:, 1] = u1 .+ U1
  CUDA.@allowscalar U[:, 2] = u2 .+ U2
  CUDA.@allowscalar U[:, 3] = u3 .+ U3

  μ, ν, nν = 0.1, 0.05, 1

  η0, σx, σy = 1.0, Lx/25, Ly/20
   η = @. η0 * exp( -(x + Lx/8)^2 / 2σx^2 - (y - Ly/8)^2 / 2σy^2)
  ηx = @. - (x + Lx/8) / σx^2 * η

  ψ1, ψ2, ψ3, q1, q2, q3, ψ1x, ψ2x, ψ3x, q1x, q2x, q3x, Δq1, Δq2, Δq3, Δψ3 = constructtestfields_3layer(gr)

  Ff1 = FourierFlows.jacobian(ψ1, q1, gr) + @. (β - uyy1                            - 20 * ((U2+u2) - (U1+u1))) * ψ1x + (U1+u1) * q1x - ν * Δq1
  Ff2 = FourierFlows.jacobian(ψ2, q2, gr) + @. (β - uyy2 - 20 * ((U1+u1) - (U2+u2)) - 24 * ((U3+u3) - (U2+u2))) * ψ2x + (U2+u2) * q2x - ν * Δq2
  Ff3 = FourierFlows.jacobian(ψ3, q3, gr) + @. (β - uyy3 - 12 * ((U2+u2) - (U3+u3))                           ) * ψ3x + (U3+u3) * q3x - ν * Δq3
  Ff3 .+= FourierFlows.jacobian(ψ3, η, gr) + @. (U3+u3) * ηx + μ * Δψ3

  T = eltype(gr)

  Ff = zeros(dev, T, (gr.nx, gr.ny, nlayers))
  @views Ff[:, :, 1] = Ff1
  @views Ff[:, :, 2] = Ff2
  @views Ff[:, :, 3] = Ff3

  Ffh = zeros(dev, Complex{T}, (gr.nkr, gr.nl, nlayers))
  @views Ffh[:, :, 1] = rfft(Ff1)
  @views Ffh[:, :, 2] = rfft(Ff2)
  @views Ffh[:, :, 3] = rfft(Ff3)

  function calcFq!(Fqh, sol, t, cl, v, p, g)
    Fqh .= Ffh
    return nothing
  end

  prob = MultiLayerQG.Problem(nlayers, dev; nx, ny, Lx, Ly, f₀, H, b, U,
                              eta=η, β, μ, ν, nν, calcFq=calcFq!, stepper, dt)

  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  qf = zeros(dev, T, (gr.nx, gr.ny, nlayers))
  @views qf[:, :, 1] = q1
  @views qf[:, :, 2] = q2
  @views qf[:, :, 3] = q3

  ψf = zeros(dev, T, (gr.nx, gr.ny, nlayers))
  @views ψf[:, :, 1] = ψ1
  @views ψf[:, :, 2] = ψ2
  @views ψf[:, :, 3] = ψ3

  MultiLayerQG.set_q!(prob, qf)

  stepforward!(prob, nt)
  MultiLayerQG.updatevars!(prob)

  return isapprox(vs.q, qf, rtol=rtol_multilayerqg) &&
         isapprox(vs.ψ, ψf, rtol=rtol_multilayerqg)
end

"""
    test_mqg_linearadvection(dt, stepper, dev; kwargs...)

Tests the advection term of the linearized equations by timestepping a test
problem with timestep dt and timestepper identified by the string stepper.
The test 2-layer problem is derived by picking a solution q1f, q2f (with
streamfunctions ψ1f, ψ2f) for which the advection terms J(ψn, qn) are non-zero.
Next, a forcing Ff is derived such that a solution to the problem forced by this
Ff is then qf. (This solution may not be realized, at least at long times, if it
is unstable.)
"""
function test_mqg_linearadvection(dt, stepper, dev::Device=CPU();
                                  n=128, L=2π, nlayers=2, μ=0.0, ν=0.0, nν=1)

  A = device_array(dev)

  tf = 0.5
  nt = round(Int, tf/dt)

  nx, ny = 64, 66
  Lx, Ly = 2π, 2π
  gr = TwoDGrid(dev; nx, Lx, ny, Ly)

  x, y = gridpoints(gr)
  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly # fundamental wavenumbers

  nlayers = 2       # these choice of parameters give the
  f₀ = 1            # desired PV-streamfunction relations
  H = [0.2, 0.8]    # q1 = Δψ1 + 25*(ψ2-ψ1), and
  b = [-1.0, -1.2]  # q2 = Δψ2 + 25/4*(ψ1-ψ2).

  β = 0.35

  U1, U2 = 0.1, 0.05

  u1 = @. 0.5sech(gr.y / (Ly/15))^2; u1 = A(reshape(u1, (1, gr.ny)))
  u2 = @. 0.02cos(3l₀ * gr.y);       u2 = A(reshape(u2, (1, gr.ny)))

  uyy1 = real.(ifft( -gr.l.^2 .* fft(u1) ))
  uyy2 = real.(ifft( -gr.l.^2 .* fft(u2) ))

  U = zeros(ny, nlayers)
  CUDA.@allowscalar U[:, 1] = u1 .+ U1
  CUDA.@allowscalar U[:, 2] = u2 .+ U2

  μ, ν, nν = 0.1, 0.05, 1

  η0, σx, σy = 1.0, Lx/25, Ly/20
  η = @. η0 * exp( - (x + Lx/8)^2 / 2σx^2 - (y - Ly/8)^2 / 2σy^2)
  ηx = @. -(x + Lx/8) / σx^2 * η

  ψ1, ψ2, q1, q2, ψ1x, ψ2x, q1x, q2x, Δψ2, Δq1, Δq2 = constructtestfields_2layer(gr)

  Ff1 = @. (β - uyy1 -   25 * ((U2+u2) - (U1+u1)) ) * ψ1x + (U1+u1) * q1x - ν * Δq1
  Ff2 = @. (β - uyy2 - 25/4 * ((U1+u1) - (U2+u2)) ) * ψ2x + (U2+u2) * q2x - ν * Δq2
  Ff2 .+= FourierFlows.jacobian(ψ2, η, gr) + @. (U2+u2) * ηx + μ * Δψ2

  T = eltype(gr)

  Ff = zeros(dev, T, (gr.nx, gr.ny, nlayers))
  @views Ff[:, :, 1] = Ff1
  @views Ff[:, :, 2] = Ff2

  Ffh = zeros(dev, Complex{T}, (gr.nkr, gr.nl, nlayers))
  @views Ffh[:, :, 1] = rfft(Ff1)
  @views Ffh[:, :, 2] = rfft(Ff2)

  function calcFq!(Fqh, sol, t, cl, v, p, g)
    Fqh .= Ffh
    return nothing
  end

  prob = MultiLayerQG.Problem(nlayers, dev; nx, ny, Lx, Ly, f₀, H, b, U,
                              eta=η, β, μ, ν, nν, calcFq=calcFq!, stepper, dt, linear=true)

  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  qf = zeros(dev, T, (gr.nx, gr.ny, nlayers))
  @views qf[:, :, 1] = q1
  @views qf[:, :, 2] = q2

  ψf = zeros(dev, T, (gr.nx, gr.ny, nlayers))
  @views ψf[:, :, 1] = ψ1
  @views ψf[:, :, 2] = ψ2

  MultiLayerQG.set_q!(prob, qf)

  stepforward!(prob, nt)
  MultiLayerQG.updatevars!(prob)

  return isapprox(vs.q, qf, rtol=rtol_multilayerqg) &&
         isapprox(vs.ψ, ψf, rtol=rtol_multilayerqg)
end

"""
    test_mqg_energies(dev; kwargs...)

Tests the kinetic (KE) and potential (PE) energies function by constructing a
2-layer problem and initializing it with a flow field whose KE and PE are known.
"""
function test_mqg_energies(dev::Device=CPU();
                           dt=0.001, stepper="ForwardEuler", n=128, L=2π, nlayers=2, μ=0.0, ν=0.0, nν=1)
  nx, ny = 64, 66
  Lx, Ly = 2π, 2π
  gr = TwoDGrid(dev; nx, Lx, ny, Ly)

  x, y = gridpoints(gr)
  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly # fundamental wavenumbers
  nlayers = 2

  f₀ = 1
  H = [2.5, 7.5]   # sum(params.H) = 10
  b = [-1.0, -1.5] # Make g′ = 1/2

  prob = MultiLayerQG.Problem(nlayers, dev; nx, ny, Lx, Ly, f₀, H, b)
  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  ψ = zeros(dev, eltype(gr), (gr.nx, gr.ny, nlayers))

  CUDA.@allowscalar @views @. ψ[:, :, 1] =       cos(2k₀ * x) * cos(2l₀ * y)
  CUDA.@allowscalar @views @. ψ[:, :, 2] = 1/2 * cos(2k₀ * x) * cos(2l₀ * y)

  MultiLayerQG.set_ψ!(prob, ψ)

  KE1_calc = 1/4      # = H1/H
  KE2_calc = 3/4/4    # = H2/H/4
  PE_calc  = 1/16/10  # = 1/16/H

  KE, PE = MultiLayerQG.energies(prob)

  return isapprox(KE[1], KE1_calc, rtol=rtol_multilayerqg) &&
         isapprox(KE[2], KE2_calc, rtol=rtol_multilayerqg) &&
         isapprox(PE[1], PE_calc, rtol=rtol_multilayerqg) &&
         MultiLayerQG.addforcing!(prob.timestepper.RHS₁, sol, cl.t, cl, vs, pr, gr)==nothing
end

function test_mqg_energysinglelayer(dev::Device=CPU();
                                    dt=0.001, stepper="ForwardEuler", nlayers=1, μ=0.0, ν=0.0, nν=1)
  nx, Lx  = 64, 2π
  ny, Ly  = 64, 3π
  gr = TwoDGrid(dev; nx, Lx, ny, Ly)

  x, y = gridpoints(gr)
  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly # fundamental wavenumbers

  energy_calc = 29/9

  ψ0 = @. sin(2k₀*x)*cos(2l₀*y) + 2sin(k₀*x)*cos(3l₀*y)
  q0 = @. -((2k₀)^2+(2l₀)^2)*sin(2k₀*x)*cos(2l₀*y) - (k₀^2+(3l₀)^2)*2sin(k₀*x)*cos(3l₀*y)

  prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx, ny, Ly, stepper, U=zeros(ny))

  MultiLayerQG.set_q!(prob, reshape(q0, (nx, ny, nlayers)))

  energyq0 = MultiLayerQG.energies(prob)

  return isapprox(energyq0, energy_calc, rtol=rtol_multilayerqg)
end

"""
    test_mqg_fluxes(dt, stepper; kwargs...)

Tests the lateral and vertical eddy fluxes by constructing a 2-layer problem and
initializing it with a flow field whose fluxes are known.
"""
function test_mqg_fluxes(dev::Device=CPU(); nlayers, dt=0.001, stepper="ForwardEuler", n=128, L=2π, μ=0.0, ν=0.0, nν=1)

  if nlayers < 2 || nlayers > 3; error("only works for nlayers = 2 or 3"); end

  nx, ny = 128, 126
  Lx, Ly = 2π, 2π
  gr = TwoDGrid(dev; nx, Lx, ny, Ly)

  x, y = gridpoints(gr)
  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly # fundamental wavenumbers

  f₀ = 1

  if nlayers == 2
    H = [0.2, 0.8]    # q1 = Δψ1 + 25*(ψ2-ψ1), and
    b = [-1.0, -1.2]  # q2 = Δψ2 + 25/4*(ψ1-ψ2).
  elseif nlayers == 3
    # we make a 3-layer problem for which layers 2 and 3 act as one
    # this way we can check that fluxes for nlayers>2 are still correct
    H = [0.2, 0.35, 0.45]
    b = [-1.0, -1.2, -1.2+1e-9]
  end

  U = zeros(ny, nlayers)
  U[:, 1] = @. sech(gr.y / 0.2)^2

  prob = MultiLayerQG.Problem(nlayers, dev; nx, ny, Lx, Ly, f₀, H, b, U)

  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  x, y = gridpoints(gr)
  ψ1 = @. cos(k₀*x) * cos(l₀*y) + sin(k₀*x)
  ψ2 = @. cos(k₀*x + π/10) * cos(l₀*y)

  ψ = zeros(dev, eltype(gr), (gr.nx, gr.ny, nlayers))

  view(ψ, :, :, 1) .= ψ1
  view(ψ, :, :, 2) .= ψ2

  if nlayers == 3
    view(ψ, :, :, 3) .= ψ2
  end

  MultiLayerQG.set_ψ!(prob, ψ)
  lateralfluxes, verticalfluxes = MultiLayerQG.fluxes(prob)

  testfluxes = CUDA.@allowscalar isapprox(lateralfluxes[1], 0.00626267, rtol=1e-6) &&
               CUDA.@allowscalar isapprox(lateralfluxes[2], 0, atol=1e-12) &&
               CUDA.@allowscalar isapprox(verticalfluxes[1], -0.196539, rtol=1e-6)

  if nlayers == 3
    testfluxes = testfluxes &&
                 CUDA.@allowscalar isapprox(lateralfluxes[3], 0, atol=1e-12) &&
                 CUDA.@allowscalar isapprox(verticalfluxes[2], 0, atol=1e-12)
  end

  return testfluxes
end

"""
    test_mqg_fluxessinglelayer(dt, stepper; kwargs...)

Tests the lateral eddy fluxes by constructing a 1-layer problem and initializing
it with a flow field whose fluxes are known.
"""
function test_mqg_fluxessinglelayer(dev::Device=CPU(); dt=0.001, stepper="ForwardEuler", n=128, L=2π, μ=0.0, ν=0.0, nν=1)
  nlayers = 1

  nx, ny = 128, 126
  Lx, Ly = 2π, 2π
  gr = TwoDGrid(dev; nx, Lx, ny, Ly)

  x, y = gridpoints(gr)
  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly # fundamental wavenumbers

  U = zeros(ny, nlayers)
  U = @. sech(gr.y / 0.2)^2

  prob = MultiLayerQG.Problem(nlayers, dev; nx, ny, Lx, Ly, U)

  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  ψ = @. cos(k₀*x) * cos(l₀*y) + sin(k₀*x)

  MultiLayerQG.set_ψ!(prob, ψ)
  lateralfluxes = MultiLayerQG.fluxes(prob)

  return CUDA.@allowscalar isapprox(lateralfluxes[1], 0.0313134, atol=1e-7)
end

"""
    test_setqsetψ(dt, stepper; kwargs...)

Tests the set_q!() and set_ψ!() functions that initialize sol with a flow with
given `q` or `ψ` respectively.
"""
function test_mqg_setqsetψ(dev::Device=CPU(); dt=0.001, stepper="ForwardEuler", n=64, L=2π, nlayers=2, μ=0.0, ν=0.0, nν=1)
  nx, ny = 32, 34
  L = 2π
  gr = TwoDGrid(dev; nx, Lx=L, ny, Ly=L)

  x, y = gridpoints(gr)
  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly # fundamental wavenumbers

  nlayers = 2       # these choice of parameters give the
  f₀ = 1            # desired PV-streamfunction relations
  H = [0.2, 0.8]    # q1 = Δψ1 + 25*(ψ2-ψ1), and
  b = [-1.0, -1.2]  # q2 = Δψ2 + 25/4*(ψ1-ψ2).

  prob = MultiLayerQG.Problem(nlayers, dev; nx, ny, Lx=L, f₀, H, b)

  sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

  T = eltype(gr)

  f1 = @. 2cos(k₀*x) * cos(l₀*y)
  f2 = @.  cos(k₀*x + π/10) * cos(2l₀*y)

  f = zeros(dev, T, (gr.nx, gr.ny, nlayers))
  view(f, :, :, 1) .= f1
  view(f, :, :, 2) .= f2

  ψtest = zeros(dev, T, size(f))
  MultiLayerQG.set_ψ!(prob, f)
  @. vs.qh = sol
  MultiLayerQG.streamfunctionfrompv!(vs.ψh, vs.qh, pr, gr)
  MultiLayerQG.invtransform!(ψtest, vs.ψh, pr)

  qtest = zeros(dev, T, size(f))
  MultiLayerQG.set_q!(prob, f)
  @. vs.qh = sol
  MultiLayerQG.invtransform!(qtest, vs.qh, pr)

  return isapprox(ψtest, f, rtol=rtol_multilayerqg) &&
         isapprox(qtest, f, rtol=rtol_multilayerqg)
end

"""
    test_mqg_set_topographicPV_largescale_gradient(dt, stepper; kwargs...)

Ensures that the error of setting a topographic PV large-scale gradient is small.
"""
function test_mqg_set_topographicPV_largescale_gradient(dev::Device=CPU(); dt=0.001, stepper="ForwardEuler", n=64, L=2π, nlayers=2, μ=0.0, ν=0.0, nν=1)
  nx, ny = 32, 34
  L = 2π
  gr = TwoDGrid(dev; nx, Lx=L, ny, Ly=L)
  T = eltype(gr)

  x, y = gridpoints(gr)
  k₀, l₀ = 2π/gr.Lx, 2π/gr.Ly    # fundamental wavenumbers

  nlayers = 2                    # these choice of parameters give the
  f₀ = 1                         # desired PV-streamfunction relations
  H = [0.2, 0.8]                 # q1 = Δψ1 + 25   * (ψ2 - ψ1), and
  b = [-1.0, -1.2]               # q2 = Δψ2 + 25/4 * (ψ1 - ψ2).
  U = zeros(nlayers)             # the imposed mean zonal flow in each layer
  U[1] = 1.0
  U[2] = 0.0

  # topographic amplitude
  h₀ = 0.1

  # topographic amplitude
  topographic_slope_x, topographic_slope_y = 1e-2, 1e-1

  # topography (bumps)
  h = @. h₀ * cos(k₀*x) * cos(l₀*y)

  # topographic PV
  eta = f₀ * h / H[2]

  prob = MultiLayerQG.Problem(nlayers, dev;
                              nx, ny, Lx=L, Ly=L, f₀, H, b, U, μ, β=0, T, eta,
                              topographic_pv_gradient = f₀ / H[2] .* (topographic_slope_x, topographic_slope_y),
                              dt, stepper, aliased_fraction=0)

  # Test to see if the internally-computed total bottom Qx and Qy are correct
  g′ = b[1] - b[2]
  F1 = f₀^2 / (H[2] * g′)
  Psi1y, Psi2y = -U[1], -U[2]

  Q2x_analytic = @. f₀ * (topographic_slope_x - h₀ * k₀ * sin(k₀*x) * cos(l₀*y)) / H[2]
  Q2y_analytic = @. f₀ * (topographic_slope_y - h₀ * l₀ * cos(k₀*x) * sin(l₀*y)) / H[2] + F1 * (Psi1y - Psi2y)

  return prob.params.topographic_pv_gradient == f₀ / H[2] .* (topographic_slope_x, topographic_slope_y) &&
         isapprox(prob.params.Qx[:, :, 2], Q2x_analytic, rtol=rtol_multilayerqg) &&
         isapprox(prob.params.Qy[:, :, 2], Q2y_analytic, rtol=rtol_multilayerqg)
end

"""
    test_paramsconstructor(; kwargs...)

Tests that `Params` constructor works with both mean flow `U` being a floats
(i.e., constant `U` in each layer) or vectors (i.e., `U(y)` in each layer).
"""
function test_mqg_paramsconstructor(dev::Device=CPU(); dt=0.001, stepper="ForwardEuler")
  nx, ny = 32, 34
  L = 2π
  gr = TwoDGrid(dev; nx, Lx=L, ny, Ly=L)

  nlayers = 2       # these choice of parameters give the
  f₀ = 1            # desired PV-streamfunction relations
  H = [0.2, 0.8]    # q1 = Δψ1 + 25*(ψ2-ψ1), and
  b = [-1.0, -1.2]  # q2 = Δψ2 + 25/4*(ψ1-ψ2).

  U1, U2 = 0.1, 0.05

  T = eltype(gr)

  Uvectors = zeros(dev, T, (ny, nlayers))
  Uvectors[:, 1] .= U1
  Uvectors[:, 2] .= U2

  Ufloats = zeros(dev, T, nlayers)
  CUDA.@allowscalar Ufloats[1] = U1
  CUDA.@allowscalar Ufloats[2] = U2

  probUvectors = MultiLayerQG.Problem(nlayers, dev; nx, ny, Lx=L, f₀, H, b, U=Uvectors)
  probUfloats = MultiLayerQG.Problem(nlayers, dev; nx, ny, Lx=L, f₀, H, b, U=Ufloats)

  return isapprox(probUfloats.params.U, probUvectors.params.U, rtol=rtol_multilayerqg)
end

function test_mqg_problemtype(dev, T)
  prob1 = MultiLayerQG.Problem(1, dev; T)
  prob2 = MultiLayerQG.Problem(2, dev; T)

  A = device_array(dev)

  return typeof(prob1.sol)<:A{Complex{T}, 3} &&
         typeof(prob1.grid.Lx)==T &&
         typeof(prob1.vars.u)<:A{T, 3} &&
         typeof(prob2.sol)<:A{Complex{T}, 3} &&
         typeof(prob2.grid.Lx)==T && typeof(prob2.vars.u)<:A{T, 3}
end

"""
    test_mqg_rossbywave(; kwargs...)

Evolves a Rossby wave on a beta plane with an imposed zonal flow U and compares
with the analytic solution.
"""
function test_mqg_rossbywave(stepper, dt, nsteps, dev::Device=CPU())
  nlayers = 1
       nx = 64
       Lx = 2π
        β = 2
        U = 0.5

  prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx, U, β, stepper, dt)

  sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid

  x, y = gridpoints(g)

  # the Rossby wave initial condition
   ampl = 1e-2
  kwave, lwave = 3 * 2π/g.Lx, 2 * 2π/g.Ly
      ω = kwave * ( U - p.β / (kwave^2 + lwave^2) ) # Doppler-shifted Rossby frequency
     q0 = @. ampl * cos(kwave * x) * cos(lwave * y)
     ψ0 = @. - q0 / (kwave^2 + lwave^2)

  MultiLayerQG.set_q!(prob, q0)

  stepforward!(prob, nsteps)
  dealias!(sol, g)
  MultiLayerQG.updatevars!(prob)

  q_theory = @. ampl * cos(kwave * (x - ω / kwave * cl.t)) * cos(lwave * y)

  return isapprox(q_theory, v.q, rtol=g.nx*g.ny*nsteps*1e-12)
end

function test_numberoflayers(dev::Device=CPU())
  prob_nlayers1 = MultiLayerQG.Problem(1, dev)
  prob_nlayers2 = MultiLayerQG.Problem(2, dev)

  return MultiLayerQG.numberoflayers(prob_nlayers1)==1 &&
         MultiLayerQG.numberoflayers(prob_nlayers2)==2
end

function test_mqg_stochasticforcedproblemconstructor(dev::Device=CPU())

  function calcFq!(Fqh, sol, t, clock, vars, params, grid)
    Fqh .= device_array(dev)(rand(eltype(prob.sol), size(prob.sol)))
    return nothing
  end

  prob = MultiLayerQG.Problem(2, dev; calcFq=calcFq!, stochastic=true)

  @test maximum(abs, prob.sol) == 0

  sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
  GeophysicalFlows.MultiLayerQG.addforcing!(sol, sol, 0, clock, vars, params, grid)

  @test minimum(abs, prob.sol) > 0

  return typeof(prob.vars.prevsol)==typeof(prob.sol)
end
