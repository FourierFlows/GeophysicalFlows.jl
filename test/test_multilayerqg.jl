"""
    test_pvtofromstreamfunction()

Tests the pvfromstreamfunction function that gives qh from psih. To do so, it creates a 2-layer problem with parameters such that
    q1 = Δψ1 + 25*(ψ2-ψ1),
    q2 = Δψ2 + 25/4*(ψ2-ψ2).
Then given a ψ1 and ψ2 the test checks if pvfromstreamfunction gives the expected q1 and q2. Similarly, given q1, q2 checks that streamfunctionfrompv gives ψ1, ψ2.
"""
function test_pvtofromstreamfunction()
nlayers = 2
n, L = 128, 2π
gr = TwoDGrid(n, L)

beta = 0.0
  f0 = 1
   g = 1
   U = zeros(nlayers)
   u = zeros(gr.ny, nlayers)
   H = [0.2, 0.8]
 rho = [5.0, 6.0]

  mu, nu, nnu = 0.0, 0.0, 1

  x, y = gridpoints(gr)
  k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers

  η = @. 0*x

  pr = MultilayerQG.Params(nlayers, g, f0, beta, rho, H, U, u, η, mu, nu, nnu, gr)
  eq = MultilayerQG.Equation(pr, gr)
  vs = MultilayerQG.Vars(gr, pr)

  # a set of streafunctions ψ1 and ψ2 and their corresponding q1, q2
  a11, a12, a21, a22 = 1/4, 1/3, 1, 1/2
  m11, m12, m21, m22 = 2, 3, 3, 4
  n11, n12, n21, n22 = 5, 3, 4, 2

  psi1 = @. a11*cos(m11*k0*x)*cos(n11*l0*y) + a12*cos(m12*k0*x)*cos(n12*l0*y)
  psi2 = @. a21*cos(m21*k0*x)*cos(n21*l0*y) + a22*cos(m22*k0*x)*cos(n22*l0*y)

  q1 = @. 1/6 *(   75cos(4k0*x)*cos(2l0*y) +   2cos(3k0*x)*( -43cos(3l0*y) + 75cos(4*l0*y) ) - 81cos(2k0*x)*cos(5l0*y) )
  q2 = @. 1/48*( -630cos(4k0*x)*cos(2l0*y) + 100cos(3k0*x)*(    cos(3l0*y) - 15cos(4*l0*y) ) + 75cos(2k0*x)*cos(5l0*y) )

  vs.psih[:, :, 1] .= rfft(psi1)
  vs.psih[:, :, 2] .= rfft(psi2)

  MultilayerQG.pvfromstreamfunction!(vs.qh, vs.psih, pr.S, gr)
  MultilayerQG.invtransform!(vs.q, vs.qh, pr)

  vs.qh[:, :, 1] .= rfft(q1)
  vs.qh[:, :, 2] .= rfft(q2)

  MultilayerQG.streamfunctionfrompv!(vs.psih, vs.qh, pr.invS, gr)
  MultilayerQG.invtransform!(vs.psi, vs.psih, pr)

  isapprox(q1, vs.q[:, :, 1], rtol=rtol_multilayerqg) && isapprox(q2, vs.q[:, :, 2], rtol=rtol_multilayerqg) && isapprox(psi1, vs.psi[:, :, 1], rtol=rtol_multilayerqg) && isapprox(psi2, vs.psi[:, :, 2], rtol=rtol_multilayerqg)

end
