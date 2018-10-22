test_fftwavenums() = FourierFlows.fftwavenums(6; L=2π) == [0, 1, 2, 3, -2, -1]

"""
Test the peakedisotropicspectrum function.
"""
function testpeakedisotropicspectrum()
  n, L = 128, 2π
  gr = TwoDGrid(n, L)
  k0, E0 = 6, 0.5
  qi = FourierFlows.peakedisotropicspectrum(gr, k0, E0; allones=true)
  ρ, qhρ = FourierFlows.radialspectrum(rfft(qi).*gr.invKKrsq, gr)

  ρtest = ρ[ (ρ.>15.0) .& (ρ.<=17.5)]
  qhρtest = qhρ[ (ρ.>15.0) .& (ρ.<=17.5)]

  isapprox(abs.(qhρtest)/abs(qhρtest[1]), (ρtest/ρtest[1]).^(-2), rtol=5e-3)
end

function test_rms(n)
  g = TwoDGrid(n, 2π)
  q = cos.(g.X)
  isapprox(FourierFlows.rms(q), sqrt(1/2))
end


@test test_fftwavenums()
@test testpeakedisotropicspectrum()
@test test_rms(32)
