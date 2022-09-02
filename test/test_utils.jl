"""
Test the peakedisotropicspectrum function.
"""
function testpeakedisotropicspectrum(dev::Device=CPU())
  n, L = 128, 2π
  grid = TwoDGrid(dev; nx=n, Lx=L)
  k0, E0 = 6, 0.5
  qi = peakedisotropicspectrum(grid, k0, E0; allones=true)
  ρ, qhρ = FourierFlows.radialspectrum(rfft(qi) .* grid.invKrsq, grid)

  ρtest = ρ[ (ρ.>15.0) .& (ρ.<=17.5)]
  qhρtest = qhρ[ (ρ.>15.0) .& (ρ.<=17.5)]

  return CUDA.@allowscalar isapprox(abs.(qhρtest)/abs(qhρtest[1]), (ρtest/ρtest[1]).^(-2), rtol=5e-3)
end

function testpeakedisotropicspectrum_rectangledomain(dev::Device=CPU())
  nx, ny = 32, 34
  Lx, Ly = 2π, 3π
  grid = TwoDGrid(dev; nx, Lx, ny, Ly)
  k0, E0 = 6, 0.5
  qi = peakedisotropicspectrum(grid, k0, E0; allones=true)
end
