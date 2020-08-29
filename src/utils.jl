"""
    lambdipole(U, R, grid::TwoDGrid; center=(mean(g.x), mean(g.y))

Return the 2D vorticity field of the Lamb dipole with strength `U` and radius `R`, centered on
`center=(xc, yc)` and on the `grid`. The default value of `center` is the middle of the grid.
"""
function lambdipole(U, R, grid::TwoDGrid{T, A}; center=(mean(grid.x), mean(grids.y))) where {T, A}
  firstzero = 3.8317059702075123156
  k = firstzero / R # dipole wavenumber for radius R in terms of first zero of besselj
  q0 = -2U * k / besselj(0, k * R) # dipole amplitude for strength U and radius R
  x, y = gridpoints(grid)
  xc, yc = center
  r = @. sqrt( (x - xc)^2 + (y - yc)^2 )
  CUDA.@allowscalar besselj1 = A([besselj(1, k * r[i, j]) for i=1:grid.nx, j=1:grid.ny])
  q = @. q0 * besselj1 * (y - yc) / r
  CUDA.@allowscalar @. q[r >= R] = 0
  return q
end

"""
    peakedisotropicspectrum(g, kpeak, E0; mask=mask, allones=false)
Generate a real and random two-dimensional vorticity field q(x, y) with
a Fourier spectrum peaked around a central non-dimensional wavenumber kpeak and
normalized so that its total energy is E0.
"""
function peakedisotropicspectrum(g::TwoDGrid{T, A}, kpeak::Real, E0::Real; mask=ones(size(g.Krsq)), allones=false) where {T, A}
  if g.Lx !== g.Ly
    error("the domain is not square")
  else
    k0 = kpeak * 2π / g.Lx
    modk = sqrt.(g.Krsq)
    psik = A(zeros(T, (g.nk, g.nl)))
    psik = @. (modk^2 * (1 + (modk / k0)^4))^(-0.5)
    CUDA.@allowscalar psik[1, 1] = 0.0
    phases = randn(Complex{T}, size(g.Krsq))
    phases_real, phases_imag = real.(phases), imag.(phases)
    phases = A(phases_real) + im * A(phases_imag)
    psih = @. phases * psik
    if allones; psih = psik; end
    psih = psih .* A(mask)
    Ein = real(sum(g.Krsq .* abs2.(psih) / (g.nx * g.ny)^2))
    psih = psih * sqrt(E0 / Ein)
    CUDA.@allowscalar q = A(-irfft(g.Krsq .* psih, g.nx))
  end
  return q
end
