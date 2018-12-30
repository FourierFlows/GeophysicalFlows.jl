"""
    lambdipole(U, R, g::TwoDimGrid; center=(mean(g.x), mean(g.y))

Return the 2D vorticity field of the Lamb dipole with strength `U` and radius `R`, centered on
`center=(xc, yc)` and on the grid `g`. The default value of `center` is the middle of the grid.
"""
function lambdipole(U, R, g; center=(mean(g.x), mean(g.y)))
  firstzero = 3.8317059702075123156
  k = firstzero/R # dipole wavenumber for radius R in terms of first zero of besselj
  q0 = -2U*k/besselj(0, k*R) # dipole amplitude for strength U and radius R

  xc, yc = center
  r = @. sqrt( (g.x-xc)^2 + (g.y-yc)^2 )
  q = @. q0 * besselj(1, k*r) * (g.y-yc)/r
  @. q[r >= R] = 0
  q
end

"""
    peakedisotropicspectrum(g, kpeak, E0; mask=mask, allones=false)
Generate a real and random two-dimensional vorticity field q(x, y) with
a Fourier spectrum peaked around a central non-dimensional wavenumber kpeak and
normalized so that its total energy is E0.
"""
function peakedisotropicspectrum(g::TwoDGrid, kpeak::Real, E0::Real; mask=ones(size(g.Krsq)), allones=false)
  if g.Lx !== g.Ly
      error("the domain is not square")
  else
    k0 = kpeak*2π/g.Lx
    modk = sqrt.(g.Krsq)
    psik = zeros(g.nk, g.nl)
    psik = @. (modk^2 * (1 + (modk/k0)^4))^(-0.5)
    psik[1, 1] = 0.0
    psih = @. randn(Complex{typeof(g.Lx)}, size(g.Krsq))*psik
    if allones; psih = psik; end
    psih = psih.*mask
    Ein = real(sum(g.Krsq.*abs2.(psih)/(g.nx*g.ny)^2))
    psih = psih*sqrt(E0/Ein)
    q = -irfft(g.Krsq.*psih, g.nx)
  end
end
