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
