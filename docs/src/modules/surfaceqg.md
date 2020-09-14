# SurfaceQG Module

### Basic Equations

This module solves the non-dimensional surface quasi-geostrophic (SQG) equation for surface buoyancy $b_s=b(x,y,z=0)$ described in Capet et al., 2008. The buoyancy and fluid motion at the surface are related through streamfunction $\psi$ as $(u_s, \upsilon_s, b_s) = (-\partial_y\psi, \partial_x\psi, -\partial_z\psi)$. This model solves the time evolution of the surface buoyancy,

$$\partial_t b_s + \mathsf{J}(\psi, b_s) = \underbrace{-\nu(-1)^{n_\nu} \nabla^{2n_\nu} b_s}_{\textrm{buoyancy diffusion}} + \underbrace{f}_{\textrm{forcing}}\.$$

The evolution of buoyancy is only solved for the surface layer, but $b_s$ is a function of the vertical gradient of $\psi$. In the SQG system, the potential vorticity in the interior of the flow is identically zero. That is, relative vorticity is identical and opposite to the vertical stretching of buoyancy layers,

$$\underbrace{\left(\partial_x^2 + \partial_y^2 \right) \psi}_{\textrm{relative vorticity}} + \underbrace{\partial_z^2 \psi}_{\textrm{stretching term}} = 0  $$

with the boundary conditions $b_s = -\partial_z\psi|_{z=0}$ and $\psi\rightarrow 0$ as $z \rightarrow -\infty$.

These equations describe a system where the streamfunction (and hence the dynamics) at all depths is prescribed entirely by the surface buoyancy. In horizontal spectral space $(k,l,z)$ this relation is.

$$\widehat{\psi}(k,l,z)t = -\frac{\widehat{\b_s}}{K}e^{Kz} $$

where $K=\sqrt(k^2+l^2)$ and $\widehat{\cdot}$ denotes a horizontal Fourier transform.

### Implementation

The equation is time-stepped forward in Fourier space:

$$\partial_t \widehat{b_s} = \underbrace{- \widehat{\mathsf{J}(\psi, b_s)}}_{\textrm{Non-linear term} \mathcal{N}(\widehat{b_s})} - \underbrace{\nu k^{2n_\nu}  \widehat{\b_s}}_{\textrm{Linear term} \mathcal{L}}  + \widehat{f}\ .$$

In doing so the Jacobian is computed in the conservative form: $\mathsf{J}(f,g) =
\partial_y [ (\partial_x f) g] -\partial_x[ (\partial_y f) g]$.

Thus:
$$ \widehat{u} = \frac{il}{K}\widehat{b_s}, \qquad \widehat{v} = -\frac{ik}{K}\widehat{b_s} ,$$
$$\mathcal{L} = \beta\frac{\mathrm{i}k_x}{k^2} - \mu - \nu k^{2n_\nu}\ ,$$
$$\mathcal{N}(\widehat{b_s}) = - \mathrm{i}k_x \mathrm{FFT}(u q)-
	\mathrm{i}k_y \mathrm{FFT}(\upsilon q)\ .$$


## Examples

- `examples/surfaceqg_decaying.jl`: A script that simulates decaying surface quasi-geostrophic flow with a prescribed initial buoyancy field, producing a video of the evolution of buoyancy and velocity fields.

- `examples/surfaceqg_decaying_budget.jl`: A script that simulates decaying surface quasi-geostrophic flow with a prescribed initial buoyancy field, producing plots of buoyancy variance and kinetic energy budget terms.


  > Capet, X. et al., (2008). Surface kinetic energy transfer in surface quasi-geostrophic flows. *J. Fluid Mech.*, **604**, 165-174.
