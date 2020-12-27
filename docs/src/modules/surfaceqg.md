# SurfaceQG Module

### Basic Equations

This module solves the non-dimensional surface quasi-geostrophic (SQG) equation for surface 
buoyancy ``b_s = b(x, y, z=0)``, as described in Capet et al., 2008. The buoyancy and the fluid 
velocity at the surface are related through a streamfunction ``\psi`` via:

```math
(u_s, v_s, b_s) = (-\partial_y \psi, \partial_x \psi, -\partial_z \psi) .
```

The SQG model evolves the surface buoyancy,

```math
\partial_t b_s + \mathsf{J}(\psi, b_s) = \underbrace{-\nu(-1)^{n_\nu} \nabla^{2n_\nu} b_s}_{\textrm{buoyancy diffusion}} + \underbrace{F}_{\textrm{forcing}} .
```

The evolution of buoyancy is only solved for the surface layer, but ``b_s`` is a function of the vertical gradient of ``\psi``. In the SQG system, the potential vorticity in the interior of the flow is identically zero. That is, relative vorticity is identical and opposite to the vertical stretching of buoyancy layers,

```math
\underbrace{\left(\partial_x^2 + \partial_y^2 \right) \psi}_{\textrm{relative vorticity}} + \underbrace{\partial_z^2 \psi}_{\textrm{stretching term}} = 0 ,
```

with the boundary conditions ``b_s = -\partial_z\psi|_{z=0}`` and ``\psi \rightarrow 0`` as ``z \rightarrow -\infty``. (We take here the oceanographic convention: ``z \le 0``.)

These equations describe a system where the streamfunction (and hence the dynamics) at all depths is prescribed entirely by the surface buoyancy. By taking the Fourier transform in the horizontal (``x`` and ``y``), the streamfunction-buoyancy relation is:

```math
\widehat{\psi}(k_x, k_y, z, t) = - \frac{\widehat{b_s}}{|𝐤|} \, e^{|𝐤|z}, 
```

where ``|𝐤| = \sqrt{k_x^2 + k_y^2}`` is the total horizontal wavenumber.

### Implementation

The buoyancy equation is time-stepped forward in Fourier space:

```math
\partial_t \widehat{b_s} = - \widehat{\mathsf{J}(\psi, b_s)} - \nu |𝐤|^{2 n_\nu} \widehat{b_s} + \widehat{f} .
```

In doing so the Jacobian is computed in the conservative form: ``\mathsf{J}(f,g) =
\partial_y [ (\partial_x f) g] -\partial_x[ (\partial_y f) g]``.

Thus:
```math
\begin{aligned}
\widehat{u} &= \frac{i k_y}{|𝐤|} \widehat{b_s}, \qquad \widehat{v} = -\frac{i k_x}{|𝐤|} \widehat{b_s}, \\
L & = - \nu |𝐤|^{2n_\nu},\\
N(\widehat{b_s}) & = - i k_x \mathrm{FFT}(u b) - i k_y \mathrm{FFT}(v b) .
\end{aligned}
```


## Examples

- `examples/surfaceqg_decaying.jl`: A script that simulates decaying surface quasi-geostrophic flow with a prescribed initial buoyancy field, producing a video of the evolution of buoyancy and velocity fields.

  > Capet, X. et al., (2008). Surface kinetic energy transfer in surface quasi-geostrophic flows. *J. Fluid Mech.*, **604**, 165-174.
