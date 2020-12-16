# SingleLayerQG Module

### Basic Equations

This module solves the barotropic or equivalent barotropic quasi-geostrophic vorticity equation 
on a beta-plane of variable fluid depth ``H - h(x, y)``. The flow is obtained through a streamfunction ``\psi`` as ``(u, \upsilon) = (-\partial_y \psi, \partial_x \psi)``. All flow 
fields can be obtained from the quasi-geostrophic potential vorticity (QGPV). Here the QGPV is

```math
	\underbrace{f_0 + \beta y}_{\text{planetary PV}} + \underbrace{\partial_x \upsilon
	- \partial_y u}_{\text{relative vorticity}} - \!\!
	\underbrace{\frac{1}{\ell^2} \psi}_{\text{vortex stretching}} \!\! + 
	\underbrace{\frac{f_0 h}{H}}_{\text{topographic PV}} \ ,
```

where ``\ell`` is the Rossby radius of deformation. Purely barotropic dynamics corresponds to 
infinite Rossby radius of deformation (``\ell = \infty``), while a flow with a finite Rossby 
radius follows is said to obey equivalent-barotropic dynamics.

The dynamical variable is the component of the vorticity of the flow normal to the plane of 
motion, ``\zeta \equiv \partial_x \upsilon- \partial_y u = \nabla^2\psi``. Also, we denote the topographic PV with ``\eta \equiv f_0 h / H``. Thus, the equation solved by the module is:

```math
\partial_t \zeta + \mathsf{J}(\psi, \underbrace{\zeta + \eta}_{\equiv q}) +
\beta\partial_x\psi = \underbrace{-\left[\mu + \nu(-1)^{n_\nu} \nabla^{2n_\nu}
\right] \zeta }_{\textrm{dissipation}} + f \ .
```

where ``\mathsf{J}(a, b) = (\partial_x a)(\partial_y b)-(\partial_y a)(\partial_x b)``. On 
the right hand side, ``f(x, y, t)`` is forcing, ``\mu`` is linear drag, and ``\nu`` is hyperviscosity of order ``n_\nu``. Plain old viscosity corresponds to ``n_\nu = 1``. The sum of relative vorticity and topographic PV is denoted with ``q \equiv \zeta + \eta``.

### Implementation

The equation is time-stepped forward in Fourier space:

```math
\partial_t \widehat{\zeta} = - \widehat{\mathsf{J}(\psi, q)} + \beta \frac{\mathrm{i} k_x}{k^2} \widehat{\zeta} - \left(\mu + \nu k^{2n_\nu} \right) \widehat{\zeta} + \widehat{f} \ .
```

In doing so the Jacobian is computed in the conservative form: ``\mathsf{J}(f,g) =
\partial_y [ (\partial_x f) g] -\partial_x[ (\partial_y f) g]``.

Thus:

```math
\begin{aligned}
\mathcal{L} & = \beta \frac{\mathrm{i} k_x}{k^2} - \mu - \nu k^{2n_\nu} \ , \\
\mathcal{N}(\widehat{\zeta}) & = - \mathrm{i} k_x \mathrm{FFT}(u q) - \mathrm{i} k_y \mathrm{FFT}(\upsilon q) \ .
\end{aligned}
```


## Examples

- `examples/singlelayerqg_betadecay.jl`: A script that simulates decaying quasi-geostrophic flow on a beta-plane demonstrating zonation.

- `examples/singlelayerqg_betaforced.jl`: A script that simulates forced-dissipative quasi-geostrophic flow on a beta-plane demonstrating zonation. The forcing is temporally delta-correlated and its spatial structure is isotropic with power in a narrow annulus in wavenumber space of total radius ``k_f`` in wavenumber space.

- `examples/singlelayerqg_decay_topography.jl`: A script that simulates two dimensional turbulence (barotropic quasi-geostrophic flow with ``beta=0``) above topography.
