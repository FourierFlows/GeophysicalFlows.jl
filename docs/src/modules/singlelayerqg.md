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
radius follows is said to obey equivalent-barotropic dynamics. We denote the sum of the relative
vorticity and the vortex stretching contributions to the QGPV with ``q = \nabla^2 \psi - \psi / \ell^2``.
Also, we denote the topographic PV with ``\eta \equiv f_0 h / H``.

The dynamical variable is ``q``.  Thus, the equation solved by the module is:

```math
\partial_t q + \mathsf{J}(\psi, q + \eta) +
\beta \partial_x \psi = \underbrace{-\left[\mu + \nu(-1)^{n_\nu} \nabla^{2n_\nu}
\right] q}_{\textrm{dissipation}} + f \ .
```

where ``\mathsf{J}(a, b) = (\partial_x a)(\partial_y b)-(\partial_y a)(\partial_x b)``. On 
the right hand side, ``f(x, y, t)`` is forcing, ``\mu`` is linear drag, and ``\nu`` is hyperviscosity of order ``n_\nu``. Plain old viscosity corresponds to ``n_\nu = 1``.

The kinetic energy of the fluid is computed via:

```@docs
GeophysicalFlows.SingleLayerQG.kinetic_energy
```

while the potential energy, for an equivalent barotropic fluid, is computed via:

```@docs
GeophysicalFlows.SingleLayerQG.potential_energy
```

The total energy is:

```@docs
GeophysicalFlows.SingleLayerQG.energy
```

### Implementation

The equation is time-stepped forward in Fourier space:

```math
\partial_t \widehat{q} = - \widehat{\mathsf{J}(\psi, q + \eta)} + \beta \frac{\mathrm{i} k_x}{k^2 + 1/\ell^2} \widehat{q} - \left(\mu + \nu k^{2n_\nu} \right) \widehat{q} + \widehat{f} \ .
```

In doing so the Jacobian is computed in the conservative form: ``\mathsf{J}(f,g) =
\partial_y [ (\partial_x f) g] -\partial_x[ (\partial_y f) g]``.

Thus:

```math
\begin{aligned}
\mathcal{L} & = \beta \frac{\mathrm{i} k_x}{k^2 + 1/\ell^2} - \mu - \nu k^{2n_\nu} \ , \\
\mathcal{N}(\widehat{q}) & = - \mathrm{i} k_x \mathrm{FFT}[u (q+\eta)] - \mathrm{i} k_y \mathrm{FFT}[\upsilon (q+\eta)] \ .
\end{aligned}
```


## Examples

- `examples/singlelayerqg_betadecay.jl`: A script that simulates decaying quasi-geostrophic flow on a beta-plane demonstrating zonation.

- `examples/singlelayerqg_betaforced.jl`: A script that simulates forced-dissipative quasi-geostrophic flow on a beta plane demonstrating zonation. The forcing is temporally delta-correlated with isotropic spatial structure with power in a narrow annulus in wavenumber space that corresponds to total wavenumber ``k_f``.

- `examples/singlelayerqg_decay_topography.jl`: A script that simulates two dimensional turbulence (barotropic quasi-geostrophic flow with ``\beta=0``) above topography.
