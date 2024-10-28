# SingleLayerQG

### Basic Equations

This module solves the barotropic or equivalent barotropic quasi-geostrophic vorticity equation 
on a beta plane of variable fluid depth ``H - h(x, y)``. The flow is obtained through a 
streamfunction ``\psi`` as ``(u, v) = (-\partial_y \psi, \partial_x \psi)``. All flow fields 
can be obtained from the quasi-geostrophic potential vorticity (QGPV). Here the QGPV is

```math
	\underbrace{f_0 + \beta y}_{\text{planetary PV}} + \underbrace{\partial_x v
	- \partial_y u}_{\text{relative vorticity}}
	\underbrace{ - \frac{1}{\ell^2} \psi}_{\text{vortex stretching}} + 
	\underbrace{\frac{f_0 h}{H}}_{\text{topographic PV}} ,
```

where ``\ell`` is the Rossby radius of deformation. Purely barotropic dynamics corresponds to 
infinite Rossby radius of deformation (``\ell = \infty``), while a flow with a finite Rossby 
radius follows is said to obey equivalent-barotropic dynamics. We denote the sum of the relative
vorticity and the vortex stretching contributions to the QGPV with ``q \equiv \nabla^2 \psi - \psi / \ell^2``.
Also, we denote the topographic PV with ``\eta \equiv f_0 h / H``.

The dynamical variable is ``q``. Including an imposed zonal flow ``U(y)``, the equation of motion is:

```math
\partial_t q + \mathsf{J}(\psi, q) + (U - \partial_y\psi) \partial_x Q +  U \partial_x q + (\partial_y Q)(\partial_x \psi) = \underbrace{-\left[\mu + \nu(-1)^{n_\nu} \nabla^{2n_\nu} \right] q}_{\textrm{dissipation}} + F ,
```

with

```math
\begin{aligned}
\partial_y Q &\equiv \beta - \partial_y^2 U + \partial_y \eta , \\
\partial_x Q &\equiv \partial_x \eta ,
\end{aligned}
```

the background PV gradient components, and with
``\mathsf{J}(a, b) = (\partial_x a)(\partial_y b) - (\partial_y a)(\partial_x b)``
the two-dimensional Jacobian. On the right hand side, ``F(x, y, t)`` is forcing, ``\mu`` is 
linear drag, and ``\nu`` is hyperviscosity of order ``n_\nu``. Plain old viscosity corresponds 
to ``n_\nu = 1``.

In the case that the imposed background zonal flow is just a constant, the above simplifies to:

```math
\partial_t q + \mathsf{J}(\psi, q + \eta) + U \partial_x (q + \eta) + β \partial_x \psi = \underbrace{-\left[\mu + \nu(-1)^{n_\nu} \nabla^{2n_\nu} \right] q}_{\textrm{dissipation}} + F ,
```

and thus the advection of ``q`` can be incorporated in the linear term ``L``.

### Implementation

The equation is time-stepped forward in Fourier space:

```math
\partial_t \widehat{q} = - \widehat{\mathsf{J}(\psi, q)} - \widehat{U \partial_x Q} - \widehat{U \partial_x q}
+ \widehat{(\partial_y \psi) (\partial_x Q)}  - \widehat{(\partial_x \psi)(\partial_y Q)} - \left(\mu + \nu |𝐤|^{2n_\nu} \right) \widehat{q} + \widehat{F} .
```

In doing so the Jacobian is computed in the conservative form: ``\mathsf{J}(f,g) =
\partial_y [ (\partial_x f) g] - \partial_x[ (\partial_y f) g]``.

The state variable `sol` is the Fourier transform of the sum of relative vorticity and vortex 
stretching (when the latter is applicable), [`qh`](@ref GeophysicalFlows.SingleLayerQG.Vars).

The linear operator is constructed in `Equation`

```@docs
GeophysicalFlows.SingleLayerQG.Equation
```

The nonlinear terms are computed via

```@docs
GeophysicalFlows.SingleLayerQG.calcN!
```

which in turn calls [`calcN_advection!`](@ref GeophysicalFlows.SingleLayerQG.calcN_advection!) 
and [`addforcing!`](@ref GeophysicalFlows.SingleLayerQG.addforcing!).


### Parameters and Variables

All required parameters are included inside [`Params`](@ref GeophysicalFlows.SingleLayerQG.Params)
and all module variables are included inside [`Vars`](@ref GeophysicalFlows.SingleLayerQG.Vars).

For the decaying case (no forcing, ``F = 0``), variables are constructed with [`Vars`](@ref GeophysicalFlows.SingleLayerQG.Vars).
For the forced case (``F \ne 0``) variables are constructed with either [`ForcedVars`](@ref GeophysicalFlows.SingleLayerQG.ForcedVars)
or [`StochasticForcedVars`](@ref GeophysicalFlows.SingleLayerQG.StochasticForcedVars).


### Helper functions

Some helper functions included in the module are:

```@docs
GeophysicalFlows.SingleLayerQG.updatevars!
GeophysicalFlows.SingleLayerQG.set_q!
```


### Diagnostics

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

Other diagnostic include: [`energy_dissipation`](@ref GeophysicalFlows.SingleLayerQG.energy_dissipation), 
[`energy_drag`](@ref GeophysicalFlows.SingleLayerQG.energy_drag), [`energy_work`](@ref GeophysicalFlows.SingleLayerQG.energy_work), 
[`enstrophy_dissipation`](@ref GeophysicalFlows.SingleLayerQG.enstrophy_dissipation), and
[`enstrophy_drag`](@ref GeophysicalFlows.SingleLayerQG.enstrophy_drag), [`enstrophy_work`](@ref GeophysicalFlows.SingleLayerQG.enstrophy_work).


## Examples

- [`examples/singlelayerqg_betadecay.jl`](@ref singlelayerqg_betadecay_example): Simulate decaying quasi-geostrophic flow on
  a beta plane demonstrating zonation.

- [`examples/singlelayerqg_betaforced.jl`](@ref singlelayerqg_betaforced_example): Simulate forced-dissipative quasi-geostrophic
  flow on a beta plane demonstrating zonation. The forcing is temporally delta-correlated with isotropic spatial structure with
  power in a narrow annulus in wavenumber space with total wavenumber ``k_f``.

- [`examples/singlelayerqg_decay_topography.jl`](@ref singlelayerqg_decay_topography_example): Simulate two dimensional turbulence
  (barotropic quasi-geostrophic flow with ``\beta=0``) above topography.

- [`examples/singlelayerqg_decaying_barotropic_equivalentbarotropic.jl`](@ref singlelayerqg_decaying_barotropic_equivalentbarotropic_example):
  Simulate two dimensional turbulence (``\beta=0``) with both infinite and finite Rossby radius of deformation and compares the evolution of the two.
