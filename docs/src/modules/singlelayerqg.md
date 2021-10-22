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

The dynamical variable is ``q``.  Thus, the equation solved by the module is:

```math
\partial_t q + \mathsf{J}(\psi, q + \eta) + \beta \partial_x \psi = 
\underbrace{-\left[\mu + \nu(-1)^{n_\nu} \nabla^{2n_\nu} \right] q}_{\textrm{dissipation}} + F .
```

where ``\mathsf{J}(a, b) = (\partial_x a)(\partial_y b)-(\partial_y a)(\partial_x b)`` is the 
two-dimensional Jacobian. On the right hand side, ``F(x, y, t)`` is forcing, ``\mu`` is 
linear drag, and ``\nu`` is hyperviscosity of order ``n_\nu``. Plain old viscosity corresponds 
to ``n_\nu = 1``.


### Implementation

The equation is time-stepped forward in Fourier space:

```math
\partial_t \widehat{q} = - \widehat{\mathsf{J}(\psi, q + \eta)} + \beta \frac{i k_x}{|𝐤|^2 + 1/\ell^2} \widehat{q} - \left(\mu + \nu |𝐤|^{2n_\nu} \right) \widehat{q} + \widehat{F} .
```

The state variable `sol` is the Fourier transform of the sum of relative vorticity and vortex 
stretching (when the latter is applicable), [`qh`](@ref GeophysicalFlows.SingleLayerQG.Vars).

The Jacobian is computed in the conservative form: ``\mathsf{J}(f, g) =
\partial_y [ (\partial_x f) g] - \partial_x[ (\partial_y f) g]``.

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

For decaying case (no forcing, ``F=0``), `vars` can be constructed with [`DecayingVars`](@ref GeophysicalFlows.SingleLayerQG.DecayingVars). 
For the forced case (``F \ne 0``) the `vars` struct is with [`ForcedVars`](@ref GeophysicalFlows.SingleLayerQG.ForcedVars) or [`StochasticForcedVars`](@ref GeophysicalFlows.SingleLayerQG.StochasticForcedVars).


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

- [`examples/singlelayerqg_betadecay.jl`](../literated/singlelayerqg_betadecay/): A script that simulates decaying quasi-geostrophic flow on a beta plane demonstrating zonation.

- [`examples/singlelayerqg_betaforced.jl`](../literated/singlelayerqg_betaforced/): A script that simulates forced-dissipative quasi-geostrophic flow on a beta plane demonstrating zonation. The forcing is temporally delta-correlated with isotropic spatial structure with power in a narrow annulus in wavenumber space with total wavenumber ``k_f``.

- [`examples/singlelayerqg_decay_topography.jl`](../literated/singlelayerqg_decay_topography/): A script that simulates two dimensional turbulence (barotropic quasi-geostrophic flow with ``\beta=0``) above topography.

- [`examples/singlelayerqg_decaying_barotropic_equivalentbarotropic.jl`](../literated singlelayerqg_decaying_barotropic_equivalentbarotropic/): A script that simulates two dimensional turbulence (``\beta=0``) with both infinite and finite Rossby radius of deformation and compares the evolution of the two.