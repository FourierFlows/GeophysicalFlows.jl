# TwoDNavierStokes


### Basic Equations

This module solves two-dimensional incompressible Navier-Stokes equations using the 
vorticity-streamfunction formulation. The flow ``\bm{u} = (u, v)`` is obtained through a 
streamfunction ``\psi`` as ``(u, v) = (-\partial_y \psi, \partial_x \psi)``. The only non-zero 
component of vorticity is that normal to the plane of motion, 
``\partial_x v - \partial_y u = \nabla^2 \psi``. The module solves the two-dimensional 
vorticity equation:

```math
\partial_t \zeta + \mathsf{J}(\psi, \zeta) = \underbrace{-\left [ \mu (-\nabla^2)^{n_\mu}
+ \nu (-\nabla^2)^{n_\nu} \right ] \zeta}_{\textrm{dissipation}} + F ,
```

where ``\mathsf{J}(\psi, \zeta) = (\partial_x \psi)(\partial_y \zeta) - (\partial_y \psi)(\partial_x \zeta)`` 
is the two-dimensional Jacobian and ``F(x, y, t)`` is forcing. The Jacobian term is the advection
of relative vorticity, ``\mathsf{J}(ψ, ζ) = \bm{u \cdot \nabla} \zeta``. Both ``ν`` and ``μ`` 
terms are viscosities; typically the former is chosen to act at small scales (``n_ν ≥ 1``), 
while the latter at large scales (``n_ν ≤ 0``). Plain old viscocity corresponds to ``n_ν=1`` 
while ``n_μ=0`` corresponds to linear drag. Values of ``n_ν ≥ 2`` or ``n_μ ≤ -1`` are referred 
to as hyper- or hypo-viscosities, respectively.


### Implementation

The equation is time-stepped forward in Fourier space:

```math
\partial_t \widehat{\zeta} = - \widehat{\mathsf{J}(\psi, \zeta)} - \left ( \mu |𝐤|^{2n_\mu}
+ \nu |𝐤|^{2n_\nu} \right ) \widehat{\zeta} + \widehat{F} .
```

The state variable `sol` is the Fourier transform of vorticity, [`ζh`](@ref GeophysicalFlows.TwoDNavierStokes.Vars).

The Jacobian is computed in the conservative form: ``\mathsf{J}(a, b) = 
\partial_y [(\partial_x a) b] - \partial_x[(\partial_y a) b]``.

The linear operator is constructed in `Equation`

```@docs
GeophysicalFlows.TwoDNavierStokes.Equation
```

The nonlinear terms are computed via

```@docs
GeophysicalFlows.TwoDNavierStokes.calcN!
```

which in turn calls [`calcN_advection!`](@ref GeophysicalFlows.TwoDNavierStokes.calcN_advection!) 
and [`addforcing!`](@ref GeophysicalFlows.TwoDNavierStokes.addforcing!).


### Parameters and Variables

All required parameters are included inside [`Params`](@ref GeophysicalFlows.TwoDNavierStokes.Params)
and all module variables are included inside [`Vars`](@ref GeophysicalFlows.TwoDNavierStokes.Vars).

For decaying case (no forcing, ``F=0``), `vars` can be constructed with [`Vars`](@ref GeophysicalFlows.TwoDNavierStokes.Vars). 
For the forced case (``F \ne 0``) the `vars` struct is with [`ForcedVars`](@ref GeophysicalFlows.TwoDNavierStokes.ForcedVars) or [`StochasticForcedVars`](@ref GeophysicalFlows.TwoDNavierStokes.StochasticForcedVars).


### Helper functions

Some helper functions included in the module are:

```@docs
GeophysicalFlows.TwoDNavierStokes.updatevars!
GeophysicalFlows.TwoDNavierStokes.set_ζ!
```


### Diagnostics

Some useful diagnostics are:

```@docs
GeophysicalFlows.TwoDNavierStokes.energy
GeophysicalFlows.TwoDNavierStokes.enstrophy
```

Other diagnostic include: [`energy_dissipation`](@ref GeophysicalFlows.TwoDNavierStokes.energy_dissipation), 
[`energy_work`](@ref GeophysicalFlows.TwoDNavierStokes.energy_work), 
[`enstrophy_dissipation`](@ref GeophysicalFlows.TwoDNavierStokes.enstrophy_dissipation), and
[`enstrophy_work`](@ref GeophysicalFlows.TwoDNavierStokes.enstrophy_work).


## Examples

- [`examples/twodnavierstokes_decaying.jl`](../generated/twodnavierstokes_decaying/): A script that simulates decaying two-dimensional turbulence reproducing the results by

  > McWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. *J. Fluid Mech.*, **146**, 21-43.

- [`examples/twodnavierstokes_stochasticforcing.jl`](../generated/twodnavierstokes_stochasticforcing/): A script that simulates forced-dissipative two-dimensional turbulence with isotropic temporally delta-correlated stochastic forcing.

- [`examples/twodnavierstokes_stochasticforcing_budgets.jl`](../generated/twodnavierstokes_stochasticforcing_budgets/): A script that simulates forced-dissipative two-dimensional turbulence demonstrating how we can compute the energy and enstrophy budgets.
