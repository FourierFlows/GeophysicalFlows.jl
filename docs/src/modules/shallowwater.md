# ShallowWater

### Basic Equations

This module solves the two-dimensional rotating shallow water equations for a fluid layer
over bottom bathymetry:

```math
\begin{aligned}
\partial_t u + \bm{u \cdot \nabla} u - f v & = - g \partial_x \eta , \\
\partial_t v + \bm{u \cdot \nabla} v + f u & = - g \partial_y \eta , \\
\partial_t h + \bm{\nabla \cdot} (\bm{u} h) & = 0 ,
\end{aligned}
```

where ``\bm{u}(\bm{x}, t) = (u, v)`` is the horizontal fluid velocity, ``h(\bm{x}, t)`` is the
total fluid depth, ``f`` is the Coriolis parameter, and ``g`` is the gravitational acceleration.
The free-surface elevation ``\eta(\bm{x}, t)`` and the bottom bathymetry ``b(\bm{x})`` are both
measured from the rest height of the free surface, ``z = 0``. Thus,

```math
\eta(\bm{x}, t) = h(\bm{x}, t) + b(\bm{x}) .
```

For example, a flat bottom at depth ``H`` corresponds to ``b = -H`` and then, at rest, ``h = H``.

The module evolves the equations in conservative form, using the volume fluxes ``q_u = h u``
and ``q_v = h v`` and the depth ``h`` as variables:

```math
\begin{aligned}
\partial_t q_u + \partial_x \left( \frac{q_u^2}{h} \right) + \partial_y \left( \frac{q_u q_v}{h} \right) - f q_v & = - g h \partial_x \eta + \mathcal{D} q_u + F_{q_u} , \\
\partial_t q_v + \partial_x \left( \frac{q_u q_v}{h} \right) + \partial_y \left( \frac{q_v^2}{h} \right) + f q_u & = - g h \partial_y \eta + \mathcal{D} q_v + F_{q_v} , \\
\partial_t h + \partial_x q_u + \partial_y q_v & = \mathcal{D} h + F_h ,
\end{aligned}
```

where ``\mathcal{D} = -\nu (-\nabla^2)^{n_\nu}`` is a (hyper)-viscous operator and
``F_{q_u}``, ``F_{q_v}``, ``F_h`` are optional forcing terms.

The pressure-gradient term is written as ``- g h \bm{\nabla} \eta``. For a flat bottom, this is
equivalent to the divergence form ``- g \bm{\nabla} (h^2/2)``. With non-trivial bathymetry,
``- g h \bm{\nabla} \eta = - g \bm{\nabla} (h^2/2) - g h \bm{\nabla} b``. Writing it as
``- g h \bm{\nabla} \eta`` has a numerical benefit: a lake at rest, that is, ``\bm{u} = 0``
and ``\eta = `` constant, is an exact steady solution of the discrete equations, because the
spectral derivative of a constant ``\eta`` vanishes identically. (For that to hold, the
bathymetry is dealiased; see [`Params`](@ref GeophysicalFlows.ShallowWater.Params).)

The (hyper)-viscous operator acts on each of ``q_u``, ``q_v``, and ``h``. Since it vanishes for
the domain-averaged components, dissipation conserves the total mass ``\int h \, 𝖽x 𝖽y`` and the
total volume fluxes ``\int q_u \, 𝖽x 𝖽y`` and ``\int q_v \, 𝖽x 𝖽y``.

The depth ``h`` must remain positive everywhere; the equations are singular when ``h = 0``,
so wetting and drying are not supported.

In the absence of dissipation and forcing, the equations conserve the total mass and the total energy

```math
E = \int \left[ \tfrac1{2} h (u^2 + v^2) + \tfrac1{2} g \eta^2 \right] \frac{𝖽x 𝖽y}{L_x L_y} .
```

### Implementation

The state variable `sol` is a three-dimensional array of size `(nkr, nl, 3)` that holds the
Fourier transforms of ``q_u``, ``q_v``, and ``h``; that is, `sol[:, :, 1]`, `sol[:, :, 2]`,
and `sol[:, :, 3]` correspond to ``\widehat{q_u}``, ``\widehat{q_v}``, and ``\widehat{h}``
respectively. The equations are time-stepped forward in Fourier space:

```math
\begin{aligned}
\partial_t \widehat{q_u} & = - i k_x \widehat{q_u q_u / h} - i k_y \widehat{q_u q_v / h} + f \widehat{q_v} - g \widehat{h \partial_x \eta} - \nu |𝐤|^{2 n_\nu} \widehat{q_u} + \widehat{F_{q_u}} , \\
\partial_t \widehat{q_v} & = - i k_x \widehat{q_u q_v / h} - i k_y \widehat{q_v q_v / h} - f \widehat{q_u} - g \widehat{h \partial_y \eta} - \nu |𝐤|^{2 n_\nu} \widehat{q_v} + \widehat{F_{q_v}} , \\
\partial_t \widehat{h} & = - i k_x \widehat{q_u} - i k_y \widehat{q_v} - \nu |𝐤|^{2 n_\nu} \widehat{h} + \widehat{F_h} .
\end{aligned}
```

Products are computed in physical space and are dealiased.

The linear operator is constructed in `Equation`

```@docs
GeophysicalFlows.ShallowWater.Equation
```

while the nonlinear terms via

```@docs
GeophysicalFlows.ShallowWater.calcN!
```

which in turn calls [`calcN_advection!`](@ref GeophysicalFlows.ShallowWater.calcN_advection!)
and [`addforcing!`](@ref GeophysicalFlows.ShallowWater.addforcing!).

The initial condition is prescribed in terms of the primitive variables ``u``, ``v``, and ``h``
via [`set_uvh!`](@ref GeophysicalFlows.ShallowWater.set_uvh!).


### Parameters and Variables

All required parameters are included inside [`Params`](@ref GeophysicalFlows.ShallowWater.Params)
and all module variables are included inside [`Vars`](@ref GeophysicalFlows.ShallowWater.Vars).

For the decaying case (no forcing, ``F = 0``), `vars` can be constructed with
[`DecayingVars`](@ref GeophysicalFlows.ShallowWater.DecayingVars). For the forced case
(``F \ne 0``) the `vars` struct is with [`ForcedVars`](@ref GeophysicalFlows.ShallowWater.ForcedVars)
or [`StochasticForcedVars`](@ref GeophysicalFlows.ShallowWater.StochasticForcedVars). The forcing
array `Fh` has the same size as `sol`, i.e., it holds the forcing for each of ``q_u``, ``q_v``,
and ``h``.


### Helper functions

```@docs
GeophysicalFlows.ShallowWater.updatevars!
GeophysicalFlows.ShallowWater.set_uvh!
```


### Diagnostics

The domain-averaged mass, kinetic energy, potential energy, and total energy are computed via
[`mass`](@ref GeophysicalFlows.ShallowWater.mass),
[`kinetic_energy`](@ref GeophysicalFlows.ShallowWater.kinetic_energy),
[`potential_energy`](@ref GeophysicalFlows.ShallowWater.potential_energy), and
[`energy`](@ref GeophysicalFlows.ShallowWater.energy).


## Examples

- [`examples/shallowwater_geostrophicadjustment.jl`](@ref shallowwater_geostrophicadjustment_example):
  A simulation of the geostrophic adjustment of an elliptical Gaussian bump in the free surface.
