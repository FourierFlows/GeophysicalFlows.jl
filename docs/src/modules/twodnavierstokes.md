# TwoDNavierStokes Module


### Basic Equations

This module solves two-dimensional incompressible Navier-Stokes equations using the 
vorticity-streamfunction formulation. The flow ``\boldsymbol{u} = (u, v)`` is obtained through 
a streamfunction ``\psi`` as ``(u, v) = (-\partial_y \psi, \partial_x \psi)``. The only non-zero 
component of vorticity is that normal to the plane of motion, 
``\partial_x v - \partial_y u = \nabla^2 \psi``. The equation solved by the module is:

```math
\partial_t \zeta + \mathsf{J}(\psi, \zeta) = \underbrace{-\left [ \mu (-\nabla^2)^{n_\mu}
+ \nu (-\nabla^2)^{n_\nu} \right ] \zeta}_{\textrm{dissipation}} + F \ .
```

where ``\mathsf{J}(a, b) = (\partial_x a)(\partial_y b) - (\partial_y a)(\partial_x b)``. On
the right hand side, ``F(x, y, t)`` is forcing. The ``ν`` and ``μ`` terms are both viscosities 
and typically the former is chosen to act at small scales (``n_ν ≥ 1``) and the latter at 
large scales (``n_ν ≤ 1``). Plain old viscocity corresponds to ``n_ν=1`` while ``n_μ=0`` 
corresponds to linear drag. Values of ``n_ν ≥ 2`` and ``n_μ ≤ -1`` are referred to as 
hyper- and hypo-viscosities, respectively.


### Implementation

The equation is time-stepped forward in Fourier space:

```math
\partial_t \widehat{\zeta} = - \widehat{\mathsf{J}(\psi, \zeta )} - \left ( \mu |𝐤|^{2n_\mu}
+ \nu |𝐤|^{2n_\nu} \right ) \widehat{\zeta} + \widehat{f} \ .
```

In doing so the Jacobian is computed in the conservative form: ``\mathsf{J}(a,b) =
\partial_y [ (\partial_x a) b] -\partial_x[ (\partial_y a) b]``.

The linear operator is constructed in `Equation`

```@docs
GeophysicalFlows.TwoDNavierStokes.Equation
```

The nonlinear terms is computed via

```@docs
GeophysicalFlows.TwoDNavierStokes.calcN!
```

which, in turn, calls 

```@docs
GeophysicalFlows.TwoDNavierStokes.calcN_advection!
```
and

```@docs
GeophysicalFlows.TwoDNavierStokes.addforcing!
```


### AbstractTypes and Functions

**Params**

For the unforced case (``f=0``) parameters AbstractType is build with `Params` and it includes:
- `ν`:   Float; viscosity or hyperviscosity coefficient.
- `nν`: Integer ``>0``; the order of viscosity ``n_ν``. Case ``n_ν = 1`` gives normal viscosity.
- `μ`: Float; bottom drag or hypoviscosity coefficient.
- `nμ`: Integer ``\ge 0``; the order of hypodrag ``n_μ``. Case ``n_μ = 0`` gives plain linear drag ``μ``.

For the forced case (``F\ne 0``) parameters AbstractType is build with `ForcedParams`. It includes all parameters in `Params` and additionally:
- `calcF!`: Function that calculates the forcing ``\widehat{F}``


**Vars**

For the unforced case (``F=0``) variables AbstractType is build with `Vars` and it includes:
- `ζ`: Array of Floats; relative vorticity.
- `u`: Array of Floats; ``x``-velocity, ``u``.
- `v`: Array of Floats; ``y``-velocity, ``v``.
- `sol`: Array of Complex; the solution, ``\widehat{\zeta}``.
- `ζh`: Array of Complex; the Fourier transform ``\widehat{\zeta}``.
- `uh`: Array of Complex; the Fourier transform ``\widehat{u}``.
- `vh`: Array of Complex; the Fourier transform ``\widehat{v}``.

For the forced case (``f \ne 0``) variables AbstractType is build with `ForcedVars`. It includes all variables in `Vars` and additionally:
- `Fh`: Array of Complex; the Fourier transform ``\widehat{f}``.
- `prevsol`: Array of Complex; the values of the solution `sol` at the previous time-step (useful for calculating the work done by the forcing).

### Helper functions

```@docs
GeophysicalFlows.TwoDNavierStokes.updatevars!
```

```@docs
GeophysicalFlows.TwoDNavierStokes.set_ζ!
```


### Diagnostics

```@docs
GeophysicalFlows.TwoDNavierStokes.energy
```

```@docs
GeophysicalFlows.TwoDNavierStokes.enstrophy
```

```@docs
GeophysicalFlows.TwoDNavierStokes.energy_dissipation
```

```@docs
GeophysicalFlows.TwoDNavierStokes.enstrophy_dissipation
```

```@docs
GeophysicalFlows.TwoDNavierStokes.energy_drag
```

```@docs
GeophysicalFlows.TwoDNavierStokes.enstrophy_drag
```

```@docs
GeophysicalFlows.TwoDNavierStokes.energy_work
```

```@docs
GeophysicalFlows.TwoDNavierStokes.enstrophy_work
```


## Examples

- `examples/twodnavierstokes_decaying.jl`: A script that simulates decaying two-dimensional turbulence reproducing the results of the paper by

  > McWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. *J. Fluid Mech.*, **146**, 21-43.

- `examples/twodnavierstokes_stochasticforcing.jl`: A script that simulates forced-dissipative two-dimensional turbulence with isotropic temporally delta-correlated stochastic forcing.

- `examples/twodnavierstokes_stochasticforcing_budgets.jl`: A script that simulates forced-dissipative two-dimensional turbulence demonstrating how we can compute the energy and enstrophy budgets.
