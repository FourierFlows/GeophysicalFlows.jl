# TwoDNavierStokes Module


### Basic Equations

This module solves two-dimensional incompressible Navier-Stokes equations using the 
vorticity-streamfunction formulation. The flow ``\boldsymbol{u} = (u, v)`` is obtained through 
a streamfunction ``\psi`` as ``(u, v) = (-\partial_y \psi, \partial_x \psi)``. The only non-zero 
component of vorticity is that normal to the plane of motion, 
``\partial_x v - \partial_y u = \nabla^2 \psi``. The module solves the two-dimensional 
vorticity equation:

```math
\partial_t \zeta + \mathsf{J}(\psi, \zeta) = \underbrace{-\left [ \mu (-\nabla^2)^{n_\mu}
+ \nu (-\nabla^2)^{n_\nu} \right ] \zeta}_{\textrm{dissipation}} + F .
```

where ``\mathsf{J}(a, b) = (\partial_x a)(\partial_y b) - (\partial_y a)(\partial_x b)``. On
the right hand side, ``F(x, y, t)`` is forcing. The ``ν`` and ``μ`` terms are both viscosities 
and typically the former is chosen to act at small scales (``n_ν ≥ 1``) while the latter at 
large scales (``n_ν ≤ 0``). Plain old viscocity corresponds to ``n_ν=1`` while ``n_μ=0`` 
corresponds to linear drag. Values of ``n_ν ≥ 2`` and ``n_μ ≤ -1`` are referred to as 
hyper- and hypo-viscosities, respectively.


### Implementation

The equation is time-stepped forward in Fourier space:

```math
\partial_t \widehat{\zeta} = - \widehat{\mathsf{J}(\psi, \zeta )} - \left ( \mu |𝐤|^{2n_\mu}
+ \nu |𝐤|^{2n_\nu} \right ) \widehat{\zeta} + \widehat{f} \ .
```

In doing so, the Jacobian is computed in the conservative form: ``\mathsf{J}(a, b) =
\partial_y [ (\partial_x a) b] - \partial_x[ (\partial_y a) b]``.

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

For the unforced case (``F = 0``), the `params` struct is build with `Params` and it includes:
- `ν :: Float`: small-scale (hyper)-viscosity coefficient,
- `nν :: Int`: (hyper)-viscosity order, `nν ≥ 1`,
- `μ :: Float`: large-scale (hypo)-viscosity coefficient,
- `nμ :: Int`: (hypo)-viscosity order, `nν ≤ 0`,
- `calcF! :: Function`: Function that computes the forcing ``F̂``.

**Vars**

For the unforced case (``F=0``) the `vars` struct with `Vars` and it includes:
- `ζ :: Array{Float}`: relative vorticity, ``ζ``,
- `u :: Array{Float}`: ``x``-velocity, ``u``,
- `v :: Array{Float}`: ``y``-velocity, ``v``,
- `ζh :: Array{Complex{Float}}`: the Fourier transform of ``ζ``,
- `uh :: Array{Complex{Float}}`: the Fourier transform of ``u``,
- `vh :: Array{Complex{Float}}`: the Fourier transform of ``v``.

The state variable `sol` is `ζh`.

For the forced case (``F \ne 0``) the `vars` struct is build with `ForcedVars` or `StochasticForcedVars`. It includes all variables in `Vars` and additionally:
- `Fh :: Complex{Float}`: the Fourier transform ``\widehat{f}``.
- `prevsol :: Array{Complex{Float}}`: the solution `sol` at the previous time-step (needed to calculate the work done by the forcing when forcing is stochastic).

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
GeophysicalFlows.TwoDNavierStokes.enstrophy
```

```@docs
GeophysicalFlows.TwoDNavierStokes.energy_dissipation
GeophysicalFlows.TwoDNavierStokes.energy_work
```

```@docs
GeophysicalFlows.TwoDNavierStokes.enstrophy_dissipation
GeophysicalFlows.TwoDNavierStokes.enstrophy_work
```


## Examples

- `examples/twodnavierstokes_decaying.jl`: A script that simulates decaying two-dimensional turbulence reproducing the results of the paper by

  > McWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. *J. Fluid Mech.*, **146**, 21-43.

- `examples/twodnavierstokes_stochasticforcing.jl`: A script that simulates forced-dissipative two-dimensional turbulence with isotropic temporally delta-correlated stochastic forcing.

- `examples/twodnavierstokes_stochasticforcing_budgets.jl`: A script that simulates forced-dissipative two-dimensional turbulence demonstrating how we can compute the energy and enstrophy budgets.
