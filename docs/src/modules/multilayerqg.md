# MultiLayerQG

### Basic Equations

This module solves the layered quasi-geostrophic equations on a beta plane of variable fluid 
depth ``H - h(x, y)``. The flow in each layer is obtained through a streamfunction ``\psi_j`` as 
``(u_j, v_j) = (-\partial_y \psi_j, \partial_x \psi_j)``, ``j = 1, \dots, n``, where ``n`` 
is the number of fluid layers.

The QGPV in each layer is

```math
\mathrm{QGPV}_j = q_j + \underbrace{f_0 + \beta y}_{\textrm{planetary PV}} + \delta_{j, n} \underbrace{\frac{f_0 h}{H_n}}_{\textrm{topographic PV}}, \quad j = 1, \dots, n .
```

where ``q_j`` incorporates the relative vorticity in each layer ``\nabla^2 \psi_j`` and the 
vortex stretching terms:

```math
\begin{aligned}
q_1 &= \nabla^2 \psi_1 + F_{3/2, 1} (\psi_2 - \psi_1) ,\\
q_j &= \nabla^2 \psi_j + F_{j-1/2, j} (\psi_{j-1} - \psi_j) + F_{j+1/2, j} (\psi_{j+1} - \psi_j) , \quad j = 2, \dots, n-1 ,\\
q_n &= \nabla^2 \psi_n + F_{n-1/2, n} (\psi_{n-1} - \psi_n) .
\end{aligned}
```

with

```math
F_{j+1/2, k} = \frac{f_0^2}{g'_{j+1/2} H_k} \quad \text{and} \quad
g'_{j+1/2} = g \frac{\rho_{j+1} - \rho_j}{\rho_{j+1}} .
```

In view of the relationships above, when we convert to Fourier space ``q``'s and ``\psi``'s are 
related via the matrix equation

```math
\begin{pmatrix} \widehat{q}_{𝐤, 1}\\\vdots\\\widehat{q}_{𝐤, n} \end{pmatrix} =
\underbrace{\left(-|𝐤|^2 \mathbb{1} + \mathbb{F} \right)}_{\equiv \mathbb{S}_{𝐤}}
\begin{pmatrix} \widehat{\psi}_{𝐤, 1}\\\vdots\\\widehat{\psi}_{𝐤, n} \end{pmatrix}
```

where

```math
\mathbb{F} \equiv \begin{pmatrix}
 -F_{3/2, 1} &              F_{3/2, 1}  &   0   &  \cdots    & 0\\
  F_{3/2, 2} & -(F_{3/2, 2}+F_{5/2, 2}) & F_{5/2, 2} &       & \vdots\\
 0           &                  \ddots  &   \ddots   & \ddots & \\
 \vdots      &                          &            &        &  0 \\
 0           &       \cdots             &   0   & F_{n-1/2, n} & -F_{n-1/2, n}
\end{pmatrix} .
```

Including an imposed zonal flow ``U_j(y)`` in each layer, the equations of motion are:

```math
\partial_t q_j + \mathsf{J}(\psi_j, q_j ) + (U_j - \partial_y\psi_j) \partial_x Q_j +  U_j \partial_x q_j  + (\partial_y Q_j)(\partial_x \psi_j) = -\delta_{j, n} \mu \nabla^2 \psi_n - \nu (-1)^{n_\nu} \nabla^{2 n_\nu} q_j ,
```

with

```math
\begin{aligned}
\partial_y Q_j &\equiv \beta - \partial_y^2 U_j - (1-\delta_{j,1}) F_{j-1/2, j} (U_{j-1} - U_j) - (1 - \delta_{j,n}) F_{j+1/2, j} (U_{j+1} - U_j) + \delta_{j,n} \partial_y \eta , \\
\partial_x Q_j &\equiv \delta_{j, n} \partial_x \eta .
\end{aligned}
```


### Implementation

Matrices ``\mathbb{S}_{𝐤}`` as well as ``\mathbb{S}^{-1}_{𝐤}`` are included in `params` as 
`params.S` and `params.S⁻¹` respectively. Additionally, the background PV gradients 
``\partial_x Q`` and ``\partial_y Q`` are also included in the `params` as `params.Qx` and 
`params.Qy`.

One can get the ``\widehat{\psi}_j`` from ``\widehat{q}_j`` via 
`streamfunctionfrompv!(psih, qh, params, grid)`, while the inverse, i.e. obtain ``\widehat{q}_j`` from ``\widehat{\psi}_j``, is done via  `pvfromstreamfunction!(qh, psih, params, grid)`.

The equations of motion are time-stepped forward in Fourier space:

```math
\partial_t \widehat{q}_j = - \widehat{\mathsf{J}(\psi_j, q_j)}  - \widehat{U_j \partial_x Q_j} - \widehat{U_j \partial_x q_j}
+ \widehat{(\partial_y \psi_j) \partial_x Q_j}  - \widehat{(\partial_x \psi_j)(\partial_y Q_j)} + \delta_{j, n} \mu |𝐤|^{2} \widehat{\psi}_n - \nu |𝐤|^{2n_\nu} \widehat{q}_j .
```

In doing so the Jacobian is computed in the conservative form: ``\mathsf{J}(f,g) =
\partial_y [ (\partial_x f) g] - \partial_x[ (\partial_y f) g]``.

The state variable `sol` consists of the Fourier transforms of ``q_j`` at each layer, i.e., 
[`qh`](@ref GeophysicalFlows.MultiLayerQG.Vars).

The linear operator is constructed in `Equation`

```@docs
GeophysicalFlows.MultiLayerQG.Equation
GeophysicalFlows.MultiLayerQG.hyperviscosity
```

The nonlinear terms are computed via

```@docs
GeophysicalFlows.MultiLayerQG.calcN!
```

which in turn calls [`calcN_advection!`](@ref GeophysicalFlows.MultiLayerQG.calcN_advection!) 
and [`addforcing!`](@ref GeophysicalFlows.MultiLayerQG.addforcing!).

!!! tip "Linearized MultiLayerQG dynamics"
    The `MultiLayerQG` module includes also a linearized version of the dynamics about a base
    flow ``U_j(y)``, ``j = 1, \dots, n``; see [`LinearEquation`](@ref GeophysicalFlows.MultiLayerQG.LinearEquation), 
    [`calcNlinear!`](@ref GeophysicalFlows.MultiLayerQG.calcNlinear!), and 
    [`calcN_linearadvection!`](@ref GeophysicalFlows.MultiLayerQG.calcN_linearadvection!).


### Parameters and Variables

All required parameters are included inside [`Params`](@ref GeophysicalFlows.MultiLayerQG.Params)
and all module variables are included inside [`Vars`](@ref GeophysicalFlows.MultiLayerQG.Vars).

For the decaying case (no forcing, ``F=0``), `vars` can be constructed with [`DecayingVars`](@ref GeophysicalFlows.MultiLayerQG.DecayingVars). 
For the forced case (``F \ne 0``) the `vars` struct is with [`ForcedVars`](@ref GeophysicalFlows.MultiLayerQG.ForcedVars) or [`StochasticForcedVars`](@ref GeophysicalFlows.MultiLayerQG.StochasticForcedVars).


### Helper functions

```@docs
GeophysicalFlows.MultiLayerQG.set_q!
GeophysicalFlows.MultiLayerQG.set_ψ!
GeophysicalFlows.MultiLayerQG.updatevars!
```

### Diagnostics

The eddy kinetic energy in each layer and the eddy potential energy that corresponds to each 
fluid interface is computed via `energies`:

```@docs
GeophysicalFlows.MultiLayerQG.energies
```

The lateral eddy fluxes in each layer and the vertical fluxes across fluid interfaces are
computed via `fluxes`:

```@docs
GeophysicalFlows.MultiLayerQG.fluxes
```


## Examples

 - [`examples/multilayerqg_2layer.jl`](@ref multilayerqg_2layer_example): Simulate the growth and equilibration of baroclinic
   eddy turbulence in the Phillips 2-layer model.
