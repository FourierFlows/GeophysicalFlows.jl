# MultilayerQG Module

```math
\newcommand{\J}{\mathsf{J}}
```

### Basic Equations

This module solves the layered quasi-geostrophic equations on a beta-plane of variable fluid depth $H-h(x,y)$. The flow in each layer is obtained through a streamfunction $\psi_j$ as $(u_j, \upsilon_j) = (-\partial_y\psi_j, \partial_x\psi_j)$, $j=1,...,n$, where $n$ is the number of fluid layers.

The QGPV in each layer is

```math
\mathrm{QGPV}_j = q_j  + \underbrace{f_0+\beta y}_{\textrm{planetary PV}} + \delta_{j,n}\underbrace{\frac{f_0 h}{H_n}}_{\textrm{topographic PV}},\quad j=1,...,n.
```

where

```math
q_1 = \nabla^2\psi_1 + F_{3/2, 1} (\psi_2-\psi_1),\\
q_j = \nabla^2\psi_j + F_{j-1/2, j} (\psi_{j-1}-\psi_j) + F_{j+1/2, j} (\psi_{j+1}-\psi_j),\quad j=2,\dots,n-1,\\
q_n = \nabla^2\psi_n + F_{n-1/2, n} (\psi_{n-1}-\psi_n).
```

with

```math
F_{j+1/2, k} = \frac{f_0^2}{g'_{j+1/2} H_k}\quad\text{and}\quad
g'_{j+1/2} = g\frac{\rho_{j+1}-\rho_j}{\rho_{j+1}} .
```

Therefore, in Fourier space the $q$'s and $\psi$'s are related through

```math
\begin{pmatrix} \widehat{q}_{\boldsymbol{k},1}\\\vdots\\\widehat{q}_{\boldsymbol{k},n} \end{pmatrix} =
\underbrace{\left(-|\boldsymbol{k}|^2\mathbb{1} + \mathbb{F} \right)}_{\equiv \mathbb{S}_{\boldsymbol{k}}}
\begin{pmatrix} \widehat{\psi}_{\boldsymbol{k},1}\\\vdots\\\widehat{\psi}_{\boldsymbol{k},n} \end{pmatrix}
```

where

```math
\mathbb{F} \equiv \begin{pmatrix}
 -F_{3/2, 1} &              F_{3/2, 1}  &   0   &  \cdots    & 0\\
  F_{3/2, 2} & -(F_{3/2, 2}+F_{5/2, 2}) & F_{5/2, 2} &       & \vdots\\
 0           &                  \ddots  &   \ddots   & \ddots & \\
 \vdots      &                          &            &        &  0 \\
 0           &       \cdots             &   0   & F_{n-1/2, n} & -F_{n-1/2, n}
\end{pmatrix}
```

Including an imposed zonal flow $U_j(y)$ in each layer the equations of motion are:

```math
\partial_t q_j + \J(\psi_j, q_j ) + (U_j - \partial_y\psi_j) \partial_x Q_j +  U_j \partial_x q_j  + (\partial_y Q_j)(\partial_x\psi_j) = -\delta_{j,n}\mu\nabla^2\psi_n - \nu(-1)^{n_\nu} \nabla^{2n_\nu} q_j,
```

with

```math
\partial_y Q_j \equiv \beta - \partial_y^2 U_j - (1-\delta_{j,1})F_{j-1/2, j} (U_{j-1}-U_j) - (1-\delta_{j,n})F_{j+1/2, j} (U_{j+1}-U_j) + \delta_{j,n}\partial_y\eta, \\
\partial_x Q_j \equiv \delta_{j,n}\partial_x\eta.
```

The eddy kinetic energy in each layer is:

```math
\textrm{KE}_j = \dfrac{H_j}{H} \int \dfrac1{2} |\boldsymbol{\nabla}\psi_j|^2 \frac{\mathrm{d}^2\boldsymbol{x}}{L_x L_y},\quad j=1,\dots,n,
```

while the eddy potential energy related to each of fluid interface is

```math
\textrm{PE}_{j+1/2} = \int \dfrac1{2} \dfrac{f_0^2}{g'_{j+1/2}} (\psi_j-\psi_{j+1})^2 \frac{\mathrm{d}^2\boldsymbol{x}}{L_x L_y},\quad j=1,\dots,n-1.
```

The lateral eddy fluxes in each layer are:

```math
\textrm{lateralfluxes}_j = \dfrac{H_j}{H} \int \dfrac1{2} U_j\,\upsilon_j \,\partial_y u_j \frac{\mathrm{d}^2\boldsymbol{x}}{L_x L_y},\quad j=1,\dots,n,
```

while the vertical fluxes accros fluid interfaces are:

```math
\textrm{verticalfluxes}_{j+1/2} = \int \dfrac{f_0^2}{g'_{j+1/2} H} (U_j-U_{j+1})\,\upsilon_{j+1}\,\psi_{j} \frac{\mathrm{d}^2\boldsymbol{x}}{L_x L_y},\quad j=1,\dots,n-1.

```


### Implementation

Matrices $\mathbb{S}_{\boldsymbol{k}}$ as well as $\mathbb{S}^{-1}_{\boldsymbol{k}}$ are included in `params` as `params.S` and `params.invS` respectively.

You can get $\widehat{\psi}_j$ from $\widehat{q}_j$ with `streamfunctionfrompv!(psih, qh, invS, grid)`, while to go from $\widehat{\psi}_j$ back to $\widehat{q}_j$ `pvfromstreamfunction!(qh, psih, S, grid)`.




The equations are time-stepped forward in Fourier space:

```math
\partial_t \widehat{q}_j = - \widehat{\J(\psi_j, q_j)}  - \widehat{U_j \partial_x Q_j} - \widehat{U_j \partial_x q_j}
+ \widehat{(\partial_y\psi_j) \partial_x Q_j}  - \widehat{(\partial_x\psi_j)(\partial_y Q_j)} + \delta_{j,n}\mu k^{2} \widehat{\psi}_n - \nu k^{2n_\nu} \widehat{q}_j
```

In doing so the Jacobian is computed in the conservative form: $\J(f,g) =
\partial_y [ (\partial_x f) g] -\partial_x[ (\partial_y f) g]$.


Thus:

$$\mathcal{L} = - \nu k^{2n_\nu}\ ,$$
$$\mathcal{N}(\widehat{q}_j) = - \widehat{\J(\psi_j, q_j)} - \widehat{U_j \partial_x Q_j} - \widehat{U_j \partial_x q_j}
 + \widehat{(\partial_y\psi_j)(\partial_x Q_j)} - \widehat{(\partial_x\psi_j)(\partial_y Q_j)} + \delta_{j,n}\mu k^{2} \widehat{\psi}_n\ .$$
