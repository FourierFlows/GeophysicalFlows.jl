# MultilayerQG Module

```math
\newcommand{\J}{\mathsf{J}}
```

### Basic Equations

This module solves the layered quasi-geostrophic equations on a beta-plane of variable fluid depth $H-h(x,y)$. The flow in each layer is obtained through a streamfunction $\psi_j$ as $(u_j, \upsilon_j) = (-\partial_y\psi_j, \partial_x\psi_j)$, $j=1,...,n$, where $n$ is the number of fluid layers.


The QGPV in each layer is

```math
QGPV = q_j  + f_0+\beta y - \delta_{1,j}\frac{f_0 h_{\rm s}}{H_1} + \delta_{1,n}\frac{f_0 h_{\rm b}}{H_n}
```

where

```math
q_1 = \nabla^2\psi_1 + F_{3/2, 1} (\psi_2-\psi_1)\\
q_j = \nabla^2\psi_j + F_{j-1/2, j} (\psi_{j-1}-\psi_j) + F_{j+1/2, j} (\psi_{j+1}-\psi_j) \\
q_n = \nabla^2\psi_n + F_{n-1/2, n} (\psi_{n-1}-\psi_n) \\
```

with

```math
F_{j+1/2, j} = \frac{f_0^2}{g'_{j+1/2} H_j}\quad\text{and}\quad
g'_{j+1/2} = g\frac{\rho_{j+1}-\rho_j}{\rho_{j+1}}.
```




### Implementation

The equation is time-stepped forward in Fourier space:
