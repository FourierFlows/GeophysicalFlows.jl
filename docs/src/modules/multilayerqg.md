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



$$\underbrace{f_0 + \beta y}_{\text{planetary PV}} + \underbrace{(\partial_x \upsilon
	- \partial_y u)}_{\text{relative vorticity}} +
	\underbrace{\frac{f_0 h}{H}}_{\text{topographic PV}}.$$

The dynamical variable is the component of the vorticity of the flow normal to the plane of motion, $\zeta\equiv \partial_x \upsilon- \partial_y u = \nabla^2\psi$. Also, we denote the topographic PV with $\eta\equiv f_0 h/H$. Thus, the equation solved by the module is:

$$\partial_t \zeta + \J(\psi, \underbrace{\zeta + \eta}_{\equiv q}) +
\beta\partial_x\psi = \underbrace{-\left[\mu + \nu(-1)^{n_\nu} \nabla^{2n_\nu}
\right] \zeta }_{\textrm{dissipation}} + f\ .$$

where $\J(a, b) = (\partial_x a)(\partial_y b)-(\partial_y a)(\partial_x b)$. On the right hand side, $f(x,y,t)$ is forcing, $\mu$ is linear drag, and $\nu$ is hyperviscosity. Plain old viscosity corresponds to $n_{\nu}=1$. The sum of relative vorticity and topographic PV is denoted with $q\equiv\zeta+\eta$.

### Implementation

The equation is time-stepped forward in Fourier space:

$$\partial_t \widehat{\zeta} = - \widehat{\J(\psi, q)} +\beta\frac{\mathrm{i}k_x}{k^2}\widehat{\zeta} -\left(\mu
+\nu k^{2n_\nu}\right) \widehat{\zeta}  + \widehat{f}\ .$$

In doing so the Jacobian is computed in the conservative form: $\J(f,g) =
\partial_y [ (\partial_x f) g] -\partial_x[ (\partial_y f) g]$.

Thus:

$$\mathcal{L} = \beta\frac{\mathrm{i}k_x}{k^2} - \mu - \nu k^{2n_\nu}\ ,$$
$$\mathcal{N}(\widehat{\zeta}) = - \mathrm{i}k_x \mathrm{FFT}(u q)-
	\mathrm{i}k_y \mathrm{FFT}(\upsilon q)\ .$$
