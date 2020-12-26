# BarotropicQGQL Module

### Basic Equations

This module solves the *quasi-linear* quasi-geostrophic barotropic vorticity equation on a beta-plane of variable fluid depth ``H - h(x, y)``. 
Quasi-linear refers to the dynamics that *neglect* the eddy--eddy interactions in the eddy evolution equation after an eddy--mean flow decomposition, e.g., 

```math
\phi(x, y, t) = \overline{\phi}(y, t) + \phi'(x, y, t) ,
```

where overline above denotes a zonal mean, ``\overline{\phi}(y, t) = \int \phi(x, y, t) \, 𝖽x / L_x``, and prime denotes deviations from the zonal mean. This approximation is used in many process-model studies of zonation, e.g., 

- Farrell, B. F. and Ioannou, P. J. (2003). [Structural stability of turbulent jets.](http://doi.org/10.1175/1520-0469(2003)060<2101:SSOTJ>2.0.CO;2) *J. Atmos. Sci.*, **60**, 2101-2118.
- Srinivasan, K. and Young, W. R. (2012). [Zonostrophic instability.](http://doi.org/10.1175/JAS-D-11-0200.1) *J. Atmos. Sci.*, **69 (5)**, 1633-1656.
- Constantinou, N. C., Farrell, B. F., and Ioannou, P. J. (2014). [Emergence and equilibration of jets in beta-plane turbulence: applications of Stochastic Structural Stability Theory.](http://doi.org/10.1175/JAS-D-13-076.1) *J. Atmos. Sci.*, **71 (5)**, 1818-1842.

As in the [BarotropicQG module](barotropicqg.md), the flow is obtained through a streamfunction $\psi$ as ``(u, v) = (-\partial_y \psi, \partial_x \psi)``. All flow fields can be obtained from the quasi-geostrophic potential vorticity (QGPV). Here the QGPV is

```math
\underbrace{f_0 + \beta y}_{\text{planetary PV}} + \underbrace{(\partial_x v
	- \partial_y u)}_{\text{relative vorticity}} +
	\underbrace{\frac{f_0 h}{H}}_{\text{topographic PV}} .
```

The dynamical variable is the component of the vorticity of the flow normal to the plane of motion, ``\zeta \equiv \partial_x v - \partial_y u = \nabla^2 \psi``. Also, we denote the topographic PV with ``\eta \equiv f_0 h/H``. After we apply the eddy-mean flow decomposition above, the QGPV dynamics are:

```math
\begin{aligned}
\partial_t \overline{\zeta} & + \mathsf{J}(\overline{\psi}, \underbrace{\overline{\zeta} + \overline{\eta}}_{\equiv \overline{q}}) + \overline{\mathsf{J}(\psi', \underbrace{\zeta' + \eta'}_{\equiv q'})} = \underbrace{- \left[\mu + \nu(-1)^{n_\nu} \nabla^{2n_\nu}
\right] \overline{\zeta} }_{\textrm{dissipation}} ,\\
\partial_t \zeta' &+ \mathsf{J}(\psi', \overline{q}) + \mathsf{J}(\overline{\psi}, q') + \underbrace{\mathsf{J}(\psi', q') - \overline{\mathsf{J}(\psi', q')}}_{\textrm{EENL}} + 
\beta \partial_x \psi' = \underbrace{-\left[\mu + \nu(-1)^{n_\nu} \nabla^{2n_\nu}
\right] \zeta'}_{\textrm{dissipation}} + f .
\end{aligned}
```

where ``\mathsf{J}(a, b) = (\partial_x a)(\partial_y b) - (\partial_y a)(\partial_x b)``. On the right hand side, ``f(x, y, t)`` is forcing (which is assumed to have zero mean, ``\overline{f} = 0``), ``\mu`` is linear drag, and ``\nu`` is hyperviscosity. Plain old viscosity corresponds to ``n_{\nu} = 1``. The sum of relative vorticity and topographic PV is denoted with ``q \equiv \zeta + \eta``.

*Quasi-linear* dynamics **neglect the term eddy-eddy nonlinearity (EENL) term** above.

### Implementation

The equation is time-stepped forward in Fourier space:

```math
\partial_t \widehat{\zeta} = - \widehat{\mathsf{J}(\psi, q)}^{\textrm{QL}} + \beta \frac{i k_x}{|𝐤|^2} \widehat{\zeta} - \left ( \mu + \nu |𝐤|^{2n_\nu} \right ) \widehat{\zeta} + \widehat{f} .
```

In doing so the Jacobian is computed in the conservative form: ``\mathsf{J}(f,g) =
\partial_y [ (\partial_x f) g] -\partial_x[ (\partial_y f) g]``. The superscript QL in the Jacobian term above denotes that remove triad interactions that correspond to the EENL term.

Thus:

```math
\begin{aligned}
L & = \beta \frac{i k_x}{|𝐤|^2} - \mu - \nu |𝐤|^{2n_\nu} ,\\
N(\widehat{\zeta}) & = - i k_x \mathrm{FFT}(u q)^{\textrm{QL}} - i k_y \mathrm{FFT}(v q)^{\textrm{QL}} .
\end{aligned}
```


## Examples

- `examples/barotropicqgql_betaforced.jl`: A script that simulates forced-dissipative quasi-linear quasi-geostrophic flow on a beta-plane demonstrating zonation. The forcing is temporally delta-correlated and its spatial structure is isotropic with power in a narrow annulus of total radius ``k_f`` in wavenumber space. This example demonstrates that the anisotropic inverse energy cascade is not required for zonation.
