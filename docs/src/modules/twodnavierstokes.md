# TwoDNavierStokes Module


### Basic Equations

This module solves two-dimensional incompressible turbulence. The flow is given
through a streamfunction $\psi$ as $(u,v) = (-\partial_y\psi, \partial_x\psi)$.
The dynamical variable used here is the component of the vorticity of the flow
normal to the plane of motion, $\zeta=\partial_x v- \partial_y u = \nabla^2\psi$.
The equation solved by the module is:

$$\partial_t \zeta + \mathsf{J}(\psi, \zeta) = \underbrace{-\left[\mu(-1)^{n_\mu} \nabla^{2n_\mu}
+\nu(-1)^{n_\nu} \nabla^{2n_\nu}\right] \zeta}_{\textrm{dissipation}} + f\ .$$

where $\mathsf{J}(a, b) = (\partial_x a)(\partial_y b)-(\partial_y a)(\partial_x b)$. On
the right hand side, $f(x,y,t)$ is forcing, $\mu$ is hypoviscosity, and $\nu$ is
hyperviscosity. Plain old linear drag corresponds to $n_{\mu}=0$, while normal
viscosity corresponds to $n_{\nu}=1$.

### Implementation

The equation is time-stepped forward in Fourier space:

$$\partial_t \widehat{\zeta} = - \widehat{\mathsf{J}(\psi, \zeta)} -\left(\mu k^{2n_\mu}
+\nu k^{2n_\nu}\right) \widehat{\zeta}  + \widehat{f}\ .$$

In doing so the Jacobian is computed in the conservative form: $\mathsf{J}(a,b) =
\partial_y [ (\partial_x a) b] -\partial_x[ (\partial_y a) b]$.

Thus:

$$L = -\mu k^{-2n_\mu} - \nu k^{2n_\nu}\ ,$$
$$N(\widehat{\zeta}) = - ik_x \mathrm{FFT}(u \zeta)-
	ik_y \mathrm{FFT}(v \zeta) + \widehat{f}\ .$$


### AbstractTypes and Functions

**Params**

For the unforced case ($f=0$) parameters AbstractType is build with `Params` and it includes:
- `ν`:   Float; viscosity or hyperviscosity coefficient.
- `nν`: Integer$>0$; the order of viscosity $n_\nu$. Case $n_\nu=1$ gives normal viscosity.
- `μ`: Float; bottom drag or hypoviscosity coefficient.
- `nμ`: Integer$\ge 0$; the order of hypodrag $n_\mu$. Case $n_\mu=0$ gives plain linear drag $\mu$.

For the forced case ($f\ne 0$) parameters AbstractType is build with `ForcedParams`. It includes all parameters in `Params` and additionally:
- `calcF!`: Function that calculates the forcing $\widehat{f}$


**Vars**

For the unforced case ($f=0$) variables AbstractType is build with `Vars` and it includes:
- `zeta`: Array of Floats; relative vorticity.
- `u`: Array of Floats; $x$-velocity, $u$.
- `v`: Array of Floats; $y$-velocity, $v$.
- `sol`: Array of Complex; the solution, $\widehat{\zeta}$.
- `zetah`: Array of Complex; the Fourier transform $\widehat{\zeta}$.
- `uh`: Array of Complex; the Fourier transform $\widehat{u}$.
- `vh`: Array of Complex; the Fourier transform $\widehat{v}$.

For the forced case ($f\ne 0$) variables AbstractType is build with `ForcedVars`. It includes all variables in `Vars` and additionally:
- `Fh`: Array of Complex; the Fourier transform $\widehat{f}$.
- `prevsol`: Array of Complex; the values of the solution `sol` at the previous time-step (useful for calculating the work done by the forcing).



**`calcN!` function**

The nonlinear term $N(\widehat{\zeta})$ is computed via functions:

- `calcN_advection!`: computes $- \widehat{\mathsf{J}(\psi, \zeta)}$ and stores it in array `N`.

- `calcN_forced!`: computes $- \widehat{\mathsf{J}(\psi, \zeta)}$ via `calcN_advection!` and then adds to it the forcing $\widehat{f}$ computed via `calcF!` function. Also saves the solution $\widehat{\zeta}$ of the previous time-step in array `prevsol`.

- `updatevars!`: uses `sol` to compute $\zeta$, $u$, $v$, $\widehat{u}$, and $\widehat{v}$ and stores them into corresponding arrays of `Vars`/`ForcedVars`.


## Examples

- `examples/twodnavierstokes_decaying.jl`: A script that simulates decaying two-dimensional turbulence reproducing the results of the paper by

  > McWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. *J. Fluid Mech.*, **146**, 21-43.

- `examples/twodnavierstokes_stochasticforcing.jl`: A script that simulates forced-dissipative two-dimensional turbulence with isotropic temporally delta-correlated stochastic forcing.
