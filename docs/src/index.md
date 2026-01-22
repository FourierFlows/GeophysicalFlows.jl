# GeophysicalFlows.jl Documentation

## Overview

`GeophysicalFlows.jl` is a collection of modules which leverage the
[FourierFlows.jl](https://github.com/FourierFlows/FourierFlows.jl) framework to provide
solvers for problems in Geophysical Fluid Dynamics, on periodic domains using Fourier-based pseudospectral methods.


## Examples

Examples aim to demonstrate the main functionalities of each module. Have a look at our Examples collection!


!!! note "Fourier transforms normalization"

    Fourier-based pseudospectral methods rely on Fourier expansions. Throughout the
    documentation we denote symbols with hat, e.g., ``\hat{u}``, to be the Fourier transform
    of ``u`` like, e.g.,

    ```math
    u(x) = \sum_{k_x} \hat{u}(k_x) \, e^{i k_x x} .
    ```

    The convention used in the modules is that the Fourier transform of a variable, e.g., `u`
    is denoted with `uh` (where the trailing `h` is there to imply "hat"). Note, however,
    that `uh` is obtained via a FFT of `u` and due to different normalization factors that the
    FFT algorithm uses, `uh` _is not_ exactly the same as ``\hat{u}`` above. Instead,

    ```math
    \hat{u}(k_x) = \frac{𝚞𝚑}{n_x e^{i k_x x_0}} ,
    ```

    where ``n_x`` is the total number of grid points in ``x`` and ``x_0`` is the left-most
    point of our ``x``-grid.

    Read more in the FourierFlows.jl Documentation; see
    [Grids](https://fourierflows.github.io/FourierFlowsDocumentation/stable/grids/) section.


!!! info "Unicode"
    Oftentimes unicode symbols are used in modules for certain variables or parameters. For
    example, `ψ` is commonly used to denote the streamfunction of the flow, or `∂` is used
    to denote partial differentiation. Unicode symbols can be entered in the Julia REPL by
    typing, e.g., `\psi` or `\partial` followed by the `tab` key.

    Read more about Unicode symbols in the
    [Julia Documentation](https://docs.julialang.org/en/v1/manual/unicode-input/).


## Developers

The development of GeophysicalFlows.jl started during the 21st AOFD Meeting 2017 by [Navid C. Constantinou](http://www.navidconstantinou.com)
and [Gregory L. Wagner](https://glwagner.github.io). Since then various people have contributed, including
[Lia Siegelman](https://scholar.google.com/citations?user=BQJtj6sAAAAJ), [Brodie Pearson](https://brodiepearson.github.io),
[André Palóczy](https://scholar.google.com/citations?user=o4tYEH8AAAAJ) (see the
[example in FourierFlows.jl](https://fourierflows.github.io/FourierFlowsDocumentation/stable/literated/OneDShallowWaterGeostrophicAdjustment/)),
and [others](https://github.com/FourierFlows/GeophysicalFlows.jl/graphs/contributors).


## Citing

If you use GeophysicalFlows.jl in research, teaching, or other activities, we would be grateful
if you could mention GeophysicalFlows.jl and cite our paper in JOSS:

Constantinou et al., (2021). GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs & GPUs. _Journal of Open Source Software_, **6(60)**, 3053, doi:[10.21105/joss.03053](https://doi.org/10.21105/joss.03053).

The bibtex entry for the paper is:

```bibtex
@article{GeophysicalFlowsJOSS,
  doi = {10.21105/joss.03053},
  url = {https://doi.org/10.21105/joss.03053},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {60},
  pages = {3053},
  author = {Navid C. Constantinou and Gregory LeClaire Wagner and Lia Siegelman and Brodie C. Pearson and André Palóczy},
  title = {GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs \& GPUs},
  journal = {Journal of Open Source Software}
}
```

## Papers using `GeophysicalFlows.jl`

1. Parfenyev, V. (2025) Optimal navigation in two-dimensional regular and turbulent flows. arXiv preprint arXiv.2512.08766, doi:[10.48550/arXiv.2512.08766](https://doi.org/10.48550/arXiv.2512.08766).

1. Lemasquerier, D. (2025) Emergence and coalescence of zonal jets: A quasilinear Rossby wave-mean flow interaction model. _Physical Review Fluids_, **10**, 124802, doi:[10.1103/pqxy-x728](https://doi.org/10.1103/pqxy-x728)

1. Sterl, M. F., Palóczy, A., LaCasce, J. H., Baatsen, M. L. J., and Groeskamp, S. (2025) Asymmetric effects of topographic slopes on Lagrangian and Eulerian eddy diffusivities in two-layer QG flow. _ESS Open Archive_, doi:[10.22541/essoar.176365858.87644610/v1](https://doi.org/10.22541/essoar.176365858.87644610/v1).

1. Lobo M. (2025) Baroclinic energy cycles in two-layer quasi-geostrophic turbulence. _Journal of Fluid Mechanics_, **1023**, A2, doi:[10.1017/jfm.2025.10824](https://doi.org/10.1017/jfm.2025.10824).

1. Wagner, C. M, Lee, A., and Pearson, B. (2025) FluidSF: A Python package for calculating turbulent flow statistics. _Journal of Open Source Software_, **10(114)**, 7873, doi:[10.21105/joss.07873](https://doi.org/10.21105/joss.07873).

1. Lobo, M. and Griffies, S. M. (2025) Two-layer quasi-geostrophic waves and turbulence over a meridionally sloping bottom with linear bottom drag. _ESS Open Archive_, doi:[10.22541/essoar.175760447.79218471/v1](https://doi.org/10.22541/essoar.175760447.79218471/v1).

1. Shokar, I. J. S., Kerswell, R. R., and Haynes, P. H. (2025) Conditioning on PDE Parameters to Generalise Deep Learning Emulation of Stochastic and Chaotic Dynamics. arXiv preprint arXiv.2509.09599, doi:[10.48550/arXiv.2509.09599](https://doi.org/10.48550/arXiv.2509.09599).

1. He, J. and Wang, Y. (2025). Minimum-enstrophy and maximum-entropy equilibrium states in two-dimensional topographic turbulence. arXiv preprint arXiv.2507.00691, doi:[10.48550/arXiv.2507.00691](https://doi.org/10.48550/arXiv.2507.00691).

1. Hess, P., Gelbrecht, M., Schötz, C., Aich, M., Huang, Y., Yang, S., and Boers, N. (2025). Generating time-consistent dynamics with discriminator-guided image diffusion models. arXiv preprint arXiv.2410.14402, doi:[10.48550/arXiv.2505.09089](https://doi.org/10.48550/arXiv.2505.09089).

1. Krishna Priya, V. R., Patil, S. S. and Roy, S., Aditya, K., and Lakkaraju, R. (2025). Probing quasigeostrophic turbulence via complex networks _Physical Review Fluids_, **10(5)**, 054402, doi:[10.1103/PhysRevFluids.10.054402](https://doi.org/10.1103/PhysRevFluids.10.054402)

1. Pearson, B., Wagner, C., Fox-Kemper, B., and Samelson, R. (2025). Estimating spectral fluxes in quasi-two-dimensional flows with advective structure functions and Bessel functions. _Journal of Physical Oceanography_, **55(9)**, 1335–1352, doi:[10.1175/JPO-D-24-0211.1](https://doi.org/10.1175/JPO-D-24-0211.1).

1. Crowe, M. N., (2025). QGDipoles.jl: A Julia package for calculating dipolar vortex solutions to the Quasi-Geostrophic equations. _Journal of Open Source Software_, **10(108)**, 7767, doi:[10.21105/joss.07767](https://doi.org/10.21105/joss.07767).

1. Lobo, M., Griffies, S. M., and Zhang, W. (2025) Vertical structure of baroclinic instability in a three-layer quasi-geostrophic model over a sloping bottom. _Journal of Physical Oceanography_, in press, doi:[10.1175/JPO-D-24-0130.1](https://doi.org/10.1175/JPO-D-24-0130.1).

1. Crowe, M. N. and Sutyrin, G. G. (2025) Symmetry breaking of two-layer eastward propagating dipoles. _Phys. Fluids_, **37**, 021708, doi:[10.1063/5.0251761](https://doi.org/10.1063/5.0251761).

1. Pudig, M. and Smith, K. S. (2024) Baroclinic turbulence above rough topography: The vortex gas and topographic turbulence regimes. _ESS Open Archive_, doi:[10.22541/essoar.171995116.60993353/v1](https://doi.org/10.22541/essoar.171995116.60993353/v1).

1. Shokar, I. J. S., Haynes, P. H. and Kerswell, R. R. (2024) Extending deep learning emulation across parameter regimes to assess stochastically driven spontaneous transition events. In ICLR 2024 Workshop on AI4DifferentialEquations in Science. url: [https://openreview.net/forum?id=7a5gUX4e5q](https://openreview.net/forum?id=7a5gUX4e5q).

1. He, J. and Wang, Y. (2024) Multiple states of two-dimensional turbulence above topography. _Journal of Fluid Mechanics_, **994**, R2, doi:[10.1017/jfm.2024.633](https://doi.org/10.1017/jfm.2024.633).

1. Parfenyev, V., Blumenau, M., and Nikitin, I. (2024) Inferring parameters and reconstruction of two-dimensional turbulent flows with physics-informed neural networks. _Jetp Lett._, doi:[10.1134/S0021364024602203](https://doi.org/10.1134/S0021364024602203).

1. Shokar, I. J. S., Kerswell, R. R., and Haynes, P. H. (2024) Stochastic latent transformer: Efficient modeling of stochastically forced zonal jets. _Journal of Advances in Modeling Earth Systems_, **16**, e2023MS004177, doi:[10.1029/2023MS004177](https://doi.org/10.1029/2023MS004177).

1. Bischoff, T., and Deck, K. (2024) Unpaired downscaling of fluid flows with diffusion bridges. _Artificial Intelligence for the Earth Systems_, **3**, e230039, doi:[10.1175/AIES-D-23-0039.1](https://doi.org/10.1175/AIES-D-23-0039.1), in press.

1. Kolokolov, I. V., Lebedev, V. V., and Parfenyev, V. M. (2024) Correlations in a weakly interacting two-dimensional random flow. _Physical Review E_, **109(3)**, 035103, doi:[10.1103/PhysRevE.109.035103](https://doi.org/10.1103/PhysRevE.109.035103).

1. Parfenyev, V. (2024) Statistical analysis of vortex condensate motion in two-dimensional turbulence. _Physics of Fluids_, **36**, 015148, doi:[10.1063/5.0187030](https://doi.org/10.1063/5.0187030).

1. LaCasce, J. H., Palóczy, A., and Trodahl, M. (2024). Vortices over bathymetry. _Journal of Fluid Mechanics_, **979**, A32, doi:[10.1017/jfm.2023.1084](https://doi.org/10.1017/jfm.2023.1084).

1. Drivas, T. D. and Elgindi, T. M. (2023). Singularity formation in the incompressible Euler equation in finite and infinite time. _EMS Surveys in Mathematical Sciences_, **10(1)**, 1–100, doi:[10.4171/emss/66](https://doi.org/10.4171/emss/66).

1. Siegelman, L. and Young, W. R. (2023). Two-dimensional turbulence above topography: Vortices and potential vorticity homogenization. _Proceedings of the National Academy of Sciences_, **120(44)**, e2308018120, doi:[10.1073/pnas.2308018120](https://doi.org/10.1073/pnas.2308018120).

1. Bisits, J. I., Stanley G. J., and Zika, J. D. (2023). Can we accurately quantify a lateral diffusivity using a single tracer release? _Journal of Physical Oceanography_, **53(2)**, 647–659, doi:[10.1175/JPO-D-22-0145.1](https://doi.org/10.1175/JPO-D-22-0145.1).

1. Parfenyev, V. (2022) Profile of a two-dimensional vortex condensate beyond the universal limit. _Phys. Rev. E_, **106**, 025102, doi:[10.1103/PhysRevE.106.025102](https://doi.org/10.1103/PhysRevE.106.025102).

1. Siegelman, L., Young, W. R., and Ingersoll, A. P. (2022). Polar vortex crystals: Emergence and structure _Proceedings of the National Academy of Sciences_, **119(17)**, e2120486119, doi:[10.1073/pnas.2120486119](https://doi.org/10.1073/pnas.2120486119).

1. Dolce, M. and Drivas, T. D. (2022). On maximally mixed equilibria of two-dimensional perfect fluids. _Archive for Rational Mechanics and Analysis_, **246**, 735–770, doi:[10.1007/s00205-022-01825-w](https://doi.org/10.1007/s00205-022-01825-w).

1. Palóczy, A. and LaCasce, J. H. (2022). Instability of a surface jet over rough topography. _Journal of Physical Oceanography_, **52(11)**, 2725-2740, doi:[10.1175/JPO-D-22-0079.1](https://doi.org/10.1175/JPO-D-22-0079.1).

1. Karrasch, D. and Schilling, N. (2020). Fast and robust computation of coherent Lagrangian vortices on very large two-dimensional domains. _The SMAI Journal of Computational Mathematics_, **6**, 101-124, doi:[10.5802/smai-jcm.63](https://doi.org/10.5802/smai-jcm.63).
