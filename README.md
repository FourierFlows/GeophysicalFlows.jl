# GeophysicalFlows.jl

<!-- description -->
<p>
  <strong>💨🌏🌊 Geophysical fluid dynamics pseudospectral solvers with Julia and <a href="http://github.com/FourierFlows/FourierFlows.jl">FourierFlows.jl</a>. https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable</strong>
</p>

<!-- Badges -->
<p align="left">
    <a href="https://buildkite.com/julialang/geophysicalflows-dot-jl">
        <img alt="Buildkite CPU+GPU build status" src="https://img.shields.io/buildkite/4d921fc17b95341ea5477fb62df0e6d9364b61b154e050a123/main?logo=buildkite&label=Buildkite%20CPU%2BGPU">
    </a>
    <a href="https://github.com/FourierFlows/GeophysicalFlows.jl/actions/workflows/CI.yml">
        <img alt="CI Status" src="https://github.com/FourierFlows/GeophysicalFlows.jl/actions/workflows/CI.yml/badge.svg">
    </a>
    <a href="https://FourierFlows.github.io/GeophysicalFlowsDocumentation/stable">
        <img alt="stable docs" src="https://img.shields.io/badge/documentation-stable%20release-blue">
    </a>
    <a href="https://FourierFlows.github.io/GeophysicalFlowsDocumentation/dev">
        <img alt="latest docs" src="https://img.shields.io/badge/documentation-in%20development-orange">
    </a>
    <a href="https://codecov.io/gh/FourierFlows/GeophysicalFlows.jl">
        <img src="https://codecov.io/gh/FourierFlows/GeophysicalFlows.jl/branch/main/graph/badge.svg" />
    </a>
    <a href="https://doi.org/10.5281/zenodo.1463809">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.1463809.svg" alt="DOI">
    </a>
    <a href="https://github.com/SciML/ColPrac">
      <img alt="ColPrac: Contributor's Guide on Collaborative Practices for Community Packages" src="https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet">
    </a>
    <a href="https://doi.org/10.21105/joss.03053">
      <img src="https://joss.theoj.org/papers/10.21105/joss.03053/status.svg" alt="DOI badge" >
    </a>
</p>

This package leverages the [FourierFlows.jl] framework to provide modules for solving problems in
Geophysical Fluid Dynamics on periodic domains using Fourier-based pseudospectral methods.


## Installation

To install, use Julia's  built-in package manager (accessed by pressing `]` in the Julia REPL command prompt) to add the package and also to instantiate/build all the required dependencies

```julia
julia> ]
(v1.10) pkg> add GeophysicalFlows
(v1.10) pkg> instantiate
```

GeophysicalFlows.jl requires Julia v1.6 or later. However, the package has continuous integration testing on
Julia v1.10 (the current long-term release), v1.11, and v1.12. _We strongly urge you to use one of these Julia versions._

## Examples

See `examples/` for example scripts. These examples are best viewed by browsing them within 
the package's [documentation]. 

Some animations created with GeophysicalFlows.jl are [online @ youtube].


## Modules

* [`TwoDNavierStokes`](https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/modules/twodnavierstokes/): the
  two-dimensional vorticity equation.
* [`SingleLayerQG`](https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/modules/singlelayerqg/): the barotropic
  or equivalent-barotropic quasi-geostrophic equation, which generalizes `TwoDNavierStokes` to cases with topography, Coriolis
  parameters of the form `f = f₀ + βy`, and finite Rossby radius of deformation.
* [`MultiLayerQG`](https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/modules/multilayerqg/): a multi-layer
  quasi-geostrophic model over topography allowing to impose a zonal flow `U_n(y)` in each layer.
* [`SurfaceQG`](https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/modules/surfaceqg/): a surface
  quasi-geostrophic model.
* [`BarotropicQGQL`](https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/modules/barotropicqgql/): the
  quasi-linear barotropic quasi-geostrophic equation.


## Scalability

For now, GeophysicalFlows.jl is restricted to run on either a single CPU or single GPU. These
restrictions come from FourierFlows.jl. Multi-threading can enhance performance for the Fourier
transforms. By default, FourierFlows.jl will use the maximum number of threads available on 
your machine. You can set the number of threads used by FourierFlows.jl by setting the 
environment variable, e.g.,

```
$ export JULIA_NUM_THREADS=4
```

For more information on multi-threading users are directed to the [Julia Documentation](https://docs.julialang.org/en/v1/manual/multi-threading/).

If your machine has more than one GPU available, then functionality within CUDA.jl package 
enables the user to choose the GPU device that FourierFlows.jl should use. The user is referred
to the [CUDA.jl Documentation](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#Device-Management);
in particular, [`CUDA.devices`](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#CUDA.devices) 
and [`CUDA.CuDevice`](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#CUDA.CuDevice). 
The user is also referred to the [GPU section](https://fourierflows.github.io/FourierFlowsDocumentation/stable/gpu/) in the FourierFlows.jl documentation.


## Getting help

Interested in GeophysicalFlows.jl or trying to figure out how to use it? Please feel free to 
ask us questions and get in touch! Check out the 
[examples](https://github.com/FourierFlows/GeophysicalFlows.jl/tree/main/examples) and 
[open an issue](https://github.com/FourierFlows/GeophysicalFlows.jl/issues/new) or 
[start a discussion](https://github.com/FourierFlows/GeophysicalFlows.jl/discussions/new) 
if you have any questions, comments, suggestions, etc.


## Citing

If you use GeophysicalFlows.jl in research, teaching, or other activities, we would be grateful 
if you could mention GeophysicalFlows.jl and cite our paper in JOSS:

> Constantinou et al., (2021). GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs & GPUs. _Journal of Open Source Software_, **6(60)**, 3053, doi:[10.21105/joss.03053](https://doi.org/10.21105/joss.03053).

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
  title = {{GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs \& GPUs}},
  journal = {Journal of Open Source Software}
}
```


## Contributing

If you're interested in contributing to the development of GeophysicalFlows.jl we are excited 
to get your help, no matter how big or small a contribution you make! It's always great to have 
new people look at the code with fresh eyes: you will see errors that other developers have missed.

Let us know by [open an issue](https://github.com/FourierFlows/GeophysicalFlows.jl/issues/new) 
or [start a discussion](https://github.com/FourierFlows/GeophysicalFlows.jl/discussions/new) 
if you'd like to work on a new feature or implement a new module, if you're new to open-source 
and want to find a cool little project or issue to work on that fits your interests! We're more 
than happy to help along the way.

For more information, check out our [contributors' guide](https://github.com/FourierFlows/GeophysicalFlows.jl/blob/main/CONTRIBUTING.md).

[FourierFlows.jl]: https://github.com/FourierFlows/FourierFlows.jl
[documentation]: https://fourierflows.github.io/GeophysicalFlowsDocumentation/dev/
[online @ youtube]: https://www.youtube.com/channel/UCO_0ugkNUwCsFUMtepwYTqw
