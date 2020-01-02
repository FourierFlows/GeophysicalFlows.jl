# GeophysicalFlows.jl

<p align="left">
    <a href="https://travis-ci.com/FourierFlows/GeophysicalFlows.jl">
        <img alt="Build Status for CPU" src="https://img.shields.io/travis/com/FourierFlows/GeophysicalFlows.jl/master?label=CPU&logo=travis&logoColor=white&style=flat-square">
    </a>
    <a href="https://gitlab.com/JuliaGPU/GeophysicalFlows-jl/commits/master">
      <img alt="Build Status for GPU" src="https://img.shields.io/gitlab/pipeline/JuliaGPU/GeophysicalFlows-jl/master?label=GPU&logo=gitlab&logoColor=white&style=flat-square">
    </a>
    <a href="https://ci.appveyor.com/project/navidcy/geophysicalflows-jl">
        <img alt="Build Status for Window" src="https://img.shields.io/appveyor/ci/navidcy/geophysicalflows-jl/master?label=Window&logo=appveyor&logoColor=white&style=flat-square">
    </a>
    <a href="https://fourierflows.github.io/GeophysicalFlows.jl/stable/">
        <img src="https://img.shields.io/badge/docs-stable-blue.svg">
    </a>
    <a href="https://fourierflows.github.io/GeophysicalFlows.jl/dev/">
        <img src="https://img.shields.io/badge/docs-dev-blue.svg">
    </a>
    <a href='https://coveralls.io/github/FourierFlows/GeophysicalFlows.jl?branch=master'><img src='https://coveralls.io/repos/github/FourierFlows/GeophysicalFlows.jl/badge.svg?branch=master' alt='Coverage Status' />
    </a>
    <a href="https://codecov.io/gh/FourierFlows/GeophysicalFlows.jl">
        <img src="https://codecov.io/gh/FourierFlows/GeophysicalFlows.jl/branch/master/graph/badge.svg" />
    </a>
    <a href="https://doi.org/10.5281/zenodo.1463809">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.1463809.svg" alt="DOI">
    </a>

</p>

This package leverages the [FourierFlows.jl] framework to provide modules for solving problems in
Geophysical Fluid Dynamics on periodic domains using Fourier-based pseudospectral methods.

## Installation

To install, do
```julia
] add GeophysicalFlows
```

See `examples/` for example scripts.

## Modules

All modules provide solvers on two-dimensional domains. We currently provide

* `TwoDTurb`: the two-dimensional vorticity equation.
* `BarotropicQG`: the barotropic quasi-geostrophic equation, which generalizes `TwoDTurb` to cases with topography and Coriolis parameters of the form `f = f₀ + βy`.
* `BarotropicQGQL`: the quasi-linear barotropic quasi-geostrophic equation.
* `MultilayerQG`: a multi-layer quasi-geostrophic model over topography and with ability to impose a zonal flow `U_n(y)` in each layer.



## Cite

The code is citable via [zenodo](https://zenodo.org). Please cite as:

> Navid C. Constantinou, & Gregory L. Wagner. (2019). FourierFlows/GeophysicalFlows.jl: GeophysicalFlows v0.3.0  (Version v0.3.0). Zenodo.  [http://doi.org/10.5281/zenodo.1463809](http://doi.org/10.5281/zenodo.1463809)


[FourierFlows.jl]: https://github.com/FourierFlows/FourierFlows.jl




