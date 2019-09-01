# GeophysicalFlows.jl

<p align="left">
    <a href="https://travis-ci.org/FourierFlows/GeophysicalFlows.jl">
        <img src="https://travis-ci.org/FourierFlows/GeophysicalFlows.jl.svg?branch=master" title="Build Status">
    </a>
    <a href="https://ci.appveyor.com/project/navidcy/geophysicalflows-jl">
        <img src="https://ci.appveyor.com/api/projects/status/7c5f4wfckq5gb6qv?svg=true" title="Build Status">
    </a>
    <a href="https://fourierflows.github.io/GeophysicalFlows.jl/stable/">
        <img src="https://img.shields.io/badge/docs-stable-blue.svg">
    </a>
    <a href="https://fourierflows.github.io/GeophysicalFlows.jl/latest/">
        <img src="https://img.shields.io/badge/docs-latest-blue.svg">
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




