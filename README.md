# GeophysicalFlows.jl

<p align="left">
    <a href="https://buildkite.com/julialang/geophysicalflows-dot-jl">
        <img alt="Buildkite CPU+GPU build status" src="https://img.shields.io/buildkite/4d921fc17b95341ea5477fb62df0e6d9364b61b154e050a123/master?logo=buildkite&label=Buildkite%20CPU%2BGPU">
    </a>
    <a href="https://ci.appveyor.com/project/navidcy/geophysicalflows-jl">
        <img alt="Build Status for Window" src="https://img.shields.io/appveyor/ci/navidcy/geophysicalflows-jl/master?label=Window&logo=appveyor&logoColor=white&style=flat-square">
    </a>
    <a href="https://FourierFlows.github.io/GeophysicalFlowsDocumentation/stable">
        <img alt="stable docs" src="https://img.shields.io/badge/documentation-stable%20release-blue">
    </a>
    <a href="https://FourierFlows.github.io/GeophysicalFlowsDocumentation/dev">
        <img alt="latest docs" src="https://img.shields.io/badge/documentation-in%20development-orange">
    </a>
    <a href="https://codecov.io/gh/FourierFlows/GeophysicalFlows.jl">
        <img src="https://codecov.io/gh/FourierFlows/GeophysicalFlows.jl/branch/master/graph/badge.svg" />
    </a>
    <a href="https://doi.org/10.5281/zenodo.1463809">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.1463809.svg" alt="DOI">
    </a>
    <a href="https://joss.theoj.org/papers/a8cdf26beae8bcecc751ab4ded53b308">
        <img src="https://joss.theoj.org/papers/a8cdf26beae8bcecc751ab4ded53b308/status.svg">
    </a>
</p>

This package leverages the [FourierFlows.jl] framework to provide modules for solving problems in
Geophysical Fluid Dynamics on periodic domains using Fourier-based pseudospectral methods.

## Installation

To install, do

```julia
] add GeophysicalFlows
```

## Examples

See `examples/` for example scripts. These examples are best viewed by browsing them within the package's [documentation]. 

Some animations created with GeophysicalFlows.jl are [online @ youtube].


## Modules

* `TwoDNavierStokes`: the two-dimensional vorticity equation.
* `SingleLayerQG`: the barotropic or equivalent-barotropic quasi-geostrophic equation, which generalizes `TwoDNavierStokes` to cases with topography, Coriolis parameters of the form `f = f₀ + βy`, and finite Rossby radius of deformation.
* `MultiLayerQG`: a multi-layer quasi-geostrophic model over topography and with the ability to impose a zonal flow `U_n(y)` in each layer.
* `SurfaceQG`: a surface quasi-geostrophic model.
* `BarotropicQGQL`: the quasi-linear barotropic quasi-geostrophic equation.


## Cite

The code is citable via [zenodo](https://zenodo.org). Please cite as:

> Navid C. Constantinou, Gregory L. Wagner, and co-contributors. (2021). FourierFlows/GeophysicalFlows.jl: GeophysicalFlows v0.11.3  (Version v0.11.3). Zenodo.  [http://doi.org/10.5281/zenodo.1463809](http://doi.org/10.5281/zenodo.1463809)


[FourierFlows.jl]: https://github.com/FourierFlows/FourierFlows.jl
[documentation]: https://fourierflows.github.io/GeophysicalFlowsDocumentation/dev/
[online @ youtube]: https://www.youtube.com/channel/UCO_0ugkNUwCsFUMtepwYTqw
