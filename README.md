# GeophysicalFlows.jl

<table>
    <tr align="center">
        <td><b>Documentation</b></td> <td>Travis</td> <td>Appveyor</td>
    </tr>
    <tr align="center">
        <td><a href="https://fourierflows.github.io/GeophysicalFlows.jl/latest/"><img src="https://img.shields.io/badge/docs-latest-blue.svg"></a><!--</br><a href="https://GeophysicalFlows.github.io/GeophysicalFlows.jl/stable/"><img src="https://img.shields.io/badge/docs-stable-blue.svg"></a>--></td> <td><a href="https://travis-ci.org/FourierFlows/GeophysicalFlows.jl"><img src="https://travis-ci.org/FourierFlows/GeophysicalFlows.jl.svg?branch=master" title="Build Status"></a><td><a href="https://ci.appveyor.com/project/navidcy/geophysicalflows-jl"><img src="https://ci.appveyor.com/api/projects/status/7c5f4wfckq5gb6qv?svg=true" title="Build Status"></a></td> 
    </tr>
 </table>

This package leverages the [FourierFlows.jl]() framework to provide modules for solving problems in
Geophysical Fluid Dynamics on periodic domains using Fourier-based pseudospectral methods.

## Installation

To install, do
```julia
using Pkg
Pkg.add("https://github.com/FourierFlows/GeophysicalFlows.jl.git")
```

See `examples/` for example scripts.

## Modules

All modules provide solvers on two-dimensional domains. We currently provide

* `TwoDTurb`: the two-dimensional vorticity equation.
* `BarotropicQG`: the barotropic quasi-geostrophic equation, which generalizes `TwoDTurb` to cases with topography and Coriolis parameters of the form `f = f₀ + βy`.
* `BarotropicQGQL`: the quasi-linear barotropic quasi-geostrophic equation.
* `NIWQG`: [a two-mode truncation]() of the [NIW-QG model]().


[FourierFlows.jl]: https://github.com/FourierFlows/FourierFlows.jl
[two-mode truncation]: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/stimulated-generation-extraction-of-energy-from-balanced-flow-by-nearinertial-waves/900227E2C12AA98ECEBBE64F4FF21C43
[XV15]: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/generalisedlagrangianmean-model-of-the-interactions-between-nearinertial-waves-and-mean-flow/C4FB1C5ABFBAC3A39B52DDC10F4C723F
[WY16]: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/threecomponent-model-for-the-coupled-evolution-of-nearinertial-waves-quasigeostrophic-flow-and-the-nearinertial-second-harmonic/4F2E61BDD531DEA02D24FBE9A2617DAB 
