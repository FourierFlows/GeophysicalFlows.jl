# GeophysicalFlows.jl Documentation

## Overview

`GeophysicalFlows.jl` is a collection of modules which leverage the 
[FourierFlows.jl](https://github.com/FourierFlows/FourierFlows.jl) framework to provide
solvers for problems in Geophysical Fluid Dynamics, on periodic domains using Fourier-based pseudospectral methods.

## Examples

Examples aim to demonstrate the main functionalities of each module. Have a look at our Examples collection!


!!! note "Fourier transforms normalization"
    
    Fourier methods are based on Fourier expansions. Throughout the documentation we denote
    symbols with overhat, e.g., ``\hat{u}``, to be the Fourier transform of ``u`` like, e.g.,
    ```math
    u(x, t) = \sum_{k_x} \hat{u}(k_x, t) \, \mathrm{e}^{\mathrm{i} k_x x} \ ,
    ```
    
    The convention used inside modules for the Fourier transforms of variable `u` is `uh` (with
    `h` denoting "hat"). Note, however, that `uh` _is not_ exactly ``\hat{u}`` but rather they 
    differ by a normalization factor. Read more in the [FourierFlows.jl Documentation](https://fourierflows.github.io/FourierFlowsDocumentation/stable/grids/).


!!! info "Unicode"
    Oftentimes we use unicode symbols for variables or parameters in the modules. For example,
    `ψ` for streamfunction. Unicode symbols can be entered in the REPL by typing, e.g., `\psi`
    followed by `tab` key. Read more in the [Julia Documentation](https://docs.julialang.org/en/v1/manual/unicode-input/).


## Developers

The development of GeophysicalFlows.jl started by [Navid C. Constantinou](http://www.navidconstantinou.com) and [Gregory L. Wagner](https://glwagner.github.io). During the course of time various people have contributed, including [Lia Siegelman](https://scholar.google.com/citations?user=BQJtj6sAAAAJ), [Brodie Pearson](https://brodiepearson.github.io), and [André Palóczy](https://scholar.google.com/citations?user=o4tYEH8AAAAJ) (see [example in FourierFlows.jl](https://fourierflows.github.io/FourierFlowsDocumentation/stable/generated/OneDShallowWaterGeostrophicAdjustment/)).

## Cite

The code is citable via [zenodo](https://doi.org/10.5281/zenodo.1463809).
