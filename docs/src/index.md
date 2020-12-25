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
    u(x) = \sum_{k_x} \hat{u}(k_x) \, e^{i k_x x} \ ,
    ```
    
    The convention used inside modules for the Fourier transforms of variable, e.g., `u` is 
    `uh` (where `h` at the end denotes "hat"). Note, however, that `uh` _is not_ exactly 
    ``\hat{u}`` due to different normalization factors that FFT algorithm uses. Instead,
    
    ```math
    \hat{u}(k_x) = \frac{𝚞𝚑}{n_x e^{- i k_x x_0}} ,
    ```
    
    where ``n_x`` is the total number of grid points in ``x`` and ``x_0`` is the left-most 
    point of our grid.
    
    Read more in the [FourierFlows.jl Documentation](https://fourierflows.github.io/FourierFlowsDocumentation/stable/grids/).


!!! info "Unicode"
    Oftentimes unicode symbols appear in modules for variables or parameters. For example,
    `ψ` is commonly used to denote the  streamfunction. Unicode symbols can be entered in 
    the Julia REPL by typing, e.g., `\psi` followed by `tab` key. Read more about Unicode 
    symbols in the [Julia Documentation](https://docs.julialang.org/en/v1/manual/unicode-input/).


## Developers

The development of GeophysicalFlows.jl started by [Navid C. Constantinou](http://www.navidconstantinou.com) and [Gregory L. Wagner](https://glwagner.github.io). During the course of time various people have contributed, including [Lia Siegelman](https://scholar.google.com/citations?user=BQJtj6sAAAAJ), [Brodie Pearson](https://brodiepearson.github.io), and [André Palóczy](https://scholar.google.com/citations?user=o4tYEH8AAAAJ) (see [example in FourierFlows.jl](https://fourierflows.github.io/FourierFlowsDocumentation/stable/generated/OneDShallowWaterGeostrophicAdjustment/)).

## Cite

The code is citable via [zenodo](https://doi.org/10.5281/zenodo.1463809).
