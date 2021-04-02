---
title: 'GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs & GPUs'
tags:
  - geophysical fluid dynamics
  - computational fluid dynamics
  - Fourier methods
  - pseudospectral
  - Julia
  - gpu
authors:
  - name: Navid C. Constantinou
    orcid: 0000-0002-8149-4094
    affiliation: "1, 2"
  - name: Gregory LeClaire Wagner
    orcid: 0000-0001-5317-2445
    affiliation: 3
  - name: Lia Siegelman
    orcid: 0000-0003-3330-082X
    affiliation: 4
  - name: Brodie C. Pearson
    orcid: 0000-0002-0202-0481
    affiliation: 5
  - name: André Palóczy
    orcid: 0000-0001-8231-8298
    affiliation: 4
affiliations:
 - name: Australian National University
   index: 1
 - name: ARC Centre of Excellence for Climate Extremes
   index: 2
 - name: Massachussetts Institute of Technology
   index: 3
 - name: University of California San Diego
   index: 4
 - name: Oregon State University
   index: 5
date: 30 March 2021
bibliography: paper.bib
---


# Summary

`GeophysicalFlows.jl` is a Julia package that contains partial differential equations solvers 
for a collection of geophysical fluid systems in periodic domains. All modules use Fourier-based 
pseudospectral numerical methods and leverage the framework provided by the `FourierFlows.jl` 
[@FourierFlows] Julia package for time-stepping, custom diagnostics, and saving output.


# Statement of need

Conceptual models in simple domains often provide stepping stones for better understanding geophysical and astrophysical systems, particularly the atmospheres and oceans of Earth and other planets. These conceptual models are used in research but also are of great value for helping students in class to grasp on new concepts and phenomena. Oftentimes people end up coding their own versions of solvers for the same partial differential equations for research or classwork. `GeophysicalFlows.jl` package is designed to be easily utilized and adaptable for a wide variety of both research and pedagogical purposes.

On top of the above-mentioned needs, the recent explosion of machine-learning applications in atmospheric and oceanic sciences advocates for the need that solvers for partial differential equations can be run on GPUs. 

`GeophysicalFlows.jl` provides a collection of modules for solving sets of partial differential equations often used as conceptual models. These modules are continuously tested (unit tests and tests for the physics involved) and are well-documented. `GeophysicalFlows.jl` utilizes Julia's functionality and abstraction to enable all modules to run on CPUs or GPUs, and to provide a high level of customizability within modules. The abstractions allow simulations to be tailored for specific research questions, via the choice of parameters, domain properties, and schemes for damping, forcing, time-stepping etc. Simulations can easily be carried out on different computing architectures. Selection of the architecture on which equations are solved is done by providing the argument `CPU()` or `GPU()` during the construction of a particular problem.
 
Documented examples for each geophysical system (module) appear in the package's documentation, 
providing a stepping stone for new users and for the development of new or customized modules. 
Current modules include two-dimensional flow and a variety of quasi-geostrophic (QG) dynamical 
systems, which provide analogues to the large-scale dynamics of atmospheres and oceans. The QG 
systems currently in `GeophysicalFlows.jl` extend two-dimensional dynamics to include the leading
order effects of a third dimension through planetary rotation, topography, surface boundary 
conditions, stratification and quasi-two-dimensional layering. A community-based collection 
of diagnostics throughout the modules are used to compute quantities like energy, enstrophy, 
dissipation, etc.

![Snapshots from a nonlinearly equilibrated simulation of the Eady instability over a
meridional ridge. Simulation used `MultiLayerQG` module of `GeophysicalFlows.jl`. The Eady 
problem was approximated here using 5 layers stacked up in the vertical. Each layer was 
simulated with 512² grid-points. Plots were made with the `Plots.jl` Julia package, 
which utilizes the `cmocean` colormaps collection [@Thyng2016]. Scripts to reproduce the 
simulation reside in the repository `github.com/FourierFlows/MultilayerQG-example`. 
\label{fig1}](PV_eady_nlayers5.png)


# State of the field

`GeophysicalFlows.jl` is a unique Julia package and shares similarities in functionality to 
the Python package `pyqg` [@pyqg]. Beyond their base language, the major differences between these 
packages are that `GeophysicalFlows.jl` can be run on GPUs or CPUs and leverages a separate 
package (`FourierFlows.jl`; which is continuously developed) to solve differential equations 
and compute diagnostics, while `pyqg` can only be run on CPUs and uses a self-contained kernel. 
Dedalus [@Burns2020] is a Python package with an intuitive script-based interface that uses spectral 
methods to solve general partial differential equations, such as the ones within `GeophysicalFlows.jl`. `Oceananigans.jl` [@Oceananigans] is an incompressible fluid solver that can be run in 1-D, 2-D, and 3-D domains. Both Oceananigans.jl and GeophysicalFlows.jl contain modules for two-dimensional flow, but the other examples provided with Oceananigans focus on three-dimensional flows which are generally present at smaller scales than the dynamics of GeophysicalFlows.jl's other modules. The `MAOOAM` [@MAOOAM] package, and its expanded Python implementation `qgs` [@qgs], simulate two atmospheric layers with QG dynamics, above either land or an oceanic layer with reduced-gravity QG dynamics. The dynamics of individual layers have overlap with the `MultiLayerQG` and `SingleLayerQG` modules, however the layer configuration of `MOAAM` and `qgs` is specifically designed to study the dynamics of Earth's mid-latitude atmosphere. Neither `MAOOAM` nor `qgs` can run on GPUs. There exist also some other isolated codes/scripts in personal websites and in open-source public repositories that have similar functionality as some `GeophysicalFlows.jl` modules. 

`GeophysicalFlows.jl` can be used to investigate a variety of scientific research questions 
thanks to its various modules and high customizability, and its ease-of-use makes it an ideal 
teaching tool for fluids courses [@GeophysicalFlows-Examples; @CLExWinterSchool2020]. 
`GeophysicalFlows.jl` has been used in developing Lagrangian vortices identification algorithms 
[@Karrasch2020]. Currently, `GeophysicalFlows.jl` is being used, e.g., (i) to test new theories 
for diagnosing turbulent energy transfers in geophysical flows [e.g. @Pearson2021], (ii) to compare 
different observational sampling techniques in these flows, (iii) to study the bifurcation properties 
Kolmogorov flows [@KolmogorovFlow], (iv) to study the genesis and persistence of the polygons 
of vortices present at Jovian high latitudes (Siegelman, Young, and Ingersoll; in prep)."


# Acknowledgements

We acknowledge discussions with Keaton Burns, Valentin Churavy, Cesar Rocha, and William Young. 
B. C. P. was supported by the National Science Foundation under Grant No. 2023721. We would 
also like to take a moment to remember Sean R. Haney (February 1987-January 2021) who left 
us a bit too early.


# References
