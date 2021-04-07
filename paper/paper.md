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
    affiliation: 6
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
 - name: University of Oslo
   index: 6
date: 7 April 2021
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
providing a starting point for new users and for the development of new or customized modules. 
Current modules include two-dimensional flow and a variety of quasi-geostrophic (QG) dynamical 
systems, which provide analogues to the large-scale dynamics of atmospheres and oceans. The QG 
systems currently in `GeophysicalFlows.jl` extend two-dimensional dynamics to include the leading
order effects of a third dimension through planetary rotation, topography, surface boundary 
conditions, stratification and quasi-two-dimensional layering. A community-based collection 
of diagnostics throughout the modules are used to compute quantities like energy, enstrophy, 
dissipation, etc.

![Potential vorticity snapshots from a nonlinearly equilibrated simulation of the Eady instability 
over a meridional ridge. Simulation used `MultiLayerQG` module of `GeophysicalFlows.jl`. The Eady 
problem was approximated here using 5 layers stacked up in the vertical. Each layer was simulated 
with 512² grid-points. Plots were made with the `Plots.jl` Julia package, which utilizes the 
`cmocean` colormaps collection [@Thyng2016]. Scripts to reproduce the simulation reside in the 
repository `github.com/FourierFlows/MultilayerQG-example`. \label{fig1}](PV_eady_nlayers5.png)


# State of the field

`GeophysicalFlows.jl` is a unique Julia package that shares some features and similarities with 
other packages. In particular:

- `pyqg` [@pyqg] (Python)

  Beyond their base language, the major differences between `GeophysicalFlows.jl` and `pyqg` 
  is that `GeophysicalFlows.jl` can be run on GPUs or CPUs and leverages a separate package (`FourierFlows.jl`; which is continuously developed) to solve differential equations and compute diagnostics, while `pyqg` can only be run on CPUs and uses a self-contained kernel. 
  
- Dedalus [@Burns2020] (Python)
  
  Dedalus is a Python package with an intuitive script-based interface that uses spectral methods 
  to solve general partial differential equations, such as the ones within `GeophysicalFlows.jl`.
  Dedalus allows for more general boundary conditions in one of the dimensions. It only runs on 
  CPUs (not on GPUs) but can be MPI-parallelized.
  
- `Oceananigans.jl` [@Oceananigans] (Julia)
  
  `Oceananigans.jl` is a fluid solver focussed on the Navier-Stokes equations under the Boussinesq
  approximation. `Oceananigans.jl` also runs on GPUs, and it allows for more variety of boundary
  conditions but it does not have spectral accuracy as it uses finite-volume discretization methods.
  
- `MAOOAM` [@MAOOAM] (Fortran, Python, and Lua) and its expanded Python implementation `qgs` [@qgs]

  `MAOOAM` and `qgs` simulate two atmospheric layers with QG dynamics, above either land or 
  an oceanic fluid layer with reduced-gravity QG dynamics. The dynamics of individual layers 
  have overlap with the `MultiLayerQG` and `SingleLayerQG` modules, however the layer configuration 
  of `MOAAM` and `qgs` is specifically designed to study the dynamics of Earth's mid-latitude 
  atmosphere. Neither `MAOOAM` nor `qgs` can run on GPUs.
  
- Isolated codes/scripts 

  Several codes/scripts exist in personal websites and in open-source public repositories with
  similar functionality as some `GeophysicalFlows.jl` modules (e.g., `TwoDNavierStokes` or 
  `SingleLayerQG`). Usually, though, these codes come without any or poor documentation and 
  typically they are not continuously tested.

`GeophysicalFlows.jl` can be used to investigate a variety of scientific research questions 
thanks to its various modules and high customizability, and its ease-of-use makes it an ideal 
teaching tool for fluids courses [@GeophysicalFlows-Examples; @CLExWinterSchool2020]. 
`GeophysicalFlows.jl` has been used in developing Lagrangian vortices identification algorithms 
[@Karrasch2020]. Currently, `GeophysicalFlows.jl` is being used, e.g., (i) to test new theories 
for diagnosing turbulent energy transfers in geophysical flows [e.g. @Pearson2021], (ii) to compare 
different observational sampling techniques in these flows, (iii) to study the bifurcation properties 
of Kolmogorov flows [@KolmogorovFlow], (iv) to study the genesis and persistence of the polygons 
of vortices present at Jovian high latitudes (Siegelman, Young, and Ingersoll; in prep), (v) to study how mesoscale macroturbulence affects mixing of tracers [@QG_tracer_advection].


# Acknowledgements

We acknowledge discussions with Keaton Burns, Valentin Churavy, Theodore Drivas, Cesar Rocha, 
and William Young. B. C. P. was supported by the National Science Foundation under Grant 
No. 2023721. We would also like to take a moment to remember our friend and colleague 
Sean R. Haney (February 1987 - January 2021) who left us a bit too early.


# References
