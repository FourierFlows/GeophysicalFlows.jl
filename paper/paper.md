---
title: 'GeophysicalFlows.jl: Solvers for geophysical fluid dynamics-problems in periodic domains on CPUs and GPUs'
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
date: 25 January 2021
bibliography: paper.bib
---

<!-- 
# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

-->

# Summary

`GeophysicalFlows.jl` is a Julia package that contains partial differential solvers for a collection 
of geophysical fluid systems on periodic domains. All modules use Fourier-based pseudospectral 
numerical methods and leverage the framework provided by the `FourierFlows.jl` Julia package 
for time-stepping, diagnostics, and output.

`GeophysicalFlows.jl` utilizes Julia's functionality and abstraction to enable all modules to
run on CPUs or GPUs, and to provide a high level of customizability within modules. This allows 
simulations to be tailored for specific research questions, via the choice of parameters, domain 
properties, and schemes for damping, forcing, time-stepping etc. Simulations can easily be carried 
out on different computing architectures, selection of the architecture on which equations are solved 
is done by providing the argument `CPU()` or `GPU()` during the construction of a particular problem.

Documented examples for each geophysical system (module) appear in the package's documentation, 
providing a stepping stone for new users and for the development of new or customized modules. 
Current modules include two-dimensional (2D) flow and a variety of quasi-geostrophic (QG) dynamical 
systems, which provide analogues to the large-scale dynamics of atmospheres and oceans. The QG 
systems currently in `GeophysicalFlows.jl` extend 2D dynamics to include the leading order effects 
of a third dimension through planetary rotation, bathymetry/topography, surface boundary conditions, 
stratification and quasi-2D layering. A community-based collection of diagnostics throughout 
the modules are used to compute quantities like energy, enstrophy, dissipation, etc.

![Snapshots from a nonlinearly equilibrated simulation of the Eady instability over a
meridional ridge. Simulation used `MultiLayerQG` module of `GeophysicalFlows.jl`. The Eady 
problem was approximated here using 5 layers stacked up in the vertical. Each layer was 
simulated with 512² grid-points. Plots were made with the `Plots.jl` Julia package, 
which utilizes the `cmocean` colormap collection [@Thyng2016]. Scripts to reproduce the 
simulation are found in the repository `github.com/FourierFlows/MultilayerQG-example`. 
\label{fig1}](PV_eady_nlayers5.png)

`GeophysicalFlows.jl` is a unique Julia package and has similar functionality to the Python 
`pyqg` [@pyqg]. Beyond their base language, the major differences between these packages are
that `GeophysicalFlows.jl` can be run on GPUs or CPUs and leverages a separate package (`FourierFlows.jl`; 
which is continuously developed) to solve differential equations and compute diagnostics, 
while `pyqg` can only be run on CPUs and uses a self-contained kernel. Dedalus [@Burns2020] 
is Python software with an intuitive script-based interface that uses spectral methods to solve
general partial differential equations, such as the ones within `GeophysicalFlows.jl`. There 
are also some other isolated codes/scripts on personal websites and in open-source public 
repositories that have similar functionality as some `GeophysicalFlows.jl` modules. 

`GeophysicalFlows.jl` can be used to investigate a variety of scientific research questions 
thanks to its various modules and high customizability, and its ease-of-use makes it an ideal 
teaching tool for fluids courses. `GeophysicalFlows.jl` has been used in developing Lagrangian 
vortices identification algorithms [@Karrasch2020]. Currently, `GeophysicalFlows.jl` is being 
used (i) to test new theories for diagnosing turbulent energy transfers in geophysical flows 
[e.g. @Pearson2021], (ii) to compare different observational sampling techniques in these  flows, 
(iii) to study the bifurcation properties Kologorov flows [@KolmogorovFlow], (and others? Lia, André?)...


# Acknowledgements

We acknowledge discussions with Keaton Burns, Cesar Rocha, and William Young. We would also 
like to take a moment to remember Sean R. Haney (February 1987-January 2021) who left us a 
bit too early.

# References
