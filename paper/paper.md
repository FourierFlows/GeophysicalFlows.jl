---
title: 'GeophysicalFlows.jl: A Julia package for solving geophysical-fluid-dynamics problems on periodic domains on CPUs and GPUs'
tags:
  - julia
  - geophysical fluid dynamics
  - Fourier methods
  - gpu
authors:
  - name: Navid C. Constantinou
    orcid: 0000-0002-8149-4094
    affiliation: "1, 2"
  - name: Gregory LeClaire Wagner
    orcid: 0000-0001-5317-2445
    affiliation: 3
  - name: Brodie Pearson
    orcid: 0000-0002-0202-0481
    affiliation: 4
  - name: André Palóczy
    orcid: 0000-0001-8231-8298
    affiliation: 5
  - name: Lia Siegelman
    orcid: 0000-0003-3330-082X
    affiliation: 5
affiliations:
 - name: Australian National University
   index: 1
 - name: ARC Centre of Excellence for Climate Extremes
   index: 2
 - name: Massachussetts Institute of Technology
   index: 3
 - name: Oregon State University
   index: 4
 - name: University of California San Diego
   index: 5
date: 3 December 2020
bibliography: paper.bib
---

# Summary

GeophysicalFlows.jl is a julia package and it is very nice...

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
-->

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

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


# Figures

Figures can be included like this:

![Caption for example figure.\label{fig:example}](figure.png)

and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:
```julia
]add GeophysicalFlows
```	

# Acknowledgements

We acknowledge fruitful discussions with Cesar B. Rocha and Keaton Burns.

# References
