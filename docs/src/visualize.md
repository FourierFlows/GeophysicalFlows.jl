# Visualize output

In the examples we use [Makie.jl](https://makie.juliaplots.org/stable/) for plotting.

Makie comes with a few [backends](https://makie.juliaplots.org/stable/#makie_ecosystem). In the documented examples
we use [CairoMakie](https://makie.juliaplots.org/stable/documentation/backends/cairomakie/) since this backend
works well on headless devices, that is, devices without monitor. Since the documentation is automatically
built via GitHub actions the CairoMakie backend is necessary. Users that run GeophysicalFlows.jl on
devices with a monitor might want to change to [GLMakie](https://makie.juliaplots.org/stable/documentation/backends/glmakie/)
that display figures in an interactive window.

In GeophysicalFlows.jl simulations, we can either visualize the flow on-the-fly as the problem is stepped forward or
we can save output onto `.jld2` and after simulation is done load the output and visualize it. Most examples use the
former strategy. For a demonstration for how one can save output and load later to process/visualize look at the
[`SingeLayerQG` beta-plane forced-dissipative example](@ref singlelayerqg_betaforced_example). Furthermore, the
[Output section](https://fourierflows.github.io/FourierFlowsDocumentation/stable/output/) in FourierFlows.jl Documentation
might be useful.
