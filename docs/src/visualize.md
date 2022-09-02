# Visualize output

In the examples we use [Makie.jl](https://makie.juliaplots.org/stable/) to plot and animate flow fields.

Makie comes with few backends. The examples use [CairoMakie](https://makie.juliaplots.org/stable/documentation/backends/cairomakie/)
since this is backend works well on headless devices, that is, devices without monitor. The docs are automatically
build via GitHub actions thus CairoMakie backend is necessary. Users that run GeophysicalFlows.jl on
devices with a monitor they might find useful to change to [GLMakie](https://makie.juliaplots.org/stable/documentation/backends/glmakie/).

We can either visualize the flow on-the-fly as the problems is stepped forward or we can save output onto `.jld2`
and after simulation is done load the output and visualize it. Most examples use the former strategy. For a demonstration
for how one can save output and load later to process/visualize look at the [`singlelayerqg_betaforced.jl`](@ref singlelayerqg_betaforced_example)
example. Furthermore, the [Output section](https://fourierflows.github.io/FourierFlowsDocumentation/stable/output/) in FourierFlows.jl
Documentation might be useful.
