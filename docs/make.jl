using Documenter, DocumenterCitations, Literate

using CairoMakie
# CairoMakie.activate!(type = "svg")

using GeophysicalFlows

#####
##### Generate literated examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

examples = [
  "twodnavierstokes_decaying.jl",
  "twodnavierstokes_stochasticforcing.jl",
  "twodnavierstokes_stochasticforcing_budgets.jl",
  "singlelayerqg_betadecay.jl",
  "singlelayerqg_betaforced.jl",
  "singlelayerqg_decaying_topography.jl",
  "singlelayerqg_decaying_barotropic_equivalentbarotropic.jl",
  "barotropicqgql_betaforced.jl",
  "multilayerqg_2layer.jl",
  "surfaceqg_decaying.jl",
]


for example in examples
  withenv("GITHUB_REPOSITORY" => "FourierFlows/GeophysicalFlowsDocumentation") do
    example_filepath = joinpath(EXAMPLES_DIR, example)
    withenv("JULIA_DEBUG" => "Literate") do
      Literate.markdown(example_filepath, OUTPUT_DIR;
                        flavor = Literate.DocumenterFlavor(), execute = true)
    end
  end
end


#####
##### Build and deploy docs
#####

format = Documenter.HTML(
  collapselevel = 2,
     prettyurls = get(ENV, "CI", nothing) == "true",
      canonical = "https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/"
)

bib_filepath = joinpath(dirname(@__FILE__), "src/references.bib")
bib = CitationBibliography(bib_filepath, style=:authoryear)

makedocs(
  authors = "Navid C. Constantinou, Gregory L. Wagner, and contributors",
 sitename = "GeophysicalFlows.jl",
  modules = [GeophysicalFlows],
  plugins = [bib],
   format = format,
  doctest = true,
    clean = true,
checkdocs = :all,
    pages = Any[
                "Home" => "index.md",
                "Installation instructions" => "installation_instructions.md",
                "Aliasing" => "aliasing.md",
                "GPU" => "gpu.md",
                "Visualize output" => "visualize.md",
                "Examples" => [
                  "TwoDNavierStokes" => Any[
                    "literated/twodnavierstokes_decaying.md",
                    "literated/twodnavierstokes_stochasticforcing.md",
                    "literated/twodnavierstokes_stochasticforcing_budgets.md",
                    ],
                  "SingleLayerQG" => Any[
                    "literated/singlelayerqg_betadecay.md",
                    "literated/singlelayerqg_betaforced.md",
                    "literated/singlelayerqg_decaying_topography.md",
                    "literated/singlelayerqg_decaying_barotropic_equivalentbarotropic.md"
                    ],
                  "BarotropicQGQL" => Any[
                    "literated/barotropicqgql_betaforced.md",
                    ],
                  "MultiLayerQG" => Any[
                    "literated/multilayerqg_2layer.md"
                    ],
                  "SurfaceQG" => Any[
                    "literated/surfaceqg_decaying.md"
                    ]
                  ],
                "Modules" => Any[
                  "modules/twodnavierstokes.md",
                  "modules/singlelayerqg.md",
                  "modules/barotropicqgql.md",
                  "modules/multilayerqg.md",
                  "modules/surfaceqg.md"
                  ],
                "Stochastic forcing" => "stochastic_forcing.md",
                "Contributor's guide" => "contributing.md",
                "References" => "references.md",
                "Library" => Any[
                  "lib/types.md",
                  "lib/functions.md"
                ]
              ]
)

@info "Clean up temporary .jld2 and .nc output created by doctests or literated examples..."

"""
    recursive_find(directory, pattern)

Return list of filepaths within `directory` that contains the `pattern::Regex`.
"""
recursive_find(directory, pattern) =
    mapreduce(vcat, walkdir(directory)) do (root, dirs, files)
        joinpath.(root, filter(contains(pattern), files))
    end

files = []
for pattern in [r"\.jld2", r"\.nc"]
    global files = vcat(files, recursive_find(@__DIR__, pattern))
end

for file in files
    rm(file)
end

withenv("GITHUB_REPOSITORY" => "FourierFlows/GeophysicalFlowsDocumentation") do
  deploydocs(       repo = "github.com/FourierFlows/GeophysicalFlowsDocumentation.git",
                versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
            push_preview = false,
               forcepush = true,
               devbranch = "main"
            )
end
