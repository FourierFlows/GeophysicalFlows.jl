push!(LOAD_PATH, "..")

using
  Documenter,
  Literate,
  Plots,  # to not capture precompilation output
  GeophysicalFlows

# Gotta set this environment variable when using the GR run-time on Travis CI.
# This happens as examples will use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

#####
##### Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

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
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
    Literate.notebook(example_filepath, OUTPUT_DIR, documenter=true)
    Literate.script(example_filepath, OUTPUT_DIR, documenter=true)
  end
end

#####
##### Build and deploy docs
#####

# Set up a timer to print a space ' ' every 240 seconds. This is to avoid Travis CI
# timing out when building demanding Literate.jl examples.
Timer(t -> println(" "), 0, interval=240)

format = Documenter.HTML(
  collapselevel = 2,
     prettyurls = get(ENV, "CI", nothing) == "true",
      canonical = "https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable/"
)

makedocs(
 modules = [GeophysicalFlows],
 doctest = false,
   clean = true,
checkdocs = :all,
  format = format,
 authors = "Navid C. Constantinou and Gregory L. Wagner",
sitename = "GeophysicalFlows.jl",
   pages = Any[
            "Home"    => "index.md",
            "Installation instructions" => "installation_instructions.md",
            "GPU" => "gpu.md",
            "Examples" => [
              "TwoDNavierStokes" => Any[
                "generated/twodnavierstokes_decaying.md",
                "generated/twodnavierstokes_stochasticforcing.md",
                "generated/twodnavierstokes_stochasticforcing_budgets.md",
                ],
              "SingleLayerQG" => Any[
                "generated/singlelayerqg_betadecay.md",
                "generated/singlelayerqg_betaforced.md",
                "generated/singlelayerqg_decaying_topography.md",
                "generated/singlelayerqg_decaying_barotropic_equivalentbarotropic.md"
                ],
              "BarotropicQGQL" => Any[
                "generated/barotropicqgql_betaforced.md",
                ],
              "MultiLayerQG" => Any[
                "generated/multilayerqg_2layer.md"
                ],
              "SurfaceQG" => Any[
                "generated/surfaceqg_decaying.md"
                ]
            ],
            "Modules" => Any[
              "modules/twodnavierstokes.md",
              "modules/singlelayerqg.md",
              "modules/barotropicqgql.md",
              "modules/multilayerqg.md",
              "modules/surfaceqg.md"
            ],
            "Stochastic Forcing" => "stochastic_forcing.md",
            "Contributor's guide" => "contributing.md",
            "Library" => Any[
            "lib/types.md",
            "lib/functions.md"
            ]
           ]
)

withenv("GITHUB_REPOSITORY" => "FourierFlows/GeophysicalFlowsDocumentation") do
  deploydocs(       repo = "github.com/FourierFlows/GeophysicalFlowsDocumentation.git",
                versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
            push_preview = false
            )
end
