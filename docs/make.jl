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
    "twodnavierstokes_stochasticforcing_budgets.jl",
    "barotropicqg_betadecay.jl",
    "barotropicqg_betaforced.jl",
    "barotropicqg_acc.jl",
    "barotropicqgql_betaforced.jl",
    "multilayerqg_2layer.jl",
]

for example in examples
  example_filepath = joinpath(EXAMPLES_DIR, example)
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
      canonical = "https://fourierflows.github.io/GeophysicalFlowsDocumentation/dev/"
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
            "Forcing" => "forcing.md",
            "Modules" => Any[
              "modules/twodnavierstokes.md",
              "modules/barotropicqg.md",
              "modules/barotropicqgql.md",
              "modules/multilayerqg.md"
            ],
            "Examples" => [ 
              "TwoDNavierStokes" => Any[
                "generated/twodnavierstokes_decaying.md",
                "generated/twodnavierstokes_stochasticforcing_budgets.md",
                ],
              "BarotropicQG" => Any[
                "generated/barotropicqg_betadecay.md",
                "generated/barotropicqg_betaforced.md",
                "generated/barotropicqg_acc.md",
                ],
              "BarotropicQGQL" => Any[
                "generated/barotropicqgql_betaforced.md",
                ],
              "MultilayerQG" => Any[
                "generated/multilayerqg_2layer.md"
                ]
            ],
            "DocStrings" => Any[
            "man/types.md",
            "man/functions.md"]
           ]
)

withenv("GITHUB_REPOSITORY" => "FourierFlows/GeophysicalFlowsDocumentation") do
  deploydocs(        repo = "github.com/FourierFlows/GeophysicalFlowsDocumentation.git",
                versions = ["stable" => "v^", "v#.#", "dev" => "dev"],
            push_preview = true
            )
end
