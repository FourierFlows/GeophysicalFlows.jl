using
  Documenter,
  Literate,
  CairoMakie,   # to not capture precompilation output
  GeophysicalFlows,
  Glob

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
    Literate.markdown(example_filepath, OUTPUT_DIR; flavor = Literate.DocumenterFlavor())
    Literate.notebook(example_filepath, OUTPUT_DIR)
    Literate.script(example_filepath, OUTPUT_DIR)
  end
end

#####
##### Build and deploy docs
#####

# Set up a timer to print a space ' ' every 240 seconds. This is to avoid CI machines
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
            "Aliasing" => "aliasing.md",
            "GPU" => "gpu.md",
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
                versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
            push_preview = false,
               devbranch = "main"
            )
end

@info "Cleaning up temporary .jld2 and .nc files created by doctests..."

for file in vcat(glob("docs/*.jld2"), glob("docs/*.nc"))
    rm(file)
end
