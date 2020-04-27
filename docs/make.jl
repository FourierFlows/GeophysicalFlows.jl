push!(LOAD_PATH, "..")

using
  Documenter,
  Literate,
  GeophysicalFlows,
  GeophysicalFlows.TwoDNavierStokes

#####
##### Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
    "twodnavierstokes_decaying.jl",
    "twodnavierstokes_stochasticforcing.jl",
    "barotropicqg_betadecay.jl",
    "barotropicqg_betaforced.jl",
    "barotropicqg_acc.jl",
    "barotropicqgql_betaforced.jl",
    "multilayerqg_2layer.jl",
]

for example in examples
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
end


#####
##### Build and deploy docs
#####

# Set up a timer to print a dot '.' every 60 seconds. This is to avoid Travis CI
# timing out when building demanding Literate.jl examples.
Timer(t -> println("."), 0, interval=60)

format = Documenter.HTML(
    collapselevel = 1,
       prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://fourierflows.github.io/GeophysicalFlows.jl/dev/"
)


makedocs(
 modules = [GeophysicalFlows],
 doctest = false,
   clean = true,
checkdocs = :all,
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
 authors = "Navid C. Constantinou and Gregory L. Wagner",
sitename = "GeophysicalFlows.jl",
   pages = Any[
            "Home"    => "index.md",
            "Modules" => Any[
              "modules/twodnavierstokes.md",
              "modules/barotropicqg.md",
              "modules/barotropicqgql.md",
              "modules/multilayerqg.md"
            ],
            "Examples" => [
              "generated/twodnavierstokes_decaying.md",
              "generated/twodnavierstokes_stochasticforcing.md",
              "generated/barotropicqg_betadecay.md",
              "generated/barotropicqg_betaforced.md",
              "generated/barotropicqg_acc.md",
              "generated/barotropicqgql_betaforced.md",
              "generated/multilayerqg_2layer.md"
            ],
            "DocStrings" => Any[
            "man/types.md",
            "man/functions.md"]
           ]
)

deploydocs(repo = "github.com/FourierFlows/GeophysicalFlows.jl.git")
