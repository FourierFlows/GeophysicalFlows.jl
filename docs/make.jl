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
const OUTPUT_DIR   = joinpath(@__DIR__, "src/examples")

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

# Set up a timer to print a space ' ' every 240 seconds. This is to avoid Travis CI
# timing out when building demanding Literate.jl examples.
Timer(t -> println(" "), 0, interval=240)

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
  format = format,
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
              "examples/twodnavierstokes_decaying.md",
              "examples/twodnavierstokes_stochasticforcing.md",
              "examples/barotropicqg_betadecay.md",
              "examples/barotropicqg_betaforced.md",
              "examples/barotropicqg_acc.md",
              "examples/barotropicqgql_betaforced.md",
              "examples/multilayerqg_2layer.md"
            ],
            "DocStrings" => Any[
            "man/types.md",
            "man/functions.md"]
           ]
)

deploydocs(        repo = "github.com/FourierFlows/GeophysicalFlows.jl.git",
                 devurl = "dev",
               versions = ["stable" => "v^", "v#.#", devurl => "dev"],
           push_preview = true,
           )
