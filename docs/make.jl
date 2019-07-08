# Workaround for JuliaLang/julia/pull/28625
if Base.HOME_PROJECT[] !== nothing
  Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end

using
  Documenter,
  GeophysicalFlows

makedocs(
 modules = [GeophysicalFlows],
 doctest = false,
   clean = true,
checkdocs = :all,
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
 authors = "Navid C. Constantinou and Gregory L. Wagner",
sitename = "GeophysicalFlows.jl",
   pages = Any[
            "Home" => "index.md",
            "Modules" => Any[
              "modules/twodturb.md",
              "modules/barotropicqg.md",
              "modules/multilayerqg.md"
            ],
            "DocStrings" => Any[
            "man/types.md",
            "man/functions.md"]
           ]
)

deploydocs(repo = "github.com/FourierFlows/GeophysicalFlows.jl.git")
