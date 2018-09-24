using Documenter, GeophysicalFlows

makedocs(
   modules = [GeophysicalFlows],
   doctest = false, clean = true,
 checkdocs = :all,
    format = :html,
   authors = "Gregory L. Wagner and Navid C. Constantinou",
  sitename = "GeophysicalFlows.jl",
     pages = Any[
              "Home" => "index.md",
              "Modules" => Any[
                "modules/twodturb.md",
                "modules/barotropicqg.md",
                "modules/boussinesq.md"
              ],
              "DocStrings" => Any[
              "man/types.md",
              "man/functions.md"]
             ]
)

deploydocs(
       repo = "github.com/GeophysicalFlows/GeophysicalFlows.jl.git",
     target = "build",
      julia = "0.7",
     osname = "linux",
       deps = nothing,
       make = nothing
 )
