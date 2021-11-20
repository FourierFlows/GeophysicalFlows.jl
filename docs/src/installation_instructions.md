# Installation instructions

You can install the latest version of GeophysicalFlows.jl via the built-in package manager 
(accessed by pressing `]` in the Julia REPL command prompt) to add the package and also to 
instantiate/build all the required dependencies

```julia
julia>]
(v1.5) pkg> add GeophysicalFlows
(v1.5) pkg> instantiate
```

We recommend installing GeophysicalFlows.jl with the built-in Julia package manager, because 
this installs a stable, tagged release. Later on, you can update GeophysicalFlows.jl to the 
latest tagged release again via the package manager by typing

```julia
(v1.5) pkg> update GeophysicalFlows
```

Note that some releases might induce breaking changes to certain modules. If after anything 
happens or your code stops working, please open an [issue](https://github.com/FourierFlows/GeophysicalFlows.jl/issues) 
or start a [discussion](https://github.com/FourierFlows/GeophysicalFlows.jl/discussions). We're 
more than happy to help with getting your simulations up and running.

!!! warn "Use Julia 1.5 or newer"
    The latest GeophysicalFlows.jl requires at least Julia v1.5 to run.
    Installing GeophysicalFlows with an older version of Julia will install an older version 
    of GeophysicalFlows.jl (the latest version compatible with your version of Julia).
    
    Last version compatible with Julia v1.0.5 (the current long-term-release): GeophysicalFlows.jl v0.5.1


