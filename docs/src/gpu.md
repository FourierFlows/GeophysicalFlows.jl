# GPU

GPU-functionality is enabled via `FourierFlows.jl`. For more information on how `FourierFlows.jl`
handled with GPUs we urge you to the corresponding [`FourierFlows.jl` documentation section ](https://fourierflows.github.io/FourierFlowsDocumentation/stable/gpu/)

All `GeophysicalFlows.jl` modules can be run on GPU by providing `GPU()` as the device (`dev`) 
argument in the problem constructors. For example,

```julia
julia> GeophysicalFlows.TwoDNavierStokes.Problem(GPU())
Problem
  ├─────────── grid: grid (on GPU)
  ├───── parameters: params
  ├────── variables: vars
  ├─── state vector: sol
  ├─────── equation: eqn
  ├────────── clock: clock
  └──── timestepper: RK4TimeStepper
```

## Selecting GPU device

`FourierFlows.jl` can only utilize a single GPU. If your machine has more than one GPU available, 
then functionality within `CUDA.jl` package enables the user to choose the GPU device that 
`FourierFlows.jl` should use. The user is referred to the [`CUDA.jl` Documentation](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#Device-Management); in particular, [`CUDA.devices`](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#CUDA.devices) and [`CUDA.CuDevice`](https://juliagpu.github.io/CUDA.jl/stable/lib/driver/#CUDA.CuDevice).
