# GPU

GPU functionality is enabled via `FourierFlows.jl`. For more information on how `FourierFlows.jl`
handles GPUs, see the corresponding [`FourierFlows.jl` documentation section](https://fourierflows.github.io/FourierFlowsDocumentation/stable/gpu/).

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
you can use functionality from the `CUDA.jl` package to choose the GPU device that
`FourierFlows.jl` should use. See the [`CUDA.jl` Documentation](https://cuda.juliagpu.org/stable/lib/cudadrv/#Device-Management); in particular, [`CUDA.devices`](https://cuda.juliagpu.org/stable/lib/cudadrv/#CUDACore.devices) and [`CUDA.CuDevice`](https://cuda.juliagpu.org/stable/lib/cudadrv/#CUDACore.CuDevice).
