# Aliasing


In pseudospectral methods, computing nonlinear terms results in aliasing errors. (Read more about
aliasing errors in the [FourierFlows.jl Documentation](https://fourierflows.github.io/FourierFlowsDocumentation/stable/aliasing/).) To avoid aliasing errors, we need to apply some dealiasing to our fields 
in Fourier space before transforming to physical space to compute nonlinear terms.

!!! info "De-aliasing scheme"
    FourierFlows.jl curently implements dealiasing by zeroing out the highest-`aliased_fraction` 
    wavenumber components on a `grid`. By default in FourierFlows.jl, `aliased_fraction=1/3`.
    Users can construct a `grid` with different `aliased_fraction` via
    
    ```julia
    julia> grid = OneDGrid(64, 2π, aliased_fraction=1/2)
    
    julia> OneDimensionalGrid
             ├─────────── Device: CPU
             ├──────── FloatType: Float64
             ├────────── size Lx: 6.283185307179586
             ├──── resolution nx: 64
             ├── grid spacing dx: 0.09817477042468103
             ├─────────── domain: x ∈ [-3.141592653589793, 3.0434178831651124]
             └─ aliased fraction: 0.5
    ```
    or provide the keyword argument `aliased_fraction` to the `Problem()` constructor of each
    module, e.g.,
    
    ```julia
    julia> prob = GeophysicalFlows.TwoDNavierStokes.Problem(; aliased_fraction=1/2)
    Problem
      ├─────────── grid: grid (on CPU)
      ├───── parameters: params
      ├────── variables: vars
      ├─── state vector: sol
      ├─────── equation: eqn
      ├────────── clock: clock
      └──── timestepper: RK4TimeStepper
      
    julia> prob.grid.aliased_fraction
    0.5
    ```

Currently, all nonlinearities in all modules included in GeophysicalFlows.jl modules are quadratic 
nonlinearities. Therefore, the default `aliased_fraction` of 1/3 is appropriate.

All modules apply de-aliasing by calling, e.g., `dealias!(prob.sol, prob.grid)` both before
computing any nonlinear terms and also during updating all variable, i.e., within `updatevars!`.

To disable de-aliasing you need to create a problem with a grid that has been constructed with 
the keyword `aliased_fraction=0`.
