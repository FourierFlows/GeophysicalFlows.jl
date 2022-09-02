# Functions

## `GeophysicalFlows`

### Exported functions

```@docs
GeophysicalFlows.lambdipole
GeophysicalFlows.peakedisotropicspectrum
```


## `TwoDNavierStokes`

### Exported functions

```@docs
GeophysicalFlows.TwoDNavierStokes.Problem
GeophysicalFlows.TwoDNavierStokes.set_ζ!
GeophysicalFlows.TwoDNavierStokes.energy
GeophysicalFlows.TwoDNavierStokes.energy_dissipation_hyperviscosity
GeophysicalFlows.TwoDNavierStokes.energy_dissipation_hypoviscosity
GeophysicalFlows.TwoDNavierStokes.energy_work
GeophysicalFlows.TwoDNavierStokes.enstrophy
GeophysicalFlows.TwoDNavierStokes.enstrophy_dissipation_hyperviscosity
GeophysicalFlows.TwoDNavierStokes.enstrophy_dissipation_hypoviscosity
GeophysicalFlows.TwoDNavierStokes.enstrophy_work
```

### Private functions

```@docs
GeophysicalFlows.TwoDNavierStokes.Equation
GeophysicalFlows.TwoDNavierStokes.calcN!
GeophysicalFlows.TwoDNavierStokes.calcN_advection!
GeophysicalFlows.TwoDNavierStokes.addforcing!
GeophysicalFlows.TwoDNavierStokes.energy_dissipation
GeophysicalFlows.TwoDNavierStokes.enstrophy_dissipation
```


## `SingleLayerQG`

### Exported functions

```@docs
GeophysicalFlows.SingleLayerQG.Problem
GeophysicalFlows.SingleLayerQG.set_q!
GeophysicalFlows.SingleLayerQG.streamfunctionfrompv!
GeophysicalFlows.SingleLayerQG.energy
GeophysicalFlows.SingleLayerQG.kinetic_energy
GeophysicalFlows.SingleLayerQG.potential_energy
GeophysicalFlows.SingleLayerQG.energy_dissipation
GeophysicalFlows.SingleLayerQG.energy_work
GeophysicalFlows.SingleLayerQG.energy_drag
GeophysicalFlows.SingleLayerQG.enstrophy
GeophysicalFlows.SingleLayerQG.enstrophy_dissipation
GeophysicalFlows.SingleLayerQG.enstrophy_work
GeophysicalFlows.SingleLayerQG.enstrophy_drag
```

### Private functions

```@docs
GeophysicalFlows.SingleLayerQG.Equation
GeophysicalFlows.SingleLayerQG.calcN!
GeophysicalFlows.SingleLayerQG.calcN_advection!
GeophysicalFlows.SingleLayerQG.addforcing!
```


## `MultiLayerQG`

### Exported functions

```@docs
GeophysicalFlows.MultiLayerQG.Problem
GeophysicalFlows.MultiLayerQG.fwdtransform!
GeophysicalFlows.MultiLayerQG.invtransform!
GeophysicalFlows.MultiLayerQG.streamfunctionfrompv!
GeophysicalFlows.MultiLayerQG.pvfromstreamfunction!
GeophysicalFlows.MultiLayerQG.set_q!
GeophysicalFlows.MultiLayerQG.set_ψ!
GeophysicalFlows.MultiLayerQG.energies
GeophysicalFlows.MultiLayerQG.fluxes
```

### Private functions

```@docs
GeophysicalFlows.MultiLayerQG.LinearEquation
GeophysicalFlows.MultiLayerQG.Equation
GeophysicalFlows.MultiLayerQG.calcS!
GeophysicalFlows.MultiLayerQG.calcS⁻¹!
GeophysicalFlows.MultiLayerQG.calcN!
GeophysicalFlows.MultiLayerQG.calcNlinear!
GeophysicalFlows.MultiLayerQG.calcN_advection!
GeophysicalFlows.MultiLayerQG.calcN_linearadvection!
GeophysicalFlows.MultiLayerQG.addforcing!
```


## `SurfaceQG`

### Exported functions

```@docs
GeophysicalFlows.SurfaceQG.Problem
GeophysicalFlows.SurfaceQG.set_b!
GeophysicalFlows.SurfaceQG.kinetic_energy
GeophysicalFlows.SurfaceQG.buoyancy_variance
GeophysicalFlows.SurfaceQG.buoyancy_dissipation
GeophysicalFlows.SurfaceQG.buoyancy_work
```

### Private functions

```@docs
GeophysicalFlows.SurfaceQG.Equation
GeophysicalFlows.SurfaceQG.calcN!
GeophysicalFlows.SurfaceQG.calcN_advection!
GeophysicalFlows.SurfaceQG.addforcing!
```


## `BarotropicQGQL`

### Exported functions

```@docs
GeophysicalFlows.BarotropicQGQL.Problem
GeophysicalFlows.BarotropicQGQL.dissipation
GeophysicalFlows.BarotropicQGQL.work
GeophysicalFlows.BarotropicQGQL.drag
```

### Private functions

```@docs
GeophysicalFlows.BarotropicQGQL.calcN_advection!
GeophysicalFlows.BarotropicQGQL.addforcing!
```
