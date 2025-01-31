# Functions

## `GeophysicalFlows`
```@docs
GeophysicalFlows.GeophysicalFlows
```

### Exported functions

```@docs
GeophysicalFlows.lambdipole
GeophysicalFlows.peakedisotropicspectrum
```


## `TwoDNavierStokes`

### Exported functions

```@docs
GeophysicalFlows.TwoDNavierStokes.Problem
GeophysicalFlows.TwoDNavierStokes.energy_dissipation_hyperviscosity
GeophysicalFlows.TwoDNavierStokes.energy_dissipation_hypoviscosity
GeophysicalFlows.TwoDNavierStokes.energy_work
GeophysicalFlows.TwoDNavierStokes.enstrophy_dissipation_hyperviscosity
GeophysicalFlows.TwoDNavierStokes.enstrophy_dissipation_hypoviscosity
GeophysicalFlows.TwoDNavierStokes.enstrophy_work
GeophysicalFlows.TwoDNavierStokes.palinstrophy
```

### Private functions

```@docs
GeophysicalFlows.TwoDNavierStokes.calcN_advection!
GeophysicalFlows.TwoDNavierStokes.addforcing!
GeophysicalFlows.TwoDNavierStokes.energy_dissipation
GeophysicalFlows.TwoDNavierStokes.enstrophy_dissipation
```


## `SingleLayerQG`

### Exported functions

```@docs
GeophysicalFlows.SingleLayerQG.Problem
GeophysicalFlows.SingleLayerQG.streamfunctionfrompv!
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
```

### Private functions

```@docs
GeophysicalFlows.MultiLayerQG.LinearEquation
GeophysicalFlows.MultiLayerQG.calcS!
GeophysicalFlows.MultiLayerQG.calcS⁻¹!
GeophysicalFlows.MultiLayerQG.calcNlinear!
GeophysicalFlows.MultiLayerQG.calcN_advection!
GeophysicalFlows.MultiLayerQG.calcN_linearadvection!
GeophysicalFlows.MultiLayerQG.addforcing!
GeophysicalFlows.MultiLayerQG.pv_streamfunction_kernel!
```


## `SurfaceQG`

### Exported functions

```@docs
GeophysicalFlows.SurfaceQG.Problem
GeophysicalFlows.streamfunctionfromb!
GeophysicalFlows.SurfaceQG.buoyancy_dissipation
GeophysicalFlows.SurfaceQG.buoyancy_work
```

### Private functions

```@docs
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
