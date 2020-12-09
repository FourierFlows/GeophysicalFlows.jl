# ShallowWater Module

### Basic Equations

This module solves the 2D shallow water equations:

```math
\begin{aligned}
\frac{\partial u}{\partial t} + \boldsymbol{u \cdot \nabla} u - f v & = - g \frac{\partial \eta}{\partial x} - \mathrm{D} u, \\
\frac{\partial v}{\partial t} + \boldsymbol{u \cdot \nabla} v + f u & = - \mathrm{D} v, \\
\frac{\partial \eta}{\partial t} + \boldsymbol{\nabla \cdot} ( \boldsymbol{u} h ) & = - \mathrm{D} \eta.
\end{aligned}
```

### Implementation

## Examples
