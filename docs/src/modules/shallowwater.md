# ShallowWater Module

### Basic Equations

This module solves the 2D shallow water equations:

```math
\begin{aligned}
\frac{\partial u}{\partial t} + \boldsymbol{u \cdot \nabla} u - f v & = - g \frac{\partial \eta}{\partial x} - \mathrm{D} u, \\
\frac{\partial v}{\partial t} + \boldsymbol{u \cdot \nabla} v + f u & = - g \frac{\partial \eta}{\partial y} - \mathrm{D} v, \\
\frac{\partial h}{\partial t} + \boldsymbol{\nabla \cdot} ( \boldsymbol{u} h ) & = - \mathrm{D} h.
\end{aligned}
```

where ``\eta(x, y, t)`` is the free surface elevation, ``d(x, y)`` is the depth of the bottom (measured from the rest-height of the fluid), and, therefore

```math
h = \eta - d \ ,
```

is the total height of the fluid column.

### Implementation

## Examples
