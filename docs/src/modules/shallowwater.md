# ShallowWater Module

### Basic Equations

This module solves the 2D shallow water equations:

```math
\begin{aligned}
\partial_t u + \boldsymbol{u \cdot \nabla} u - f v & = - g \frac{\partial \eta}{\partial x} - \mathrm{D} u , \\
\partial_t v + \boldsymbol{u \cdot \nabla} v + f u & = - g \frac{\partial \eta}{\partial y} - \mathrm{D} v , \\
\partial_t h + \boldsymbol{\nabla \cdot} ( \boldsymbol{u} h ) & = 0 .
\end{aligned}
```

where ``\eta(\bm{x}, t)`` is the free surface elevation, ``b(\bm{x})`` is the depth of the bottom (measured from the rest-height of the free surface), and, therefore

```math
h(\bm{x}, t) = \eta(\bm{x}, t) - b(\bm{x}) ,
```

is the total height of the fluid column. Thus, in terms of variables ``u``, ``v``, and ``h``:

```math
\begin{aligned}
\partial_t u + \boldsymbol{u \cdot \nabla} u - f v & = - g \partial_x h + g \partial_x b - \mathrm{D} u , \\
\partial_t v + \boldsymbol{u \cdot \nabla} v + f u & = - g \partial_y h + g \partial_y b - \mathrm{D} v , \\
\partial_t h + \boldsymbol{\nabla \cdot} ( \boldsymbol{u} h ) & = 0 .
\end{aligned}
```

The bathymetric gradients ``b(\bm{x})`` enter as a forcing on the momentum equations.

Often we write shallow water dynamics in conservative form using variables ``hu``, ``hv`` and ``h``. In these variables:

```math
\begin{aligned}
\partial_t (hu) + \partial_x \left [ \frac{(hu)^2}{h} \right ] + \partial_y \left [ \frac{(hu)(hv)}{h} \right ] - f (hv) & = - g \partial_x \left( \frac{h^2}{2} \right) + g h \partial_x b - h \mathrm{D} \left[ \frac{(hu)}{h} \right ] , \\
\partial_t (hv) + \partial_x \left [ \frac{(hu)(hv)}{h} \right ] + \partial_y \left [ \frac{(hv)^2}{h} \right ] + f (hu) & = - g \partial_y \left( \frac{h^2}{2} \right) + g h \partial_y b - h \mathrm{D} \left[ \frac{(hu)}{h} \right ] , \\
\partial_t h + \partial_x (h u) + \partial_y (h v) & = 0 .
\end{aligned}
```


### Implementation

## Examples
