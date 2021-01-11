# ShallowWater Module

### Basic Equations

This module solves the 2D shallow water equations:

```math
\begin{aligned}
\partial_t u + \bm{u \cdot \nabla} u - f v & = - g \partial_x \eta , \\
\partial_t v + \bm{u \cdot \nabla} v + f u & = - g \partial_y \eta , \\
\partial_t h + \bm{\nabla \cdot} (\bm{u} h) & = 0 ,
\end{aligned}
```

where ``\eta(\bm{x}, t)`` is the free surface elevation and ``b(\bm{x})`` is the depth of the 
bottom (both measured from the ``z=0`` rest-height of the free surface). Thus, the total height 
of the fluid column is

```math
h(\bm{x}, t) = \eta(\bm{x}, t) - b(\bm{x}) .
```

In terms of variables ``u``, ``v``, and ``h``:

```math
\begin{aligned}
\partial_t u + \bm{u \cdot \nabla} u - f v & = - g \partial_x h + g \partial_x b , \\
\partial_t v + \bm{u \cdot \nabla} v + f u & = - g \partial_y h + g \partial_y b , \\
\partial_t h + \bm{\nabla \cdot} (\bm{u} h) & = 0 .
\end{aligned}
```

The bathymetric gradients ``b(\bm{x})`` enter as a forcing on the momentum equations.

Often we write shallow water dynamics in conservative form using variables ``q_u = hu``, ``q_v = hv`` and ``h``. In these variables:

```math
\begin{aligned}
\partial_t q_u + \partial_x \left ( \frac{q_u^2}{h} \right ) + \partial_y \left ( \frac{q_u q_v}{h} \right ) - f q_v & = - g \partial_x \left( \frac{h^2}{2} \right) + g h \partial_x b + \nu \bm{\nabla \cdot} \left [ \mu(h) \bm{\nabla} \left ( \frac{q_u}{h} \right) \right ] , \\
\partial_t q_v + \partial_x \left ( \frac{q_u q_v}{h} \right ) + \partial_y \left ( \frac{q_v^2}{h} \right ) + f q_u & = - g \partial_y \left( \frac{h^2}{2} \right) + g h \partial_y b  + \nu \bm{\nabla \cdot} \left [ \mu(h) \bm{\nabla} \left ( \frac{q_v}{h} \right) \right ] , \\
\partial_t h + \partial_x q_u + \partial_y q_v & = 0 .
\end{aligned}
```


### Implementation

## Examples
