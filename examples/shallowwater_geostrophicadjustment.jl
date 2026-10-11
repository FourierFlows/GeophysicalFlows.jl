# # [Geostrophic adjustment in shallow water](@id shallowwater_geostrophicadjustment_example)
#
# A simulation of the classic geostrophic adjustment problem in rotating shallow water.
# An elliptical Gaussian bump in the free surface is released from rest. Part of the initial
# disturbance radiates away as inertia-gravity waves, while the rest settles into a
# vortex in geostrophic balance, ``f \hat{\bm{z}} \times \bm{u} = - g \bm{\nabla} \eta``,
# with a horizontal scale set by the deformation radius ``L_d = \sqrt{g H} / f``.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add GeophysicalFlows, CairoMakie"
# ```

# ## Let's begin
# Let's load `GeophysicalFlows.jl` and some other packages we need.
#
using GeophysicalFlows, CairoMakie, Printf


# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
nothing #hide


# ## Numerical parameters and time-stepping parameters

      n = 256                       # 2D resolution = n²
stepper = "FilteredETDRK4"          # timestepper
     dt = 0.025                     # timestep
     tf = 40                        # length of time for simulation
 nsteps = round(Int, tf / dt)       # total number of time-steps
  nsubs = 10                        # number of time-steps for plotting (nsteps must be multiple of nsubs)
nothing #hide


# ## Physical parameters

L = 80          # domain size
f = 1.0         # Coriolis parameter
g = 1.0         # gravitational acceleration
H = 1.0         # rest depth of the fluid
nothing #hide

# The deformation radius is
L_d = sqrt(g * H) / f


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. The bathymetry
# is flat, ``b = -H``, so that at rest the fluid depth is ``h = H``.

grid = TwoDGrid(dev; nx=n, Lx=L)
x, y = gridpoints(grid)

b = - H * ones(n, n)    # flat bathymetry

prob = ShallowWater.Problem(dev; nx=n, Lx=L, f, g, b, dt, stepper)
nothing #hide

# Let's define some shortcuts.
sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y
#md nothing #hide


# ## Setting initial conditions
#
# The fluid starts from rest with an elliptical Gaussian bump in the free surface
# ``\eta = h + b``. The bump has width ``R_x`` in ``x`` and ``R_y = 2 R_x`` in ``y``, both
# comparable to the deformation radius.

η₀ = 0.1                           # amplitude of the free surface bump
Rx = 1.0                           # width of the free surface bump in x
Ry = 2Rx                           # width of the free surface bump in y

X, Y = gridpoints(grid)

ηᵢ = @. η₀ * exp(- (X^2 / 2Rx^2 + Y^2 / 2Ry^2))
hᵢ = @. H + ηᵢ
uᵢ = zero(X)
vᵢ = zero(X)

ShallowWater.set_uvh!(prob, uᵢ, vᵢ, hᵢ)
nothing #hide


# ## Diagnostics

# Create Diagnostics for the domain-averaged mass and energy.
M = Diagnostic(ShallowWater.mass, prob; nsteps)
E = Diagnostic(ShallowWater.energy, prob; nsteps)
diags = [M, E]
nothing #hide


# ## Visualizing the simulation

# We plot the free-surface elevation ``\eta``, the flow speed ``\sqrt{u^2 + v^2}``, and
# a section of ``\eta`` along ``y = 0``. Note that when plotting, we decorate the variable
# to be plotted with `Array()` to make sure it is brought back on the CPU when `vars` live
# on the GPU.

free_surface(vars, params) = Array(@. vars.h + params.b)
speed(vars) = Array(@. sqrt(vars.u^2 + vars.v^2))

η = Observable(free_surface(vars, params))
U = Observable(speed(vars))
η_section = Observable(free_surface(vars, params)[:, round(Int, n/2)+1])

title_η = Observable(@sprintf("free surface η, t = %.2f", clock.t))
title_U = Observable(@sprintf("speed √(u² + v²), t = %.2f", clock.t))

fig = Figure(size = (1300, 450))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = 1,
               limits = ((-L/4, L/4), (-L/4, L/4)))

axη = Axis(fig[1, 1]; title = title_η, axis_kwargs...)
axU = Axis(fig[1, 3]; title = title_U, axis_kwargs...)

axs = Axis(fig[1, 5];
           xlabel = "x",
           ylabel = "η(x, y=0)",
           limits = ((-L/4, L/4), (-0.02, 0.1)))

hη = heatmap!(axη, x, y, η;
              colormap = :balance, colorrange = (-0.04, 0.04))

Colorbar(fig[1, 2], hη)

hU = heatmap!(axU, x, y, U;
              colormap = :speed, colorrange = (0, 0.02))

Colorbar(fig[1, 4], hU)

lines!(axs, x, η_section; linewidth = 3)

fig


# ## Time-stepping the `Problem` forward

# We step the `Problem` forward in time and record an animation of the free surface and the flow speed.

startwalltime = time()

record(fig, "shallowwater_geostrophicadjustment.mp4", 0:round(Int, nsteps / nsubs), framerate = 24) do j
  if j % (200 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(abs, vars.u) / grid.dx, maximum(abs, vars.v) / grid.dy])
    cfl_gravitywaves = clock.dt * sqrt(g * maximum(vars.h)) / grid.dx

    log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, cfl (gravity waves): %.3f, E: %.6e, walltime: %.2f min",
                   clock.step, clock.t, cfl, cfl_gravitywaves, E.data[E.i], (time()-startwalltime)/60)

    println(log)
  end

  η[] = free_surface(vars, params)
  U[] = speed(vars)
  η_section[] = free_surface(vars, params)[:, round(Int, n/2)+1]
  title_η[] = @sprintf("free surface η, t = %.2f", clock.t)
  title_U[] = @sprintf("speed √(u² + v²), t = %.2f", clock.t)

  stepforward!(prob, diags, nsubs)
  ShallowWater.updatevars!(prob)
end
nothing #hide

# ![](shallowwater_geostrophicadjustment.mp4)


# ## Geostrophic balance
#
# After the inertia-gravity waves have radiated away, the flow near the origin is in
# geostrophic balance. Let's compare the meridional velocity ``v`` along ``y = 0`` and
# the zonal velocity ``u`` along ``x = 0`` with the geostrophic velocities
# ``v_g = (g / f) \partial_x \eta`` and ``u_g = - (g / f) \partial_y \eta``.

ηh = rfft(free_surface(vars, params))
u_g = - g / f * irfft(im * Array(grid.l)  .* ηh, n)
v_g =   g / f * irfft(im * Array(grid.kr) .* ηh, n)

i₀ = round(Int, n/2) + 1   # index for x = 0
j₀ = round(Int, n/2) + 1   # index for y = 0

fig = Figure(size = (1000, 400))

axv = Axis(fig[1, 1];
           xlabel = "x",
           ylabel = "velocity",
           title = @sprintf("meridional velocity along y = 0, t = %.2f", clock.t),
           limits = ((-L/4, L/4), nothing))

axu = Axis(fig[1, 2];
           xlabel = "y",
           ylabel = "velocity",
           title = @sprintf("zonal velocity along x = 0, t = %.2f", clock.t),
           limits = ((-L/4, L/4), nothing))

lines!(axv, x, Array(vars.v)[:, j₀]; linewidth = 3, label = "v")
lines!(axv, x, v_g[:, j₀]; linewidth = 3, linestyle = :dash, label = "(g/f) ∂η/∂x")

lines!(axu, y, Array(vars.u)[i₀, :]; linewidth = 3, label = "u")
lines!(axu, y, u_g[i₀, :]; linewidth = 3, linestyle = :dash, label = "- (g/f) ∂η/∂y")

axislegend(axv)
axislegend(axu)

fig

# Near the vortex core the flow is in geostrophic balance; farther out the departures
# from balance are the inertia-gravity waves that are still radiating away. Since the
# initial bump is wider in ``y`` than in ``x`` (``R_y > R_x``), the free-surface gradient, and thus
# the geostrophic velocity, is weaker and broader along ``x = 0`` than along ``y = 0``.

# Mass is conserved to machine precision and, since there is no explicit dissipation,
# energy is only lost to the high-wavenumber filter of the `FilteredETDRK4` timestepper:

@printf("relative change in mass: %.2e, relative change in energy: %.2e",
        (M.data[M.i] - M.data[1]) / M.data[1], (E.data[E.i] - E.data[1]) / E.data[1])
