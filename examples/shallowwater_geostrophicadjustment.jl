# # [Geostrophic adjustment in shallow water](@id shallowwater_geostrophicadjustment_example)
#
# A simulation of the classic geostrophic adjustment problem in rotating shallow water.
# A Gaussian bump in the free surface is released from rest. Part of the initial
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
stepper = "FilteredRK4"             # timestepper
     dt = 0.04                      # timestep
     tf = 16                        # length of time for simulation
 nsteps = round(Int, tf / dt)       # total number of time-steps
  nsubs = 10                        # number of time-steps for plotting (nsteps must be multiple of nsubs)
nothing #hide


# ## Physical parameters

L = 40          # domain size
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
# The fluid starts from rest with a Gaussian bump in the free surface ``\eta = h + b``,
# with width comparable to the deformation radius.

η₀ = 0.1                           # amplitude of the free surface bump
 R = 1.0                           # radius of the free surface bump

X, Y = gridpoints(grid)

ηᵢ = @. η₀ * exp(- (X^2 + Y^2) / 2R^2)
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

# We plot the free-surface elevation ``\eta`` and its section along ``y = 0``. Note that
# when plotting, we decorate the variable to be plotted with `Array()` to make sure it is
# brought back on the CPU when `vars` live on the GPU.

free_surface(vars, params) = Array(@. vars.h + params.b)

η = Observable(free_surface(vars, params))
η_section = Observable(free_surface(vars, params)[:, round(Int, n/2)+1])

title_η = Observable(@sprintf("free surface η, t = %.2f", clock.t))

fig = Figure(size = (900, 420))

axη = Axis(fig[1, 1];
           xlabel = "x",
           ylabel = "y",
           title = title_η,
           aspect = 1,
           limits = ((-L/4, L/4), (-L/4, L/4)))

axs = Axis(fig[1, 2];
           xlabel = "x",
           ylabel = "η(x, y=0)",
           limits = ((-L/2, L/2), (-0.02, 0.1)))

hm = heatmap!(axη, x, y, η;
              colormap = :balance, colorrange = (-0.04, 0.04))

Colorbar(fig[1, 0], hm, flipaxis = false)

lines!(axs, x, η_section; linewidth = 3)

fig


# ## Time-stepping the `Problem` forward

# We step the `Problem` forward in time and record an animation of the free surface.

startwalltime = time()

record(fig, "shallowwater_geostrophicadjustment.mp4", 0:round(Int, nsteps / nsubs), framerate = 18) do j
  if j % (100 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(abs, vars.u) / grid.dx, maximum(abs, vars.v) / grid.dy])
    cfl_gravitywaves = clock.dt * sqrt(g * maximum(vars.h)) / grid.dx

    log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, cfl (gravity waves): %.3f, E: %.6e, walltime: %.2f min",
                   clock.step, clock.t, cfl, cfl_gravitywaves, E.data[E.i], (time()-startwalltime)/60)

    println(log)
  end

  η[] = free_surface(vars, params)
  η_section[] = free_surface(vars, params)[:, round(Int, n/2)+1]
  title_η[] = @sprintf("free surface η, t = %.2f", clock.t)

  stepforward!(prob, diags, nsubs)
  ShallowWater.updatevars!(prob)
end
nothing #hide

# ![](shallowwater_geostrophicadjustment.mp4)


# ## Geostrophic balance
#
# After the inertia-gravity waves have radiated away, the flow near the origin is in
# geostrophic balance. Let's compare the meridional velocity ``v`` along ``y = 0`` with
# the geostrophic velocity ``v_g = (g / f) \partial_x \eta``.

ηh = rfft(free_surface(vars, params))
v_g = g / f * irfft(im * Array(grid.kr) .* ηh, n)

j₀ = round(Int, n/2) + 1

fig = Figure(size = (600, 400))

ax = Axis(fig[1, 1];
          xlabel = "x",
          ylabel = "velocity",
          title = @sprintf("meridional velocity along y = 0, t = %.2f", clock.t),
          limits = ((-L/4, L/4), nothing))

lines!(ax, x, Array(vars.v)[:, j₀]; linewidth = 3, label = "v")
lines!(ax, x, v_g[:, j₀]; linewidth = 3, linestyle = :dash, label = "(g/f) ∂η/∂x")

axislegend(ax)

fig

# Near the vortex core the flow is in geostrophic balance; farther out the departures
# from balance are the inertia-gravity waves that are still radiating away.

# Mass is conserved to machine precision and, since there is no explicit dissipation,
# energy is only lost to the high-wavenumber filter of the `FilteredRK4` timestepper:

@printf("relative change in mass: %.2e, relative change in energy: %.2e",
        (M.data[M.i] - M.data[1]) / M.data[1], (E.data[E.i] - E.data[1]) / E.data[1])
