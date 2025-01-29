# # [Forced-dissipative barotropic QG beta-plane turbulence](@id singlelayerqg_betaforced_example)
#
# A simulation of forced-dissipative barotropic quasi-geostrophic turbulence on 
# a beta plane. The dynamics include linear drag and stochastic excitation.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add GeophysicalFlows, CUDA, JLD2, CairoMakie, Statistics"
# ```

# ## Let's begin
# Let's load `GeophysicalFlows.jl` and some other packages we need.
#
using GeophysicalFlows, CUDA, JLD2, CairoMakie, Random, Printf

using Statistics: mean
using LinearAlgebra: ldiv!

parsevalsum = FourierFlows.parsevalsum
record = CairoMakie.record                # disambiguate between CairoMakie.record and CUDA.record
nothing #hide

# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
nothing #hide


# ## Numerical parameters and time-stepping parameters

      n = 128            # 2D resolution: n² grid points
stepper = "FilteredRK4"  # timestepper
     dt = 0.05           # timestep
 nsteps = 8000           # total number of timesteps
 save_substeps = 10      # number of timesteps after which output is saved
 
nothing #hide


# ## Physical parameters

L = 2π        # domain size
β = 10.0      # planetary PV gradient
μ = 0.01      # bottom drag
nothing #hide


# ## Forcing
#
# We force the vorticity equation with stochastic excitation that is delta-correlated in time 
# and while spatially homogeneously and isotropically correlated. The forcing has a spectrum 
# with power in a ring in wavenumber space of radius ``k_f`` (`forcing_wavenumber`) and width 
# ``δ_f`` (`forcing_bandwidth`), and it injects energy per unit area and per unit time 
# equal to ``\varepsilon``. That is, the forcing covariance spectrum is proportional to 
# ``\exp{[-(|\bm{k}| - k_f)^2 / (2 δ_f^2)]}``.

forcing_wavenumber = 14.0 * 2π/L  # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 1.5  * 2π/L  # the width of the forcing spectrum, `δ_f`
ε = 0.001                         # energy input rate by the forcing

grid = TwoDGrid(dev; nx=n, Lx=L)

K = @. sqrt(grid.Krsq)            # a 2D array with the total wavenumber

forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
@CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average

ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε/ε0       # normalize forcing to inject energy at rate ε
nothing #hide


# We reset of the random number generator for reproducibility
if dev==CPU(); Random.seed!(1234); else; CUDA.seed!(1234); end
nothing #hide


# Next we construct function `calcF!` that computes a forcing realization every timestep.
# For that, we call `randn!` to obtain complex numbers whose real and imaginary part
# are normally-distributed with zero mean and variance 1/2.

function calcF!(Fh, sol, t, clock, vars, params, grid)
  randn!(Fh)
  @. Fh *= sqrt(forcing_spectrum) / sqrt(clock.dt)
  return nothing
end
nothing #hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments.
# We use `stepper = "FilteredRK4"`. Filtered timesteppers apply a wavenumber-filter 
# at every time-step that removes enstrophy at high wavenumbers and, thereby,
# stabilize the problem, despite that we use the default viscosity coefficient `ν=0`.
prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, β, μ, dt, stepper,
                             calcF=calcF!, stochastic=true)
nothing #hide

# Let's define some shortcuts.
sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x,  y  = grid.x,  grid.y
Lx, Ly = grid.Lx, grid.Ly
nothing #hide


# First let's see how a forcing realization looks like. Note that when plotting, we decorate 
# the variable to be plotted with `Array()` to make sure it is brought back on the CPU when 
# `vars` live on the GPU.
calcF!(vars.Fh, sol, 0.0, clock, vars, params, grid)

fig = Figure()

ax = Axis(fig[1, 1], 
          xlabel = "x",
          ylabel = "y",
          aspect = 1,
          title = "a forcing realization",
          limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

heatmap!(ax, x, y, Array(irfft(vars.Fh, grid.nx));
         colormap = :balance, colorrange = (-8, 8))

fig


# ## Setting initial conditions

# Our initial condition is simply fluid at rest.
SingleLayerQG.set_q!(prob, device_array(dev)(zeros(grid.nx, grid.ny)))


# ## Diagnostics

# Create Diagnostic -- `energy` and `enstrophy` are functions imported at the top.
E = Diagnostic(SingleLayerQG.energy, prob; nsteps, freq=save_substeps)
Z = Diagnostic(SingleLayerQG.enstrophy, prob; nsteps, freq=save_substeps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing #hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_forcedbetaturb"
plotname = "snapshots"
filename = joinpath(filepath, "singlelayerqg_forcedbeta.jld2")
nothing #hide

# Do some basic file management,
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing #hide

# and then create Output.
get_sol(prob) = Array(prob.sol) # extracts the Fourier-transformed solution

function get_u(prob)
  vars, grid, sol = prob.vars, prob.grid, prob.sol
  
  @. vars.qh = sol

  SingleLayerQG.streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  ldiv!(vars.u, grid.rfftplan, -im * grid.l .* vars.ψh)

  return Array(vars.u)
end

output = Output(prob, filename, (:qh, get_sol), (:u, get_u))
nothing #hide

# We first save the problem's grid and other parameters so we can use them later.
saveproblem(output)

# and then call `saveoutput(output)` once to save the initial state.
saveoutput(output)

# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

startwalltime = time()

while clock.step <= nsteps
  if clock.step % 50save_substeps == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
  
    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min", 
    clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)

    println(log)
  end

  stepforward!(prob, diags, save_substeps)
  SingleLayerQG.updatevars!(prob)

  if clock.step % save_substeps == 0
    saveoutput(output)
  end
end

savediagnostic(E, "energy", output.path)
savediagnostic(Z, "enstrophy", output.path)


# ## Load saved output and visualize

# We now have output from our simulation saved in `singlelayerqg_forcedbeta.jld2` which 
# we can load to create a time series for the fields we are interested in.

file = jldopen(output.path)

iterations = parse.(Int, keys(file["snapshots/t"]))
t = [file["snapshots/t/$i"] for i ∈ iterations]

qh = [file["snapshots/qh/$i"] for i ∈ iterations]
u  = [file["snapshots/u/$i"] for i ∈ iterations]

E_t = file["diagnostics/energy/t"]
Z_t = file["diagnostics/enstrophy/t"]
E_data = file["diagnostics/energy/data"]
Z_data = file["diagnostics/enstrophy/data"]

x,  y  = file["grid/x"],  file["grid/y"]
nx, ny = file["grid/nx"], file["grid/ny"]
Lx, Ly = file["grid/Lx"], file["grid/Ly"]

close(file)


# We create a figure using Makie's [`Observable`](https://makie.juliaplots.org/stable/documentation/nodes/)s

n = Observable(1)

qₙ = @lift irfft(qh[$n], nx)
ψₙ = @lift irfft(- Array(grid.invKrsq) .* qh[$n], nx)
q̄ₙ = @lift real(ifft(qh[$n][1, :] / ny))
ūₙ = @lift vec(mean(u[$n], dims=1))

title_q = @lift @sprintf("vorticity, μt = %.2f", μ * t[$n])

energy    = Observable([Point2f(E_t[1], E_data[1])])
enstrophy = Observable([Point2f(Z_t[1], Z_data[1])])

fig = Figure(size = (1000, 600))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = 1,
               limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

axq = Axis(fig[1, 1]; title = title_q, axis_kwargs...)

axψ = Axis(fig[2, 1]; title = "streamfunction ψ", axis_kwargs...)

axq̄ = Axis(fig[1, 2],
           xlabel = "zonal mean vorticity",
           ylabel = "y",
           aspect = 1,
           limits = ((-3, 3), (-Ly/2, Ly/2)))

axū = Axis(fig[2, 2],
           xlabel = "zonal mean u",
           ylabel = "y",
           aspect = 1,
           limits = ((-0.5, 0.5), (-Ly/2, Ly/2)))

axE = Axis(fig[1, 3],
           xlabel = "μ t",
           ylabel = "energy",
           aspect = 1,
           limits = ((-0.1, 4.1), (0, 0.055)))

axZ = Axis(fig[2, 3],
           xlabel = "μ t",
           ylabel = "enstrophy",
           aspect = 1,
           limits = ((-0.1, 4.1), (0, 3.1)))

heatmap!(axq, x, y, qₙ;
         colormap = :balance, colorrange = (-8, 8))

levels = collect(-0.32:0.04:0.32)

contourf!(axψ, x, y, ψₙ;
          levels, colormap = :viridis)
contour!(axψ, x, y, ψₙ;
         levels, color = :black)

lines!(axq̄, q̄ₙ, y; linewidth = 3)
lines!(axq̄, 0y, y; linewidth = 1, linestyle=:dash)

lines!(axū, ūₙ, y; linewidth = 3)
lines!(axū, 0y, y; linewidth = 1, linestyle=:dash)

lines!(axE, energy; linewidth = 3)
lines!(axZ, enstrophy; linewidth = 3, color = :red)

fig


# We are now ready to animate all saved output.

frames = 1:length(t)
record(fig, "singlelayerqg_betaforced.mp4", frames, framerate = 16) do i
  n[] = i

  energy[]    = push!(energy[],    Point2f(μ * E_t[i], E_data[i]))
  enstrophy[] = push!(enstrophy[], Point2f(μ * Z_t[i], Z_data[i]))
end
nothing #hide

# ![](singlelayerqg_betaforced.mp4)

## we delete the .jld2 file before deploying the docs (takes too much space) #hide
rm(output.path) #hide
