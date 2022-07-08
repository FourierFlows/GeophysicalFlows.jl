# # 2D decaying turbulence with tracers
#
# A simulation of decaying two-dimensional turbulence with tracers (in this case, 3 tracers with different initial conditions).
# 
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add GeophysicalFlows, Printf, Random, Plots"
# ```

# ## Let's begin
# Let's load `GeophysicalFlows.jl` and some other needed packages.
#
using GeophysicalFlows, Printf, Random, Plots
 
using Random: seed!
using GeophysicalFlows: peakedisotropicspectrum


# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
nothing # hide


# ## Numerical, domain, and simulation parameters
#
# First, we pick some numerical and physical parameters for our model.

n, L  = 128, 2π             # grid resolution and domain length
nothing # hide

# Then we pick the time-stepper parameters
    dt = 1e-2  # timestep
nsteps = 4000  # total number of steps
 nsubs = 20    # number of steps between each plot
 ntracers = 3   # number of tracers
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. The
# `stepper` keyword defines the time-stepper to be used.
prob = TwoDNavierStokesTracer.Problem(ntracers, dev; nx=n, Lx=L, ny=n, Ly=L, dt=dt, stepper="FilteredRK4")
nothing # hide

# Next we define some shortcuts for convenience.
sol, clock, vars, grid = prob.sol, prob.clock, prob.vars, prob.grid
x, y = grid.x, grid.y
nothing # hide


# ## Setting initial conditions

# Our initial condition tries to reproduce the initial condition used by McWilliams (_JFM_, 1984).
seed!(1234)
k₀, E₀ = 6, 0.5
ζ₀ = peakedisotropicspectrum(grid, k₀, E₀, mask=prob.timestepper.filter[:,:,1])

# Our initial condition for tracers (``c``).
# For the first tracer we use a gaussian centered at ``(x, y) = (L_x/5, 0)``.
gaussian(x, y, σ) = exp(-(x^2 + y^2) / (2σ^2))

amplitude, spread = 0.5, 0.15
for j in 1:ntracers
  if j>1
      global c₀ = cat(c₀, 0.0.*ζ₀, dims=3)
  else
      global c₀ = 0.0.*ζ₀
  end
end
c₀[:,:,1] = [amplitude * gaussian(x[i] - 0.1 * grid.Lx, y[j], spread) for i=1:grid.nx, j=1:grid.ny]
# For the second tracer we use a wider gaussian centered at a different location
c₀[:,:,2] = [amplitude * gaussian(x[i] - 0.2 * grid.Lx, y[j] + 0.1 * grid.Ly, 4*spread) for i=1:grid.nx, j=1:grid.ny]
# For the third tracer we use a straight band in the x-direction that is an eighth as wide as the domain
width = 1/8
[c₀[i,j,3] = amplitude for i=1:grid.nx, j=Int( (1/2) * grid.ny ):Int( (1/2 + width) * grid.ny )]

TwoDNavierStokesTracer.set_ζ_and_tracers!(prob, ζ₀, c₀)
nothing # hide

# Let's plot the initial vorticity field. Note that when plotting, we decorate the variable 
# to be plotted with `Array()` to make sure it is brought back on the CPU when `vars` live on 
# the GPU.
heatmap(x, y, Array(vars.ζ[:,:,1]'),
    aspectratio = 1,
              c = :balance,
           clim = (-40, 40),
          xlims = (-L/2, L/2),
          ylims = (-L/2, L/2),
         xticks = -3:3,
         yticks = -3:3,
         xlabel = "x",
         ylabel = "y",
          title = "initial vorticity",
     framestyle = :box)
            

# ## Diagnostics

# Create Diagnostics -- `energy` and `enstrophy` functions are imported at the top.
E = Diagnostic(TwoDNavierStokesTracer.energy, prob; nsteps=nsteps)
Z = Diagnostic(TwoDNavierStokesTracer.enstrophy, prob; nsteps=nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_decayingTwoDNavierStokesTracer"
plotname = "snapshots"
filename = joinpath(filepath, "decayingTwoDNavierStokesTracer.jld2")
nothing # hide

# Do some basic file management
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

# And then create Output
get_sol(prob) = prob.sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im * grid.l .* grid.invKrsq .* sol, grid.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
saveproblem(out)
nothing # hide


# ## Visualizing the simulation

# We initialize a plot with the vorticity field and the time-series of
# energy and enstrophy diagnostics.

p1 = heatmap(x, y, Array(vars.ζ[:,:,1]'),
         aspectratio = 1,
                   c = :balance,
                clim = (-40, 40),
               xlims = (-L/2, L/2),
               ylims = (-L/2, L/2),
              xticks = -3:3,
              yticks = -3:3,
              xlabel = "x",
              ylabel = "y",
               title = "vorticity, t=" * @sprintf("%.2f", clock.t),
          framestyle = :box)

p2 = plot(2, # this means "a plot with two series"
               label = ["energy E(t)/E(0)" "enstrophy Z(t)/Z(0)"],
              legend = :right,
           linewidth = 2,
               alpha = 0.7,
              xlabel = "t",
               xlims = (0, 41),
               ylims = (0, 1.1))

ptimeϕ₁ = heatmap(x, y, Array(vars.ζ[:,:,2]'),
            aspectratio = 1,
                      c = :balance,
                   clim = (-0.5, 0.5),
                  xlims = (-L/2, L/2),
                  ylims = (-L/2, L/2),
                 xticks = -3:3,
                 yticks = -3:3,
                 xlabel = "x",
                 ylabel = "y",
                  title = "ϕ₁(x, y, t=" * @sprintf("%.2f", clock.t) * ")",
             framestyle = :box)
      
  ptimeϕ₂ = heatmap(x, y, Array(vars.ζ[:,:,3]'),
             aspectratio = 1,
                       c = :balance,
                    clim = (-0.5, 0.5),
                   xlims = (-L/2, L/2),
                   ylims = (-L/2, L/2),
                  xticks = -3:3,
                  yticks = -3:3,
                  xlabel = "x",
                  ylabel = "y",
                   title = "ϕ₂(x, y, t=" * @sprintf("%.2f", clock.t) * ")",
              framestyle = :box)
      
  ptimeϕ₃ = heatmap(x, y, Array(vars.ζ[:,:,4]'),
              aspectratio = 1,
                        c = :balance,
                     clim = (-0.5, 0.5),
                    xlims = (-L/2, L/2),
                    ylims = (-L/2, L/2),
                   xticks = -3:3,
                   yticks = -3:3,
                   xlabel = "x",
                   ylabel = "y",
                    title = "ϕ₃(x, y, t=" * @sprintf("%.2f", clock.t) * ")",
               framestyle = :box)

p = plot(p1, p2, ptimeϕ₁, ptimeϕ₂, ptimeϕ₃, layout=5, size = (1500, 800))
#l = @layout Plots.grid(1, 2)
#p = plot(p1, p2, layout = l, size = (800, 360))


# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

startwalltime = time()

anim = @animate for j = 0:Int(nsteps/nsubs)
  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
    
    log = @sprintf("step: %04d, t: %d, cfl: %.2f, ΔE: %.4f, ΔZ: %.4f, walltime: %.2f min",
        clock.step, clock.t, cfl, E.data[E.i]/E.data[1], Z.data[Z.i]/Z.data[1], (time()-startwalltime)/60)

    println(log)
  end  

  p[1][1][:z] = Array(vars.ζ[:,:,1])
  p[1][:title] = "vorticity, t=" * @sprintf("%.2f", clock.t)
  push!(p[2][1], E.t[E.i], E.data[E.i]/E.data[1])
  push!(p[2][2], Z.t[Z.i], Z.data[Z.i]/Z.data[1])
  p[3][1][:z] = Array(vars.ζ[:,:,2])
  p[4][1][:z] = Array(vars.ζ[:,:,3])
  p[5][1][:z] = Array(vars.ζ[:,:,4])

  stepforward!(prob, diags, nsubs)
  TwoDNavierStokesTracer.updatevars!(prob)  
end

mp4(anim, "twodturbtracer.mp4", fps=18)


# Last we can save the output by calling
# ```julia
# saveoutput(out)`
# ```


# ## Radial energy spectrum

# After the simulation is done we plot the instantaneous radial energy spectrum to illustrate
# how `FourierFlows.radialspectrum` can be used,

E  = @. 0.5 * (vars.u^2 + vars.v^2) # energy density
Eh = rfft(E)                  # Fourier transform of energy density
kr, Ehr = FourierFlows.radialspectrum(Eh, grid, refinement=1) # compute radial specturm of `Eh`
nothing # hide

# and we plot it.
plot(kr, abs.(Ehr),
    linewidth = 2,
        alpha = 0.7,
       xlabel = "kᵣ", ylabel = "∫ |Ê| kᵣ dk_θ",
        xlims = (5e-1, grid.nx),
       xscale = :log10, yscale = :log10,
        title = "Radial energy spectrum",
       legend = false)
