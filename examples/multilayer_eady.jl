# # Eady model of Baroclinic Instability
#
# A simulation of the growth of barolinic instability in the Eady n-layer model
# when we impose a vertical mean flow shear as a difference $\Delta U$ in the
# imposed, domain-averaged, zonal flow at each layer.

using FourierFlows, Plots, Printf

using FFTW: rfft, irfft
import GeophysicalFlows.MultilayerQG
import GeophysicalFlows.MultilayerQG: energies

ENV["JULIA_NUM_THREADS"] = 8

# ## Numerical parameters and time-stepping parameters
nx = 64                   # 2D resolution = nx^2
ny = nx

stepper = "FilteredRK4"   # timestepper
min_time_step = 1e-2      # timestep
nsteps = 1000            # total number of time-steps
nsubs  = 10               # number of time-steps for plotting (nsteps must be multiple of nsubs)

# ## Physical parameters
Lx = 2π         # domain size
 μ = 5e-2       # bottom drag
 β = 5          # the y-gradient of planetary PV
 g = 1.0
f₀ = 4.0
 
# Vertical grid
nlayers = 6     # number of layers

total_depth = 1
z = [ (i - 1/2) * Δh for i = nlayers:-1:1 ] .- total_depth
H = [ Δh for i = 1:nlayers ]

# Density stratification
ρ₀ = 1
N² = 2

ρ_resting(z) = ρ₀ * (1 - N² * z / g)
ρ = ρ_resting.(z)

@info @sprintf("The largest Rossby radius of deformation is %.2f",
               sqrt(N²) * total_depth / f₀) 

@info @sprintf("The smallest Rossby radius of deformation is %.2f",
               sqrt(N²) * Δh / f₀) 

# Background shear
ΔU = 1 / (nlayers - 1)
U = [ i * ΔU for i = nlayers - 1 : -1 : 0 ]

twod_grid = TwoDGrid(nx, Lx)

x, y = gridpoints(twod_grid)

#eta(x, y) = 20 * f₀ * Δh / total_depth * exp(-y^2 / 0.125)
eta(x, y) = 40 * exp(-(x - y)^2 / 0.125) #cos(10y) * cos(10x)

# ## Problem setup
# 
# We initialize a `Problem` by providing a set of keyword arguments,

prob = MultilayerQG.Problem(nlayers = nlayers,
                                 nx = nx,
                                 Lx = Lx,
                                 f0 = f₀,
                                  g = g,
                                  H = H,
                                  ρ = ρ,
                                  U = U,
                                 dt = min_time_step,
                            stepper = stepper,
                                eta = eta.(x, y),
                                  μ = μ, 
                                  β = β
                           )

# and define some shortcuts.
sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
x, y = gr.x, gr.y

# ## Setting initial conditions

# Our initial condition is some small amplitude random noise. We smooth our initial
# condidtion using the `timestepper`'s high-wavenumber `filter`.

q_i  = 4e-3 * randn((nx, ny, nlayers))
qh_i = prob.timestepper.filter .* rfft(q_i, (1, 2)) # only apply rfft in dims=1, 2
q_i  = irfft(qh_i, gr.nx, (1, 2)) # only apply irfft in dims=1, 2

MultilayerQG.set_q!(prob, q_i)
nothing # hide

# ## Diagnostics

# Create Diagnostics -- `energies` function is imported at the top.
E = Diagnostic(energies, prob; nsteps=nsteps)
diags = [E] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.

nothing # hide

# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_eady"
plotname = "snapshots"
filename = joinpath(filepath, "eady.jld2")
nothing # hide

# Do some basic file management
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

# And then create Output
get_sol(prob) = sol # extracts the Fourier-transformed solution

function get_u(prob)
  @. v.qh = sol
  streamfunctionfrompv!(v.ψh, v.qh, p, g)
  @. v.uh = -im * g.l * v.ψh
  invtransform!(v.u, v.uh, p)
  return v.u
end

out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
nothing # hide

# ## Visualizing the simulation

# We define a function that plots the potential vorticity field and the evolution 
# of energy and enstrophy.

function plot_output(prob)
  
  layout = @layout grid(1, nlayers)
  p = plot(layout=layout, size=(1000, 600), dpi=150)
  
  for m in 1:nlayers

    heatmap!(p[m], x, y, vs.q[:, :, m],

         aspectratio = 1,
              legend = :none,
                   c = :balance,
               xlims = (-gr.Lx/2, gr.Lx/2),
               ylims = (-gr.Ly/2, gr.Ly/2),
              xticks = :none,
              yticks = :none,
            colorbar = false,
          framestyle = :box

         )
  end

  return p
end

nothing # hide

# ## Time-stepping the `Problem` forward

# Finally, we time-step the `Problem` forward in time.

p = plot_output(prob)

startwalltime = time()

anim = @animate for j = 0:Int(nsteps / nsubs)
  
    cfl = cl.dt * maximum([maximum(abs, vs.u) / gr.dx, maximum(abs, vs.v) / gr.dy])
    
    log = @sprintf("step: %04d, dt: %.3f, t: %0.2f, cfl: %.2f, KE1: %.4f, KE2: %.4f, PE: %.4f, walltime: %.2f min",
                   cl.step, cl.dt, cl.t, cfl, E.data[E.i][1][1], E.data[E.i][1][2], E.data[E.i][2][1], 
                   (time()-startwalltime)/60)

    j % (100 / nsubs) == 0 && println(log)
    
    for m in 1:nlayers
        p[m][1][:z] = @. vs.q[:, :, m]
    end
    
    # Adaptive time step!
    cl.dt = min(min_time_step, 0.5 / max(maximum(abs, vs.u) / gr.dx, maximum(abs, vs.v) / gr.dy))

    stepforward!(prob, diags, nsubs)

    MultilayerQG.updatevars!(prob)

end

mp4(anim, "multilayerqg_eady.mp4", fps=18)

# ## Save

#=
# Finally save the last snapshot.
savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename)
=#
