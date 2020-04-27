# # Phillips model of Baroclinic Instability
#
# A simulation of the growth of barolinic instability in the Phillips 2-layer model
# when we impose a vertical mean flow shear as a difference $\Delta U$ in the
# imposed, domain-averaged, zonal flow at each layer.

using FourierFlows, PyPlot, Printf

import GeophysicalFlows.MultilayerQG
import GeophysicalFlows.MultilayerQG: energies, fluxes


# ## Numerical parameters and time-stepping parameters

nx = 128          # 2D resolution = nx^2
ny = nx

stepper = "FilteredAB3"   # timestepper
dt  = 2e-3      # timestep
nsteps = 16000  # total number of time-steps
nsubs  = 4000   # number of time-steps for plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters
Lx = 2π         # domain size
 μ = 5e-2       # bottom drag
 β = 5          # the y-gradient of planetary PV
 
nlayers = 2     # number of layers
f0, g = 1, 1    # Coriolis parameter and gravitational constant
 H = [0.2, 0.8] # the rest depths of each layer
 ρ = [4.0, 5.0] # the density of each layer
 
 U = zeros(nlayers) # the imposed mean zonal flow in each layer
 U[1] = 1.0
 U[2] = 0.0
 nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments,
prob = MultilayerQG.Problem(nlayers=nlayers, nx=nx, Lx=Lx, f0=f0, g=g, H=H, ρ=ρ, U=U, dt=dt, stepper=stepper, μ=μ, β=β)
nothing # hide

# and define some shortcuts.
sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
x, y = gridpoints(gr)
nothing # hide


# ## Setting initial conditions

# Our initial condition is some small amplitude random flow.
MultilayerQG.set_q!(prob, 1e-2randn((nx, ny, nlayers)))
nothing # hide


# ## Diagnostics

# Create Diagnostics -- `energy` function is imported at the top.
E = Diagnostic(energies, prob; nsteps=nsteps)
diags = [E] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_2layer"
plotname = "snapshots"
filename = joinpath(filepath, "2layer.jld2")
nothing # hide

# Do some basic file management
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

# And then create Output
get_sol(prob) = sol # extracts the Fourier-transformed solution
function get_u(prob)
  @. v.qh = sol
  streamfunctionfrompv!(v.psih, v.qh, p.invS, g)
  @. v.uh = -im*g.l *v.psih
  invtransform!(v.u, v.uh, p)
  v.u
end
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
nothing #hide


# ## Visualizing the simulation

# We define a function that plots the potential vorticity field and the evolution 
# of energy and enstrophy.

function plot_output(prob, fig, axs; drawcolorbar=false, dpi=200)

  sol, v, p, g = prob.sol, prob.vars, prob.params, prob.grid
  MultilayerQG.updatevars!(prob)

  for j in 1:nlayers
    sca(axs[j])
    pcolormesh(x, y, v.q[:, :, j])
    axis("square")
    xlim(-Lx/2, Lx/2)
    ylim(-Lx/2, Lx/2)
    xticks([-2, 0, 2])
    yticks([-2, 0, 2])
    title(L"$q_"*string(j)*L"$")
    if drawcolorbar==true
      colorbar()
    end

    sca(axs[j+2])
    cla()
    contourf(x, y, v.psi[:, :, j])
    contour(x, y, v.psi[:, :, j], colors="k")
    axis("square")
    xlim(-Lx/2, Lx/2)
    ylim(-Lx/2, Lx/2)
    xticks([-2, 0, 2])
    yticks([-2, 0, 2])
    title(L"$\psi_"*string(j)*L"$")
    if drawcolorbar==true
      colorbar()
    end
  end

  sca(axs[5])
  cla()
  semilogy(μ*[E.t[i] for i=1:E.i], [E.data[i][1][1] for i=1:E.i], color="b", label=L"$KE_1$")
  plot(μ*[E.t[i] for i=1:E.i], [E.data[i][1][2] for i=1:E.i], color="r", label=L"$KE_2$")
  xlabel(L"\mu t")
  legend()

  sca(axs[6])
  cla()
  semilogy(μ*[E.t[i] for i=1:E.i], [E.data[i][2][1] for i=1:E.i], color="k", label=L"$PE_{3/2}$")
  xlabel(L"\mu t")
  legend()
end
nothing # hide


# ## Time-stepping the `Problem` forward

# Finally, we time-step the `Problem` forward in time.

startwalltime = time()

while cl.step < nsteps
  stepforward!(prob, diags, nsubs)

  log = @sprintf("step: %04d, t: %d, KE1: %.4f, KE2: %.4f, PE: %.4f, walltime: %.2f min", cl.step, cl.t, E.data[E.i][1][1], E.data[E.i][1][2], E.data[E.i][2][1], (time()-startwalltime)/60)

  println(log)
end
println("finished")


# ## Plot
# Now let's see what we got. We plot the output,

fig, axs = subplots(ncols=3, nrows=2, figsize=(15, 8), dpi=200)
plot_output(prob, fig, axs; drawcolorbar=false)
gcf() #hide

# and finally save the figure
savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename, dpi=240)
