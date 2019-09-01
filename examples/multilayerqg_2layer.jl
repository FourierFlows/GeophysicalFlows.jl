using
  PyPlot,
  JLD2,
  Printf,
  FourierFlows

using FFTW: ifft

import GeophysicalFlows.MultilayerQG
import GeophysicalFlows.MultilayerQG: energies, fluxes


# Numerical parameters and time-stepping parameters
nx = 128          # 2D resolution = nx^2
ny = nx

stepper = "FilteredRK4"   # timestepper
dt  = 5e-3     # timestep
nsteps = 40000 # total number of time-steps
nsubs  = 250   # number of time-steps for plotting
               # (nsteps must be multiple of nsubs)

# Physical parameters
 Lx  = 2π      # domain size
   μ = 5e-2    # bottom drag
beta = 5       # the y-gradient of planetary PV

nlayers = 2       # these choice of parameters give the
f0, g = 1, 1      # desired PV-streamfunction relations
  H = [0.2, 0.8]  # q1 = Δψ1 + 25*(ψ2-ψ1), and
rho = [4.0, 5.0]  # q2 = Δψ2 + 25/4*(ψ1-ψ2).
  U = zeros(nlayers)
  U[1] = 1.0
  U[2] = 0.0

gr = TwoDGrid(nx, Lx)
x, y = gridpoints(gr)
k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers

# Initialize problem
prob = MultilayerQG.Problem(nlayers=nlayers, nx=nx, Lx=Lx, f0=f0, g=g, H=H, rho=rho, U=U, dt=dt, stepper=stepper, μ=μ, beta=beta)
sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

# Files
filepath = "."
plotpath = "./plots_2layer"
plotname = "snapshots"
filename = joinpath(filepath, "2layer.jld2")

# File management
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end

# Initialize with zeros
MultilayerQG.set_q!(prob, 1e-2randn((nx, ny, nlayers)))


# Create Diagnostics
E = Diagnostic(energies, prob; nsteps=nsteps)
diags = [E]

# Create Output
get_sol(prob) = sol # extracts the Fourier-transformed solution
function get_u(prob)
  @. v.qh = sol
  streamfunctionfrompv!(v.psih, v.qh, p.invS, g)
  @. v.uh = -im*g.l *v.psih
  invtransform!(v.u, v.uh, p)
  v.u
end
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))


function plot_output(prob, fig, axs; drawcolorbar=false)

  # Plot the PV field and the evolution of energy and enstrophy.

  sol, v, p, g = prob.sol, prob.vars, prob.params, prob.grid
  MultilayerQG.updatevars!(prob)

  for j in 1:nlayers
    sca(axs[j])
    pcolormesh(x, y, v.q[:, :, j])
    axis("square")
    xlim(-Lx/2, Lx/2)
    ylim(-Lx/2, Lx/2)
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
    title(L"$\psi_"*string(j)*L"$")
    if drawcolorbar==true
      colorbar()
    end
  end

  sca(axs[5])
  plot(μ*E.t[E.i], E.data[E.i][1][1], ".", color="b", label=L"$KE_1$")
  plot(μ*E.t[E.i], E.data[E.i][1][2], ".", color="r", label=L"$KE_2$")
  xlabel(L"\mu t")
  ylabel(L"KE")
  # legend()

  sca(axs[6])
  plot(μ*E.t[E.i], E.data[E.i][2][1], ".", color="k", label=L"$PE_{3/2}$")
  xlabel(L"\mu t")
  ylabel(L"PE")
  # legend()

  pause(0.001)
end


fig, axs = subplots(ncols=3, nrows=2, figsize=(15, 8))
plot_output(prob, fig, axs; drawcolorbar=true)


# Step forward
startwalltime = time()

while cl.step < nsteps
  stepforward!(prob, diags, nsubs)

  # Message
  log = @sprintf("step: %04d, t: %d, KE1: %.4f, KE2: %.4f, PE: %.4f, τ: %.2f min", cl.step, cl.t, E.data[E.i][1][1], E.data[E.i][1][2], E.data[E.i][2][1], (time()-startwalltime)/60)

  println(log)

  plot_output(prob, fig, axs; drawcolorbar=false)

end
println((time()-startwalltime))

plot_output(prob, fig, axs; drawcolorbar=false)

savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename, dpi=240)
