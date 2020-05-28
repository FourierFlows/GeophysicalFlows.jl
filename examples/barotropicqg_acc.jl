# # Barotropic QG beta-plane turbulence over topography
#
#md # This example can be run online via [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/barotropicqgtopography.ipynb).
#md # Also, it can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/barotropicqgtopography.ipynb).
#
# An idealized version of the Southern Ocean. We solve the barotropic 
# quasi-geostrophic eddy dynamics in a flud with variable depth $H-h(x,y)$. We 
# also include an "Antarctic Circumpolar Current," i.e.,  a domain-average zonal 
# velocity $U(t)$ which is forced by constant wind  stress $F$ and influenced by 
# bottom drag and topographic form stress. The equations solved are: 
# $\partial_t \nabla^2\psi + \mathsf{J}(\psi-U y, \nabla^2\psi + \beta y + \eta) = -\mu\nabla^2\psi$
# and 
# $\partial_t U = F - \mu U -  \langle\psi\partial_x\eta\rangle$.


using FourierFlows, Plots, Printf

using FFTW: irfft

import GeophysicalFlows.BarotropicQG
import GeophysicalFlows.BarotropicQG: energy, meanenergy, enstrophy, meanenstrophy


# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
nothing # hide


# ## Numerical parameters and time-stepping parameters
     nx = 128            # 2D resolution = nx^2
stepper = "FilteredRK4"  # timestepper
    dt  = 0.1            # timestep
 nsteps = 10000          # total number of time-steps
 nsubs  = 25             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

Lx = 2π        # domain size
 ν = 4e-15     # viscosity
nν = 4         # viscosity order
 β = 1.4015    # the y-gradient of planetary PV
 μ = 1e-2      # linear drag
 F = 0.0012    # normalized wind stress forcing on domain-averaged zonal flow U(t) flow
nothing # hide

# Define the topographic potential vorticity, $f_0 h(x, y)/H$
topoPV(x, y) = 2cos(4x)*cos(4y)
nothing # hide

# and the forcing function $F$ (here forcing is constant in time) that acts on the domain-averaged $U$ equation.
calcFU(t) = F
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments,
prob = BarotropicQG.Problem(nx=nx, Lx=Lx, β=β, eta=topoPV,
                  calcFU=calcFU, ν=ν, nν=nν, μ=μ, dt=dt, stepper=stepper, dev=dev)
nothing # hide

# and define some shortcuts.
sol, cl, vs, pr, gr = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = gr.x, gr.y
nothing # hide


# ## Setting initial conditions

# Our initial condition is simply fluid at rest.
BarotropicQG.set_zeta!(prob, zeros(gr.nx, gr.ny))


# ## Diagnostics

# Create Diagnostics -- `energy`, `meanenergy`, `enstrophy`, and `meanenstrophy` functions are imported at the top.
E = Diagnostic(energy, prob; nsteps=nsteps)
Q = Diagnostic(enstrophy, prob; nsteps=nsteps)
Emean = Diagnostic(meanenergy, prob; nsteps=nsteps)
Qmean = Diagnostic(meanenstrophy, prob; nsteps=nsteps)
diags = [E, Emean, Q, Qmean]
nothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_barotropicqgtopography"
plotname = "snapshots"
filename = joinpath(filepath, "barotropicqgtopography.jld2")
nothing # hide

# Do some basic file management,
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

# and then create Output.
get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*gr.lr.*gr.invKrsq.*sol, gr.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
nothing # hide


# ## Visualizing the simulation

# We define a function that plots the potential vorticity field and the evolution 
# of energy and enstrophy.

function plot_output(prob)

  pq = heatmap(x, y, vs.q,
                 c = :balance,
              clim = (-2, 2),
       aspectratio = 1,
             xlims = (-gr.Lx/2, gr.Lx/2),
             ylims = (-gr.Ly/2, gr.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "∇²ψ + η",
        framestyle = :box)

  pE = plot(2,
             label = ["eddy energy" "mean energy"],
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 10.1),
             ylims = (0, 0.0008),
            xlabel = "μt")
          
  pQ = plot(2,
             label = ["eddy enstrophy" "mean enstrophy"],
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 10.1),
             ylims = (-0.02, 0.12),
            xlabel = "μt")

  l = @layout [ a{0.5w} grid(2, 1) ]
  p = plot(pq, pE, pQ, layout=l, size = (900, 600))

  return p
end
nothing # hide


# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

p = plot_output(prob)

startwalltime = time()

anim = @animate for j=0:Int(nsteps/nsubs)
  
  cfl = cl.dt*maximum([maximum(vs.U.+vs.u)/gr.dx, maximum(vs.v)/gr.dy])
  
  log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
    cl.step, cl.t, cfl, E.data[E.i], Q.data[Q.i], (time()-startwalltime)/60)
  
  if j%(2000/nsubs)==0; println(log) end
  
  p[1][1][:z] = Array(vs.q)
  p[1][:title] = "∇²ψ + η, μt="*@sprintf("%.2f", μ*cl.t)
  push!(p[2][1], μ*E.t[E.i], E.data[E.i])
  push!(p[2][2], μ*Emean.t[Emean.i], Emean.data[Emean.i])
  push!(p[3][1], μ*Q.t[Q.i], Q.data[Q.i])
  push!(p[3][2], μ*Qmean.t[Qmean.i], Qmean.data[Qmean.i])

  stepforward!(prob, diags, nsubs)
  BarotropicQG.updatevars!(prob)

end

mp4(anim, "barotropicqg_acc.mp4", fps=18)

# Note that since mean flow enstrophy is $Q_U = \beta U$ it can attain negative values. 


# ## Save

# Finally save the last snapshot.
savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename)
