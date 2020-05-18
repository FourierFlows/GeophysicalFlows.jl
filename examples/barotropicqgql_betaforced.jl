# # Quasi-Linear forced-dissipative barotropic QG beta-plane turbulence
#
#md # This example can be run online via [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/barotropicqgql_betaforced.ipynb). 
#md # Also, it can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/barotropicqgql_betaforced.ipynb).
#
# A simulation of forced-dissipative barotropic quasi-geostrophic turbulence on 
# a beta plane under the *quasi-linear approximation*. The dynamics include 
# linear drag and stochastic excitation.

using FourierFlows, Plots, Statistics, Printf, Random

using FourierFlows: parsevalsum
using FFTW: irfft
using Random: seed!
using Statistics: mean

import GeophysicalFlows.BarotropicQGQL
import GeophysicalFlows.BarotropicQGQL: energy, enstrophy


# ## Numerical parameters and time-stepping parameters

     nx = 128            # 2D resolution = nx^2
stepper = "FilteredRK4"  # timestepper
     dt = 0.05           # timestep
 nsteps = 8000           # total number of time-steps
 nsubs  = 10             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

Lx = 2π        # domain size
 β = 10.0      # planetary PV gradient
 μ = 0.01      # bottom drag
nothing # hide
 

# ## Forcing
#
# We force the vorticity equation with stochastic excitation that is delta-correlated
# in time and while spatially homogeneously and isotropically correlated. The forcing
# has a spectrum with power in a ring in wavenumber space of radious $k_f$ and 
# width $\delta k_f$, and it injects energy per unit area and per unit time equal 
# to $\varepsilon$.
forcing_wavenumber = 14.0    # the central forcing wavenumber for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 1.5     # the width of the forcing spectrum 
ε = 0.001                    # energy input rate by the forcing

gr  = TwoDGrid(nx, Lx)

k = [ gr.kr[i] for i=1:gr.nkr, j=1:gr.nl] # a 2D grid with the zonal wavenumber

forcing_spectrum = @. exp( -(sqrt(gr.Krsq)-forcing_wavenumber)^2 / (2forcing_bandwidth^2) )
@. forcing_spectrum[ gr.Krsq < (2π/Lx*2)^2  ] = 0
@. forcing_spectrum[ gr.Krsq > (2π/Lx*20)^2 ] = 0
@. forcing_spectrum[ k .< 2π/Lx ] .= 0 # make sure forcing does not have power at k=0 component
ε0 = parsevalsum(forcing_spectrum .* gr.invKrsq/2, gr) / (gr.Lx*gr.Ly)
@. forcing_spectrum = ε/ε0 * forcing_spectrum  # normalization so that forcing injects energy ε per domain area per unit time

seed!(1234) # reset of the random number generator for reproducibility
nothing # hide

# Next we construct function `calcF!` that computes a forcing realization every timestep
function calcF!(Fh, sol, t, clock, vars, params, grid)
  ξ = exp.(2π*im*rand(eltype(grid), size(sol)))/sqrt(clock.dt)
  @. Fh = ξ*sqrt.(forcing_spectrum)
  Fh[abs.(grid.Krsq).==0] .= 0
  nothing
end
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. Not providing
# a viscosity coefficient ν leads to the module's default value: ν=0. In this
# example numerical instability due to accumulation of enstrophy in high wavenumbers
# is taken care with the `FilteredTimestepper` we picked. 
prob = BarotropicQGQL.Problem(nx=nx, Lx=Lx, beta=β, mu=μ, dt=dt, stepper=stepper, 
                              calcF=calcF!, stochastic=true)
nothing # hide

# and define some shortcuts.
sol, cl, vs, pr, gr = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = gr.x, gr.y
nothing # hide


# First let's see how a forcing realization looks like.
calcF!(vs.Fh, sol, 0.0, cl, vs, pr, gr)

heatmap(x, y, irfft(vs.Fh, gr.nx),
     aspectratio = 1,
               c = :balance,
            clim = (-8, 8),
           xlims = (-gr.Lx/2, gr.Lx/2),
           ylims = (-gr.Ly/2, gr.Ly/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "a forcing realization",
      framestyle = :box)


# ## Setting initial conditions

# Our initial condition is simply fluid at rest.
BarotropicQGQL.set_zeta!(prob, zeros(gr.nx, gr.ny))
nothing # hide

# ## Diagnostics

# Create Diagnostics -- `energy` and `enstrophy` are functions imported at the top.
E = Diagnostic(energy, prob; nsteps=nsteps)
Z = Diagnostic(enstrophy, prob; nsteps=nsteps)
nothing # hide

# We can also define our custom diagnostics via functions.
function zetaMean(prob)
  sol = prob.sol
  sol[1, :]
end

zMean = Diagnostic(zetaMean, prob; nsteps=nsteps, freq=10)  # the zonal-mean vorticity
nothing # hide

# We combile all diags in a list.
diags = [E, Z, zMean] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_forcedbetaQLturb"
plotname = "snapshots"
filename = joinpath(filepath, "forcedbetaQLturb.jld2")
nothing # hide

# Do some basic file management,
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

# and then create Output.
get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*g.l.*g.invKrsq.*sol, g.nx)
out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
nothing # hide


# ## Visualizing the simulation

# We define a function that plots the vorticity and streamfunction fields, the 
# corresponding zonal-mean vorticity and zonal-mean zonal velocity and timeseries
# of energy and enstrophy.

function plot_output(prob)
  ζ̄, ζ′= prob.vars.Zeta, prob.vars.zeta
  ζ = @. ζ̄ + ζ′
  ψ̄, ψ′= prob.vars.Psi,  prob.vars.psi
  ψ = @. ψ̄ + ψ′
  ζ̄ₘ = mean(ζ̄, dims=1)'
  ūₘ = mean(prob.vars.U, dims=1)'

  pζ = heatmap(x, y, ζ,
       aspectratio = 1,
            legend = false,
                 c = :balance,
              clim = (-8, 8),
             xlims = (-gr.Lx/2, gr.Lx/2),
             ylims = (-gr.Ly/2, gr.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "vorticity ζ=∂v/∂x-∂u/∂y",
        framestyle = :box)

  pψ = contourf(x, y, ψ,
            levels = -0.32:0.04:0.32,
       aspectratio = 1,
         linewidth = 1,
            legend = false,
              clim = (-0.22, 0.22),
                 c = :viridis,
             xlims = (-gr.Lx/2, gr.Lx/2),
             ylims = (-gr.Ly/2, gr.Ly/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "streamfunction ψ",
        framestyle = :box)

  pζm = plot(ζ̄ₘ, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-3, 3),
            xlabel = "zonal mean ζ",
            ylabel = "y")
  plot!(pζm, 0*y, y, linestyle=:dash, linecolor=:black)

  pum = plot(ūₘ, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-0.5, 0.5),
            xlabel = "zonal mean u",
            ylabel = "y")
  plot!(pum, 0*y, y, linestyle=:dash, linecolor=:black)

  pE = plot(1,
             label = "energy",
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 4.1),
             ylims = (0, 0.05),
            xlabel = "μt")
          
  pZ = plot(1,
             label = "enstrophy",
         linecolor = :red,
            legend = :bottomright,
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 4.1),
             ylims = (0, 5),
            xlabel = "μt")

  l = @layout grid(2, 3)
  p = plot(pζ, pζm, pE, pψ, pum, pZ, layout=l, size = (1000, 600), dpi=150)
  
  return p
end

# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

p = plot_output(prob)

startwalltime = time()

anim = @animate for j=0:Int(nsteps/nsubs)

  cfl = cl.dt*maximum([maximum(vs.v)/gr.dy, maximum(vs.u+vs.U)/gr.dx])

  log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
    cl.step, cl.t, cfl, E.data[E.i], Z.data[Z.i],
    (time()-startwalltime)/60)

  if j%(1000/nsubs)==0; println(log) end

  p[1][1][:z] = @. vs.zeta + vs.Zeta
  p[1][:title] = "vorticity, μt="*@sprintf("%.2f", μ*cl.t)
  p[4][1][:z] = @. vs.psi + vs.Psi
  p[2][1][:x] = mean(vs.Zeta, dims=1)'
  p[5][1][:x] = mean(vs.U, dims=1)'
  push!(p[3][1], μ*E.t[E.i], E.data[E.i])
  push!(p[6][1], μ*Z.t[Z.i], Z.data[Z.i])
  
  stepforward!(prob, diags, nsubs)
  BarotropicQGQL.updatevars!(prob)

end

mp4(anim, "barotropicqgql_betaforced.mp4", fps=18)


# ## Save

# Finally save the last snapshot.
savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), cl.step)
savefig(savename)
