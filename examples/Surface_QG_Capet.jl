# # Forced-dissipative barotropic QG beta-plane turbulence
#
#md # This example can be run online via [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/barotropicqg_betaforced.ipynb).
#md # Also, it can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/barotropicqg_betaforced.ipynb).
#
# A simulation of forced-dissipative barotropic quasi-geostrophic turbulence on
# a beta plane. The dynamics include linear drag and stochastic excitation.

using FourierFlows, Plots, Statistics, Printf, Random

using FourierFlows: parsevalsum
using FFTW: irfft, rfft
using Statistics: mean
using Random: seed!
using MAT

import GeophysicalFlows.SurfaceQG
import GeophysicalFlows.SurfaceQG: kinetic_energy, buoyvariance
import GeophysicalFlows.SurfaceQG: dissipation_bb, work_bb, drag_bb


# ## Choosing a device: CPU or GPU

dev = CPU()    # Device (CPU/GPU)ENV["GRDIR"]=""
Pkg.build("GR")
nothing # hide


# ## Numerical parameters and time-stepping parameters

     nx = 512            # 2D resolution = nx^2
stepper = "FilteredRK4"  # timestepper
     dt = 0.004          # timestep
 nsteps = 250           # total number of time-steps
 nsubs  = 50          # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

Lx = 2π        # domain size
 ϵ = 1e8           # amplification factor for initial buoyancy field
 nν = 4
 ν = 0.0025*(2e-5)*(2*π/nx)^(2*nν-2)
nothing # hide


# ## Forcing
#
# We force the vorticity equation with stochastic excitation that is delta-correlated
# in time and while spatially homogeneously and isotropically correlated. The forcing
# has a spectrum with power in a ring in wavenumber space of radious $k_f$ and
# width $\delta k_f$, and it injects energy per unit area and per unit time equal
# to $\varepsilon$.

init_wavenumber = 14.0    # the central intializing wavenumber for a spectrum that is a ring in wavenumber space

gr  = TwoDGrid(nx, Lx)

k = [ gr.kr[i] for i=1:gr.nkr, j=1:gr.nl] # a 2D grid with the zonal wavenumber

initial_spectrum = @. sqrt(gr.Krsq)^7 / ((sqrt(gr.Krsq) + init_wavenumber)^12.0)
init_bh = sqrt.(initial_spectrum).*exp.(2π*im*rand(eltype(initial_spectrum), size(initial_spectrum)))
init_b = irfft(init_bh, gr.nx)
init_b *= ϵ
#@. forcing_spectrum[ gr.Krsq < (2π/Lx*2)^2  ] = 0
#@. forcing_spectrum[ gr.Krsq > (2π/Lx*200)^2 ] = 0
#@. forcing_spectrum[ k .< 2π/Lx ] .= 0 # make sure forcing does not have power at k=0 component
#ε0 = parsevalsum(forcing_spectrum .* gr.invKrsq/2, gr)/(gr.Lx*gr.Ly)
#@. forcing_spectrum = ε/ε0 * forcing_spectrum  # normalization so that forcing injects energy ε per domain area per unit time

seed!(1234) # reset of the random number generator for reproducibility
nothing # hide

# Next we construct function `calcF!` that computes a forcing realization every timestep
function calcFb!(Fh, sol, t, clock, vars, params, grid)
  ξ = ArrayType(dev)(exp.(2π*im*rand(eltype(grid), size(sol)))/sqrt(clock.dt))
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
prob = SurfaceQG.Problem(dev; nx=nx, Lx=Lx, dt=dt, stepper=stepper,
                            ν=ν, nν=nν, stochastic=true)
nothing # hide

# Let's define some shortcuts.
sol, cl, vs, pr, gr = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = gr.x, gr.y
nothing # hide


# First let's see how a forcing realization looks like.
#calcFb!(vs.Fh, sol, 0.0, cl, vs, pr, gr)

heatmap(x, y, init_b,
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
SurfaceQG.set_b!(prob, init_b)


# ## Diagnostics

# Create Diagnostic -- `energy` and `enstrophy` are functions imported at the top.
bb = Diagnostic(buoyvariance, prob; nsteps=nsteps)
D = Diagnostic(dissipation_bb, prob; nsteps=nsteps) # dissipation by hyperviscosity
#W = Diagnostic(work_bb,        prob; nsteps=nsteps) # work input of energy
KE = Diagnostic(kinetic_energy, prob; nsteps=nsteps)

diags = [bb, D, KE] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hidenothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
# Define base filename so saved data can be distinguished from other runs
base_filename = string("SurfaceQG_n_", nx, "_visc_", round(ν, sigdigits=1), "_order_", 2*nν, "_ki_", init_wavenumber, "_eps_", ϵ)
# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
datapath = "./simulations/Output/Data/"
plotpath = "./simulations/Output/Figures/"
plotname = "snapshots"

dataname = joinpath(datapath, base_filename)
plotname = joinpath(plotpath, base_filename)
jld2_filename = joinpath(datapath, string(base_filename, ".jld2"))
matfilename= joinpath(datapath, base_filename)
nothing # hide

# Do some basic file management,
if isfile(jld2_filename); rm(jld2_filename); end
if isfile(matfilename); rm(matfilename); end
#if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

# and then create Output.
get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*gr.l.*gr.invKrsq.*sol, gr.nx)
out = Output(prob, dataname, (:sol, get_sol), (:u, get_u))
nothing # hide

function write_matlab_file(writepath, vars, grid, time, diags)
    bb, D, KE= diags
    step_path = string(writepath, "_", round(Int, cl.t), ".mat")
    matfile = matopen(step_path, "w")
    write(matfile, "b", vars.b)
    write(matfile, "u", vars.u)
    write(matfile, "v", vars.v)
    write(matfile, "nx", grid.nx)
    write(matfile, "ny", grid.ny)
    write(matfile, "dx", grid.dx)
    write(matfile, "dy", grid.dy)
    write(matfile, "Lx", grid.Lx)
    write(matfile, "Ly", grid.Ly)
    write(matfile, "KE", KE.data[KE.i])
    write(matfile, "bb", bb.data[bb.i])
    write(matfile, "Diss_Rate_bb", D.data[D.i])
    close(matfile)
end



# ## Visualizing the simulation

# We define a function that plots the vorticity and streamfunction fields, their
# corresponding zonal mean structure and timeseries of energy and enstrophy.

function plot_output(prob)
    bˢ = prob.vars.b
    b̅ˢ = mean(bˢ, dims=1)'
    ū = mean(prob.vars.u, dims=1)'
    Energy =  0.5*(prob.vars.u.^2 + prob.vars.v.^2)

  pζ = heatmap(x, y, bˢ,
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
             title = "buoyancy bˢ",
        framestyle = :box)

  pψ = heatmap(x, y, bˢ,
       aspectratio = 1,
            legend = false,
                 c = :balance,
              clim = (-8, 8),
             xlims = (-gr.Lx/8, gr.Lx/8),
             ylims = (-gr.Ly/8, gr.Ly/8),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "x",
            ylabel = "y",
             title = "buoyancy bˢ",
        framestyle = :box)

  pζm = plot(b̅ˢ, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-3, 3),
            xlabel = "zonal mean bˢ",
            ylabel = "y")
  plot!(pζm, 0*y, y, linestyle=:dash, linecolor=:black)

  pum = plot(ū, y,
            legend = false,
         linewidth = 2,
             alpha = 0.7,
            yticks = -3:3,
             xlims = (-0.5, 0.5),
            xlabel = "zonal mean u",
            ylabel = "y")
  plot!(pum, 0*y, y, linestyle=:dash, linecolor=:black)

  pE = plot(1,
             label = "kinetic energy",
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 4.1),
             ylims = (0, 5e-4),
            xlabel = "μt")

  pZ = plot(1,
             label = "enstrophy",
         linecolor = :red,
            legend = :bottomright,
         linewidth = 2,
             alpha = 0.7,
             xlims = (-0.1, 4.1),
             ylims = (0, 2.5),
            xlabel = "μt")

  l = @layout grid(2, 3)
  p = plot(pζ, pζm, pE, pψ, pum, pZ, layout=l, size = (1000, 600), dpi=150)

  return p
end
nothing # hide


# ## Time-stepping the `Problem` forward

# We time-step the `Problem` forward in time.

startwalltime = time()

p = plot_output(prob)

anim = @animate for j=0:Int(nsteps/nsubs)

  cfl = cl.dt*maximum([maximum(vs.u)/gr.dx, maximum(vs.v)/gr.dy])

  #log = @sprintf("step: %04d, t: %d, cfl: %.2f, KE: %.2e, Ens: %.2e, W: %.2e, Diss: %.2e, Drag: %.2e, walltime: %.2f min",
  #cl.step, cl.t, cfl, E.data[E.i], Z.data[Z.i], W.data[W.i], D.data[D.i], R.data[R.i],
  #(time()-startwalltime)/60)
  #log = @sprintf("step: %04d, t: %d, cfl: %.2f, KE: %.2e, Ens: %.2e, walltime: %.2f min",
  #cl.step, cl.t, cfl, E.data[E.i], Z.data[Z.i],
  #(time()-startwalltime)/60)

  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, walltime: %.2f min",
        cl.step, cl.t, cfl, (time()-startwalltime)/60)

        println(log)

  log = @sprintf("Energy diagnostics - Energy: %.2e, Diss: %.2e",
            KE.data[KE.i], D.data[D.i])

      println(log)

  #if j%(1000/nsubs)==0; println(log) end

  write_matlab_file(matfilename, vs, gr, cl.t, diags)

  p[1][1][:z] = Array(vs.b)
  p[1][:title] = "buoyancy, t="*@sprintf("%.2f", cl.t)
  p[4][1][:z] = Array(vs.b.*vs.b)
  p[2][1][:x] = mean(vs.b, dims=1)'
  p[5][1][:x] = mean(vs.u, dims=1)'
  push!(p[3][1], KE.t[KE.i], KE.data[KE.i])
  push!(p[6][1], bb.t[bb.i], bb.data[bb.i])

  #bh = rfft(vs.b)      # Fourier transform of energy density
  #kr, temp_spectrum= FourierFlows.radialspectrum(bh, gr, refinement=1)
  #if  @isdefined(buoyancy_spectrum) # compute radial specturm of `bh`
#      buoyancy_spectrum = [buoyancy_spectrum temp_spectrum]
#  else
#      buoyancy_spectrum = zeros(Complex{Float64}, size(kr,1), 1)
#      buoyancy_spectrum = temp_spectrum
 # end

  stepforward!(prob, diags, nsubs)
  SurfaceQG.updatevars!(prob)

end

mp4(anim, string(plotname, ".mp4"), fps=10)


#mean_buoyancy_spectrum = mean(buoyancy_spectrum, dims=2)

l = @layout grid(2, 2)

Eᵏ = 0.5*(vs.u.^2+vs.v.^2) # Update diagnosed Kinetic Energy
L = Lx

ps1 = heatmap(x, y, vs.b,
          aspectratio = 1,
          legend = false,
               c = :balance,
            clim = (-maximum(abs.(vs.b)), maximum(abs.(vs.b))),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "bˢ(x, y, t="*@sprintf("%.2f", cl.t)*")",
      framestyle = :box)

bh = rfft(vs.b)      # Fourier transform of energy density
kr, buoyancy_spectrum_r = FourierFlows.radialspectrum(bh, gr, refinement=1) # compute radial specturm of `bh`
#nothing # hide

ps2 = plot(kr, abs.(buoyancy_spectrum_r),
    linewidth = 2,
        alpha = 0.7,
       xlabel = "kᵣ", ylabel = "∫ |b̂|² kᵣ dk_θ",
        xlims = (1e0, 1e3),
        ylims = (1e0, 1e5),
       xscale = :log10, yscale = :log10,
        title = "Radial buoyancy variance spectrum",
       legend = false)

ps3 = ps1
ps4 = ps2


plot_spectra = plot(ps1, ps2, ps3, ps4, layout=l, size = (1200, 1200))

png(joinpath(plotpath, string("Spectra_", base_filename)))



i₀ = 1
dEdt_numerical = (KE[(i₀+1):KE.i] - KE[i₀:KE.i-1])/cl.dt #numerical first-order approximation of energy tendency
ii = (i₀):KE.i-1
ii2 = (i₀+1):KE.i

t = KE.t[ii]
#dEdt_computed = W[ii2] - D[ii] - R[ii]

# residual = dEdt_computed - dEdt_numerical

l = @layout grid(2, 3)

Eᵏ = 0.5*(vs.u.^2+vs.v.^2) # Update diagnosed Kinetic Energy
L = Lx

pb4 = heatmap(x, y, vs.b,
          aspectratio = 1,
          legend = false,
               c = :balance,
            clim = (-maximum(abs.(vs.b)), maximum(abs.(vs.b))),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "bˢ(x, y, t="*@sprintf("%.2f", cl.t)*")",
      framestyle = :box)

pb2 = plot(t, [0*t -D[ii]],
           label = [ "ensemble mean work, <W>" "dissipation, D"],
       linestyle = [:solid :dash :solid :solid],
       linewidth = 2,
           alpha = 0.8,
          xlabel = "t",
          ylabel = "energy sources and sinks")

          pb3 = plot(t, KE[ii],
                     label = ["Energy, E"],
                 linestyle = [:solid :dash :solid :solid],
                 linewidth = 2,
                     alpha = 0.8,
                    xlabel = "t",
                    ylabel = "Energy evolution")

                    pb1 = heatmap(x, y, vs.b,
                              aspectratio = 1,
                              legend = false,
                                   c = :balance,
                                clim = (-maximum(abs.(vs.b)), maximum(abs.(vs.b))),
                               xlims = (-L/8, L/8),
                               ylims = (-L/8, L/8),
                              xticks = -3:3,
                              yticks = -3:3,
                              xlabel = "x",
                              ylabel = "y",
                               title = "bˢ(x, y, t="*@sprintf("%.2f", cl.t)*")",
                          framestyle = :box)

                          pb5 = plot(t, [0*t -D[ii]],
                                     label = ["ensemble mean work, <W>" "dissipation, D"],
                                 linestyle = [:solid :dash :solid :solid],
                                 linewidth = 2,
                                     alpha = 0.8,
                                    xlabel = "t",
                                    ylabel = "energy sources and sinks")

                                    pb6 = plot(t, KE[ii],
                                               label = ["Energy, E"],
                                           linestyle = [:solid :dash :solid :solid],
                                           linewidth = 2,
                                               alpha = 0.8,
                                              xlabel = "t",
                                              ylabel = "Energy evolution")

plot_budgets = plot(pb1, pb2, pb3, pb4, pb5, pb6, layout=l, size = (2400, 1200))

png(plotname)

# Last we save the output.
saveoutput(out)
