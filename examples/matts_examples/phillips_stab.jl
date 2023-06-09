# a test space for building a linear stability analysis tool in julia
# for now it is separate from GeophysicalFlows.jl Model object, but might
# be merged in the future

## load packages

using GeophysicalFlows, CairoMakie, Printf, FFTW, LinearAlgebra, Statistics

using Random: seed!

include("./mjcl_stab.jl")

## build basic model

dev = CPU()     # device (CPU)

# numerical params
Ny = Nx = n = 128                  # 2D resolution = n²
stepper = "FilteredRK4"  # time stepping scheme
dt      = 1e-2              # time step
nsteps  = 60000           # total number of time-steps 20000
nsubs   = 50              # number of time-steps for plotting (nsteps must be multiple of nsubs)

# physical params
L = 2π                   # domain size
μ = 5e-2                 # bottom drag
beta = β = 0. #1.2130692965249345e-11                    # the y-gradient of planetary PV

nlayers = 8             # number of layers
f0 = f₀= 1. #0.0001236812857687059 # coriolis param [s^-1] 
g = 9.81
H = [0.1, 0.3, 1.]           # the rest depths of each layer
rho = ρ = [4.0, 5.0, 5.1]           # the density of each layer

H=ones(8)
rho = ρ =collect(range(4.0,5.0,8))
U = collect(range(1.0,0.0,8))
V = zeros(8)

# U = zeros(nlayers)       # the imposed mean zonal flow in each layer
# U[1] = 1.0
# U[2] = 0.05
# U[3] = 0.005


# V = zeros(nlayers)       # the imposed mean zonal flow in each layer
# V[1] = 1.0
# V[2] = 0.5
# V[3] = 0.25

# setting up the ``problem''
prob = MultiLayerQG.Problem(nlayers, dev; nx=n, Lx=L, f₀, g, H, ρ, U, μ, β,
                            dt, stepper, aliased_fraction=0)

sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
x, y = grid.x, grid.y

# initial conditions
seed!(1234) # reset of the random number generator for reproducibility
q₀  = 1e-2 * device_array(dev)(randn((grid.nx, grid.ny, nlayers)))
q₀h = prob.timestepper.filter .* rfft(q₀, (1, 2)) # apply rfft  only in dims=1, 2
q₀  = irfft(q₀h, grid.nx, (1, 2))                 # apply irfft only in dims=1, 2

MultiLayerQG.set_q!(prob, q₀)

# diagnostics
E = Diagnostic(MultiLayerQG.energies, prob; nsteps)
diags = [E] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.

# output dirs
filepath = "."
plotpath = "./figs/plots_2layer"
plotname = "snapshots"
filename = joinpath(filepath, "2layer.jld2")

# file management
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end

# ``create output'' (?)
get_sol(prob) = prob.sol # extracts the Fourier-transformed solution

function get_u(prob)
  sol, params, vars, grid = prob.sol, prob.params, prob.vars, prob.grid

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l * vars.ψh
  invtransform!(vars.u, vars.uh, params)

  return vars.u
end

out = Output(prob, filename, (:sol, get_sol), (:u, get_u))

# visualizing the simulation...

Lx, Ly = grid.Lx, grid.Ly

title_KE = Observable(@sprintf("μt = %.2f", μ * clock.t))

q1 = Observable(Array(vars.q[:, :, 1]))
q2 = Observable(Array(vars.q[:, :, 2]))
q3 = Observable(Array(vars.q[:, :, 3]))
q4 = Observable(Array(vars.q[:, :, 4]))
q5 = Observable(Array(vars.q[:, :, 5]))
q6 = Observable(Array(vars.q[:, :, 6]))
q7 = Observable(Array(vars.q[:, :, 7]))
q8 = Observable(Array(vars.q[:, :, 8]))

# psi1 = Observable(Array(vars.ψ[:, :, 1]))
# psi2 = Observable(Array(vars.ψ[:, :, 2]))
# psi3 = Observable(Array(vars.ψ[:, :, 3]))
# psi4 = Observable(Array(vars.ψ[:, :, 4]))
# psi5 = Observable(Array(vars.ψ[:, :, 5]))
# psi6 = Observable(Array(vars.ψ[:, :, 6]))
# psi7 = Observable(Array(vars.ψ[:, :, 7]))
# psi8 = Observable(Array(vars.ψ[:, :, 8]))

# function compute_levels(maxf, nlevels=8)
#   # -max(|f|):...:max(|f|)
#   levelsf  = @lift collect(range(-$maxf, stop = $maxf, length=nlevels))

#   # only positive
#   levelsf⁺ = @lift collect(range($maxf/(nlevels-1), stop = $maxf, length=Int(nlevels/2)))

#   # only negative
#   levelsf⁻ = @lift collect(range(-$maxf, stop = -$maxf/(nlevels-1), length=Int(nlevels/2)))

#   return levelsf, levelsf⁺, levelsf⁻
# end

# maxpsi1 = Observable(maximum(abs, vars.ψ[:, :, 1]))
# maxpsi2 = Observable(maximum(abs, vars.ψ[:, :, 2]))
# maxpsi3 = Observable(maximum(abs, vars.ψ[:, :, 3]))
# maxpsi4 = Observable(maximum(abs, vars.ψ[:, :, 4]))
# maxpsi5 = Observable(maximum(abs, vars.ψ[:, :, 5]))
# maxpsi6 = Observable(maximum(abs, vars.ψ[:, :, 6]))
# maxpsi7 = Observable(maximum(abs, vars.ψ[:, :, 7]))
# maxpsi8 = Observable(maximum(abs, vars.ψ[:, :, 8]))


# levelspsi1, levelspsi1⁺, levelspsi1⁻ = compute_levels(maxpsi1)
# levelspsi2, levelspsi2⁺, levelspsi2⁻ = compute_levels(maxpsi2)
# levelspsi3, levelspsi3⁺, levelspsi3⁻ = compute_levels(maxpsi3)
# levelspsi4, levelspsi4⁺, levelspsi4⁻ = compute_levels(maxpsi4)
# levelspsi5, levelspsi5⁺, levelspsi5⁻ = compute_levels(maxpsi5)
# levelspsi6, levelspsi6⁺, levelspsi6⁻ = compute_levels(maxpsi6)
# levelspsi7, levelspsi7⁺, levelspsi7⁻ = compute_levels(maxpsi7)
# levelspsi8, levelspsi8⁺, levelspsi8⁻ = compute_levels(maxpsi8)

# KE₁ = Observable(Point2f[(μ * E.t[1], E.data[1][1][1])])
# KE₂ = Observable(Point2f[(μ * E.t[1], E.data[1][1][end])])
# PE  = Observable(Point2f[(μ * E.t[1], E.data[1][2][end])])

fig = Figure(resolution=(1000, 600))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = 1,
               limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

axq1 = Axis(fig[1, 1]; title = "q1", axis_kwargs...)

# axpsi1 = Axis(fig[2, 1]; title = "psi1", axis_kwargs...)

axq2 = Axis(fig[1, 2]; title = "q2", axis_kwargs...)

# axpsi2 = Axis(fig[2, 2]; title = "psi2", axis_kwargs...)

axq3 = Axis(fig[1, 3]; title = "q3", axis_kwargs...)
axq4 = Axis(fig[1, 4]; title = "q4", axis_kwargs...)
axq5 = Axis(fig[2, 1]; title = "q5", axis_kwargs...)
axq6 = Axis(fig[2, 2]; title = "q6", axis_kwargs...)
axq7 = Axis(fig[2, 3]; title = "q7", axis_kwargs...)
axq8 = Axis(fig[2, 4]; title = "q8", axis_kwargs...)

# axKE = Axis(fig[1, 3],
#             xlabel = "μ t",
#             ylabel = "KE",
#             title = title_KE,
#             yscale = log10,
#             limits = ((-0.1, 2.6), (1e-9, 5)))

# axPE = Axis(fig[2, 3],
#             xlabel = "μ t",
#             ylabel = "PE",
#             yscale = log10,
#             limits = ((-0.1, 2.6), (1e-9, 5)))

heatmap!(axq1, x, y, q1; colormap = :balance)
heatmap!(axq2, x, y, q2; colormap = :balance)
heatmap!(axq3, x, y, q3; colormap = :balance)
heatmap!(axq4, x, y, q4; colormap = :balance)
heatmap!(axq5, x, y, q5; colormap = :balance)
heatmap!(axq6, x, y, q6; colormap = :balance)
heatmap!(axq7, x, y, q7; colormap = :balance)
heatmap!(axq8, x, y, q8; colormap = :balance)

# contourf!(axpsi1, x, y, psi1;
#           levels = levelspsi1, colormap = :viridis, extendlow = :auto, extendhigh = :auto)
#  contour!(axpsi1, x, y, psi1;
#           levels = levelspsi1⁺, color=:black)
#  contour!(axpsi1, x, y, psi1;
#           levels = levelspsi1⁻, color=:black, linestyle = :dash)

# contourf!(axpsi2, x, y, psi2;
#           levels = levelspsi2, colormap = :viridis, extendlow = :auto, extendhigh = :auto)
#  contour!(axpsi2, x, y, psi2;
#           levels = levelspsi2⁺, color=:black)
#  contour!(axpsi2, x, y, psi2;
#           levels = levelspsi2⁻, color=:black, linestyle = :dash)

# ke₁ = lines!(axKE, KE₁; linewidth = 3)
# ke₂ = lines!(axKE, KE₂; linewidth = 3)
# Legend(fig[1, 4], [ke₁, ke₂,], ["KE₁", "KE₂"])

# lines!(axPE, PE; linewidth = 3)

fig

# trying to run the model now
startwalltime = time()

frames = 0:round(Int, nsteps / nsubs)

record(fig, "multilayerqg_2layer.mp4", frames, framerate = 18) do j
  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, KE₁: %.3e, KE₂: %.3e, PE: %.3e, walltime: %.2f min",
                   clock.step, clock.t, cfl, E.data[E.i][1][1], E.data[E.i][1][2], E.data[E.i][2][1], (time()-startwalltime)/60)

    println(log)
  end

  q1[] = vars.q[:, :, 1]
  q2[] = vars.q[:, :, 2]
  q3[] = vars.q[:, :, 3]
  q4[] = vars.q[:, :, 4]
  q5[] = vars.q[:, :, 5]
  q6[] = vars.q[:, :, 6]
  q7[] = vars.q[:, :, 7]
  q8[] = vars.q[:, :, end]

  # psi1[] = vars.ψ[:, :, 1]
  # psi2[] = vars.ψ[:, :, 2]
  # psi3[] = vars.ψ[:, :, 3]
  # psi4[] = vars.ψ[:, :, 4]
  # psi5[] = vars.ψ[:, :, 5]
  # psi6[] = vars.ψ[:, :, 6]
  # psi7[] = vars.ψ[:, :, 7]
  # psi8[] = vars.ψ[:, :, end]

  # maxpsi1[] = maximum(abs, vars.ψ[:, :, 1])
  # maxpsi2[] = maximum(abs, vars.ψ[:, :, 2])
  # maxpsi3[] = maximum(abs, vars.ψ[:, :, 3])
  # maxpsi4[] = maximum(abs, vars.ψ[:, :, 4])
  # maxpsi5[] = maximum(abs, vars.ψ[:, :, 5])
  # maxpsi6[] = maximum(abs, vars.ψ[:, :, 6])
  # maxpsi7[] = maximum(abs, vars.ψ[:, :, 7])
  # maxpsi8[] = maximum(abs, vars.ψ[:, :, end])


  # KE₁[] = push!(KE₁[], Point2f(μ * E.t[E.i], E.data[E.i][1][1]))
  # KE₂[] = push!(KE₂[], Point2f(μ * E.t[E.i], E.data[E.i][1][end]))
  # PE[]  = push!(PE[] , Point2f(μ * E.t[E.i], E.data[E.i][2][end]))

  # title_KE[] = @sprintf("μ t = %.2f", μ * clock.t)]

  stepforward!(prob, diags, nsubs)
  MultiLayerQG.updatevars!(prob)
  
  # savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), clock.step)
  # savefig(savename)

end


# savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), clock.step)
# savefig(savename)


## use stability functions
# Ny = div(Nx,2)

eta = 0

eve,eva,k_x,k_y,qx,qy = lin_stab(U,V,beta,eta,Nx,Ny,rho,f0,Lx,Ly)

k_xg = range(k_x[1],step=(k_x[2]-k_x[1]),length=length(k_x))
k_yg = range(k_y[1],step=(k_y[2]-k_y[1]),length=length(k_y))



# Plots.contourf(transpose(imag(eva[60:70,60:70])))

# sigma = Observable(Array(imag(eva)))

# fig = Figure(resolution=(1000, 600))

# axis_kwargs = (xlabel = "k_x",
#                ylabel = "k_y",
#                aspect = 1,
#                limits = ((k_x[60], k_x[70]), (k_y[60],k_y[70])))

# ax_sig = Axis(fig[1, 1]; title = "σ", axis_kwargs...)


# contourf!(ax_sig, k_xg, k_yg, sigma;
#           colormap = :viridis, extendlow = :auto, extendhigh = :auto)
#  contour!(ax_sig, k_xg, k_yg, sigma;
#           color=:black)



