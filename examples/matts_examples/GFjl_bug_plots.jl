# showing how two different bug fixes in GeophysicalFlows.jl
# modify vertical profile of interior PV

## load packages

using GeophysicalFlows, CairoMakie, Printf, FFTW, LinearAlgebra, Statistics

here = "/home/matt/Desktop/research/QG/julia_QG/"
include(here*"mjcl_stab.jl")

## build basic model

dev = CPU()     # device (CPU)

# numerical params
Ny = Nx = n = 128         # 2D resolution = n²
stepper = "FilteredRK4"   # time stepping scheme
dt      = 1e-2            # time step
nsteps  = 60000           # total number of time-steps 20000
nsubs   = 50              # number of time-steps for plotting (nsteps must be multiple of nsubs)

# physical params
L       = 2π                   # domain size
μ       = 5e-2                 # bottom drag
beta    = β = 0.               # no gradient of planetary PV

nlayers = 6         # number of layers
f0 = f₀ = 1.        # 0.0001236812857687059 # coriolis param [s^-1] 
g       = 9.81      # gravity

H       = ones(nlayers)                           # even depths
rho = ρ = collect(range(4.0,5.0,nlayers))   # constant N^2
U       = collect(range(1.0,0.0,nlayers))         # constant shear
V       = zeros(nlayers)                          # zonal flow only

## setting up problems
# setting up the problen with bug fixes (denoted ``f'')
prob_f = MultiLayerQG.Problem(nlayers, dev; nx=n, Lx=L, f₀, g, H, ρ, U, μ, β,
                            dt, stepper, aliased_fraction=0)

sol_f, clock_f, params_f, vars_f, grid_f = prob_f.sol, prob_f.clock, prob_f.params, prob_f.vars, prob_f.grid

# setting up the problen with bug fixes (denoted ``sf'')
include(here*"../gfjl/GeophysicalFlows.jl/src/multilayerqg_sign_fix_only.jl")

prob_sf = MultiLayerQG_sf.Problem_sf(nlayers, dev; nx=n, Lx=L, f₀, g, H, ρ, U, μ, β,
                            dt, stepper, aliased_fraction=0)

sol_sf, clock_sf, params_sf, vars_sf, grid_sf = prob_sf.sol, prob_sf.clock, prob_sf.params, prob_sf.vars, prob_sf.grid

# setting up the problen with no bug fixes (denoted ``nf'')
include(here*"../gfjl/GeophysicalFlows.jl/src/multilayerqg_no_fixes.jl")

prob_nf = MultiLayerQG_nf.Problem_nf(nlayers, dev; nx=n, Lx=L, f₀, g, H, ρ, U, μ, β,
                            dt, stepper, aliased_fraction=0)

sol_nf, clock_nf, params_nf, vars_nf, grid_nf = prob_nf.sol, prob_nf.clock, prob_nf.params, prob_nf.vars, prob_nf.grid

# linear stability analysis
Lx, Ly = grid_f.Lx, grid_f.Ly

eta = 0

eve,eva,max_eve_f,max_eva_f,k_x,k_y,qx,qy = lin_stab(U,V,beta,eta,Nx,Ny,rho,f0,Lx,Ly,params_f.Qy)
eve,eva,max_eve_sf,max_eva_sf,k_x,k_y,qx,qy = lin_stab(U,V,beta,eta,Nx,Ny,rho,f0,Lx,Ly,params_sf.Qy)
eve,eva,max_eve_nf,max_eva_nf,k_x,k_y,qx,qy = lin_stab(U,V,beta,eta,Nx,Ny,rho,f0,Lx,Ly,params_nf.Qy)

## plotting differences
z = -cumsum(H)

fig = Figure(resolution=(1000, 600))

ax1_kwargs = (xlabel = "Qy",
               ylabel = "z",
               aspect = 1.)

ax1 = Axis(fig[1, 1]; title = "Qy", ax1_kwargs...)

ylims!(ax1,minimum(z), maximum(z))

lines!(ax1,params_nf.Qy[1,1,:],z,linewidth=3.,label="no fix")
lines!(ax1,params_sf.Qy[1,1,:],z,linewidth=3.,label="sign fix only")
lines!(ax1,params_f.Qy[1,1,:],z,linewidth=3.,label="sign fix and g' fix")

axislegend(position=:lt)

ax2_kwargs = (xlabel = "|ψ|",
               ylabel = "z",
               aspect = 1.)

ax2 = Axis(fig[1, 2]; title = "|ψ|", ax2_kwargs...)

ylims!(ax2,minimum(z), maximum(z))

lines!(ax2,abs.(max_eve_nf),z,linewidth=3.,label="no fix")
lines!(ax2,abs.(max_eve_sf),z,linewidth=3.,label="sign fix only")
lines!(ax2,abs.(max_eve_f),z,linewidth=3.,label="sign fix and g' fix")

axislegend(position=:lt)



