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

Lx, Ly = grid.Lx, grid.Ly

eta = 0

eve,eva,max_eve,max_eva,k_x,k_y,qx,qy = lin_stab(U,V,beta,eta,Nx,Ny,rho,f0,Lx,Ly)

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



