## define stability functions

function lin_stab(U::Vector{Float64},V::Vector{Float64},beta,eta,Nx::Int64,Ny::Int64,rho::Vector{Float64},f0::Float64,Lx::Float64,Ly::Float64)
    # U: (Nx x Nz) vector of zonal mean background velocity
    # V: (Ny x Nz) vector of meridional mean background velocity
    # beta: 
    # 

    Nz = length(rho)
    # define wavenumbers
    k_x = reshape(fftfreq(Nx, 2π/Lx*Nx),(1,Nx))
    k_y = reshape(fftfreq(Ny, 2π/Ly*Ny),(1,Ny))

    # k_x,k_y = wavenumber_grid(Nx,Ny,Lx,Ly)

    # k2 = k_x.^2 + k_y.^2            # this is for an isotropic wavenumber grid only (i.e. Nx=Ny)

    # define stretching matrix
    S = calc_stretching_mat(Nz,rho,f0,H,rho[1])

    # change dimensions of U and V to match domain size
    U2 = zeros(1,Nz); U2[:] = U; U2 = repeat(U2,outer=(Ny,1))
    U = zeros(1,Ny,Nz); U[1,:,:] = U2

    V2 = zeros(1,Nz); V2[:] = V; V2 = repeat(V2,outer=(Nx,1))
    V = zeros(1,Nx,Nz); V[1,:,:] = V2

    # define background QG PV gradients
    Qy = calc_PV_grad_y(U,beta,eta,Ny,Nz,k_y,S)
    Qx = calc_PV_grad_x(V,eta,Nx,Nz,k_x,S)

    # perform linear stability analysis
    evecs_all,evals_all = calc_lin_stab(Qy,Qx,U,V,S,k_x,k_y,Nz)

    # keep largest growth rates per wavenumber
    evecs,evals,max_evec,max_eval = find_growth(evecs_all,evals_all,Nx,Ny,Nz)

    # def rad
    r_d = sqrt(gp(rho[1:2],rho[1])*H[1])/f0

    return fftshift(evecs),fftshift(evals),max_evec,max_eval,fftshift(k_x),fftshift(k_y),mean(Qx[1,:,:],dims=1),mean(Qy[1,:,:],dims=1)
end

function lin_stab(U::Vector{Float64},V::Vector{Float64},beta,eta,Nx::Int64,Ny::Int64,rho::Vector{Float64},f0::Float64,Lx::Float64,Ly::Float64,Qy)
    # Takes Qy as arg
    # U: (Nx x Nz) vector of zonal mean background velocity
    # V: (Ny x Nz) vector of meridional mean background velocity
    # beta: 
    # 

    Nz = length(rho)
    # define wavenumbers
    k_x = reshape(fftfreq(Nx, 2π/Lx*Nx),(1,Nx))
    k_y = reshape(fftfreq(Ny, 2π/Ly*Ny),(1,Ny))

    # k_x,k_y = wavenumber_grid(Nx,Ny,Lx,Ly)

    # k2 = k_x.^2 + k_y.^2            # this is for an isotropic wavenumber grid only (i.e. Nx=Ny)

    # define stretching matrix
    S = calc_stretching_mat(Nz,rho,f0,H,rho[1])

    # change dimensions of U and V to match domain size
    U2 = zeros(1,Nz); U2[:] = U; U2 = repeat(U2,outer=(Ny,1))
    U = zeros(1,Ny,Nz); U[1,:,:] = U2

    V2 = zeros(1,Nz); V2[:] = V; V2 = repeat(V2,outer=(Nx,1))
    V = zeros(1,Nx,Nz); V[1,:,:] = V2

    # define background QG PV gradients (TEMPORARY)
    Qy = reshape(Qy[1,:,:],(1,Ny,Nz))
    Qx = zeros(size(Qy))

    # perform linear stability analysis
    evecs_all,evals_all = calc_lin_stab(Qy,Qx,U,V,S,k_x,k_y,Nz)

    # keep largest growth rates per wavenumber
    evecs,evals,max_evec,max_eval = find_growth(evecs_all,evals_all,Nx,Ny,Nz)

    # def rad
    r_d = sqrt(gp(rho[1:2],rho[1])*H[1])/f0

    return fftshift(evecs),fftshift(evals),max_evec,max_eval,fftshift(k_x),fftshift(k_y),mean(Qx[1,:,:],dims=1),mean(Qy[1,:,:],dims=1)
end

function wavenumber_grid(Nx,Ny,Lx,Ly)
    #
    # nk_x = div(Nx,2)+1; nk_y = div(Ny,2)+1

    nk_x = Nx; nk_y = Ny

    k_x = reshape(LinRange(-2*pi/Lx*nk_x,2*pi/Lx*nk_x,nk_x),(1,Nx))
    k_y = reshape(LinRange(-2*pi/Ly*nk_y,2*pi/Ly*nk_y,nk_y),(1,Ny))

    # k_x = LinRange(0.,2*pi/Lx*nk_x,nk_x)
    # k_y = LinRange(0.,2*pi/Ly*nk_y,nk_y)

    return k_x,k_y
end

function calc_PV_grad_y(U,beta,eta,Ny::Int64,Nz::Int64,k_y,S)
    # calculates PV gradients in one meridional direction
    # U is (Ny x Nz)
    # k_y is (Ny x 1)
    # 

    Uyy = real.(ifft(-k_y.^2 .* fft(U)))

    # Uyy = repeat(Uyy, outer=(Nx, 1, 1))

    # Q_y = zeros(Nx,Nz)

    F = zeros(size(U))
    for i=1:Ny
        F[1,i,:] = S * U[1,i,:]
    end

    Q_y = beta .- (Uyy .+ F)

    return Q_y
end

function calc_PV_grad_x(V,eta,Nx::Int64,Nz::Int64,k_x,S)
    # calculates PV gradients in one zonal direction

    Vxx = real.(ifft(k_x.^2 .* fft(V)))

    # Q_y = zeros(Nx,Nz)

    F = zeros(size(V))
    for i=1:Nx
        F[1,i,:] = S * V[1,i,:]
    end

    Q_x = Vxx .+ F

    return Q_x
end

function gp(rho,rho0)
    g = 9.81
    g_prime = g*(rho[2]-rho[1])/rho0

    return g_prime
end

function calc_stretching_mat(Nz,rho,f0,H,rho0)
    #
    S = zeros((Nz,Nz,))

    alpha = 0

    S[1,1] = -f0^2/H[1]/gp(rho[1:2],rho0) + alpha
    S[1,2] = f0^2/H[1]/gp(rho[1:2],rho0)
    for i = 2:Nz-1
        S[i,i-1] = f0^2/H[i]/gp(rho[i-1:i],rho0)
        S[i,i]   = -(f0^2/H[i]/gp(rho[i:i+1],rho0) + f0^2/H[i]/gp(rho[i-1:i],rho0))
        S[i,i+1] = f0^2/H[i]/gp(rho[i:i+1],rho0)
    end
    S[Nz,Nz-1] = f0^2/H[Nz]/gp(rho[Nz-1:Nz],rho0)
    S[Nz,Nz]   = -f0^2/H[Nz]/gp(rho[Nz-1:Nz],rho0)

    return S
end

function calc_lin_stab(Qy,Qx,U,V,S,k_x,k_y,Nz)
    #
    # A = (k_x .* U .+ k_y .* V)

    # B   = zeros((1,length(k2),Nz))
    # ell = zeros((Nz,Nz,length(k_x),length(k_y)))
    # for i=eachindex(k_x)
    #     for j=eachindex(k_y)
    #         ell[:,:,i,j] = S .- (k_x[i]^2 + k_y[j]^2) * I
    #         B[1,i,j,:] = ell[:,:,i,j] * A[1,i,:]
    #     end
    # end

    # C = k_x .* Qy .- k_y .* Qx

    # M = inv(ell) * (A .+ C)

    # evecs,evals = eig(M)


    #################
    evecs = zeros(Nx,Ny,Nz,Nz) .+ 0im
    evals = zeros(Nx,Ny,Nz) .+ 0im
    k2    = zeros(Nx,Ny)

    for i=1:Nx
        for j=1:Ny
            if i==1 && j==1
                # do nothing
            else
                A = make_diag(k_x[i] .* U[:,j,:] .+ k_y[j] .* V[:,i,:]) 
                # ell_i = inv(S - (k_x[i]^2 + k_y[j]^2)*I)
                # B = ell_i .* make_diag(k_x[i] * Qy[:,j,:] - k_y[j] * Qx[:,i,:]) .+ 0im

                # evecs[i,j,:,:] = eigvecs(A .+ B)
                # evals[i,j,:] = eigvals(A .+ B)

                ell = (S - (k_x[i]^2 + k_y[j]^2) * I)
                A2 = ell * A
                ell_i = inv(ell)
                Q2 = (k_x[i] * Qy[:,j,:] - k_y[j] * Qx[:,i,:]) .+ 0im
                B2 = transpose(ell_i) * transpose(A2 .+ make_diag(Q2)) .+ 0im

                k2[i,j] = (k_x[i]^2 + k_y[j]^2)

                evecs[i,j,:,:] = eigvecs(B2)
                evals[i,j,:] = eigvals(B2)
            end

        end
    end

    return evecs,evals
end

function make_diag(array_in)
    matrix_out = zeros(length(array_in),length(array_in))
    for i=eachindex(array_in)
        matrix_out[i,i] = array_in[i]
    end
    return matrix_out
end

function find_growth(evecs_all,evals_all,Nx,Ny,Nz)
    # 
    evecs = zeros(Nx,Ny,Nz) .+ 0im; evals = zeros(Nx,Ny) .+ 0im

    for i=1:Nx
        for j=1:Ny
            indMax       = argmax(imag(evals_all[i,j,:]))
            evals[i,j]   = evals_all[i,j,indMax]
            evecs[i,j,:] = evecs_all[i,j,:,indMax]
        end
    end

    sigma = imag(evals)

    indMax = argmax(sigma)

    max_eval = sigma[indMax]

    max_evec = abs.(evecs[indMax,:])

    return evecs,evals,max_evec,max_eval
end