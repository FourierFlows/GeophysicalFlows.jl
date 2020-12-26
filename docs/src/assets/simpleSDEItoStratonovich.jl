using Plots

using Statistics: mean
using Random: randn, seed!

seed!(1234)

μ = 0.2
σ = 1/5
dt = 0.01
nsteps = 2001
T = 0:dt:(nsteps-1)*dt

# Theoretical results
nens = 1000
ΔW = sqrt(σ) * randn(nsteps, nens)/sqrt(dt)

E_theory = @. σ/4μ * (1 - exp(-2μ*T))
dEdt_theory = @. σ/2  * exp(-2μ*T)

# Numerical calculation
X = zeros(size(ΔW))
E_ito = zeros(size(ΔW))
E_str = zeros(size(ΔW))
E_numerical = zeros(size(ΔW))

for j = 1:nsteps-1 # time step the equation
  @views @. X[j+1, :] = X[j, :] + (-μ*X[j, :] + ΔW[j, :])*dt

  @views @. E_ito[j+1, :] = E_ito[j, :] + (-2*μ*E_ito[j, :] + σ/2)*dt + X[j, :]*ΔW[j, :]*dt

  Ebar = @. E_str[j, :] + (-2*μ*E_str[j, :])*dt + X[j, :]*ΔW[j, :]*dt
  @views @. E_str[j+1, :] = E_str[j, :] + (-2*μ*(0.5*(E_str[j, :] +
                        Ebar)))*dt + (0.5*(X[j, :]+X[j+1, :]))*ΔW[j, :]*dt
end

# Energy
@views @. E_numerical = 0.5 * X^2

# compute dE/dt numerically
dEdt_ito = mean((E_ito[2:nsteps, :] - E_ito[1:nsteps-1, :])/dt, dims=2)
dEdt_str = mean((E_str[2:nsteps, :] - E_str[1:nsteps-1, :])/dt, dims=2)

# compute the work and dissipation
work_ito = mean(ΔW[1:nsteps-1, :] .* X[1:nsteps-1, :], dims=2) .+ σ/2
work_str = mean(ΔW[1:nsteps-1, :] .* (X[1:nsteps-1, :] .+ X[2:nsteps, :])/2, dims=2)
diss_ito = 2*μ * (mean(E_ito[1:nsteps-1, :], dims=2))
diss_str = 2*μ * (mean(E_str[1:nsteps-1, :], dims=2))


# Make plots: compare E(t) evolution Ito, Stratonovich, direct 0.5*x^2

plot(μ*T, [E_numerical[:, 1] E_ito[:, 1] E_str[:, 1]],
          linewidth = [3 2 1],
          label = ["½ xₜ²" "Eₜ (Ito)" "Eₜ (Stratonovich) "],
          linestyle = [:solid :dash :dashdot],
          xlabel = "μ t",
          ylabel = "E",
          legend = :topleft,
           title = "comparison of E(t) for single realization")

savefig("energy_comparison.png")


# Make plots: energy budgets for a realization of the Ito integration
titlestring = stochastic ?  "Ito: 𝖽Eₜ = (-2μ Eₜ + ½σ)𝖽t + √σ Xₜ 𝖽W" : "Ito: 𝖽X/𝖽t = -μ X + √σ"

plot_E = plot(μ*T, [E_theory mean(E_ito, dims=2)],
        linewidth = [3 2],
        linestyle = [:solid :dash],
        label=["theoretical ⟨E⟩" "⟨E⟩ from $nens ensemble member(s)"],
        xlabel = "μ t",
        ylabel = "E",
        legend = :bottomright,
         title = titlestring)

plot_Ebudget = plot(μ*T[1:nsteps-1], [dEdt_ito[1:nsteps-1, 1] work_ito[1:nsteps-1, 1]-diss_ito[1:nsteps-1, 1] dEdt_theory[1:nsteps-1]],
                linestyle = [:dash :dashdot :solid],
                linewidth = [2 1 3],
                    label = ["numerical 𝖽⟨E⟩/𝖽t" "⟨work - dissipation⟩" "numerical 𝖽⟨E⟩/𝖽t"],
                   legend = :bottomleft,
                   xlabel = "μ t")

plot(plot_E, plot_Ebudget, layout=(2, 1))

savefig("energy_budgets_Ito.png")


# Make plots: energy budgets for a realization of the Stratonovich integration
titlestring = stochastic ?  "Stratonovich: 𝖽Eₜ = (-2μ Eₜ + ½σ)𝖽t + √σ xₜ∘𝖽W" : "Stratonovich: 𝖽E/𝖽t = -2μ E + √σ ẋ"

plot_E = plot(μ*T, [E_theory mean(E_str, dims=2)],
          linewidth = [3 2],
              label = ["theoretical ⟨E⟩" "⟨E⟩ from $nens ensemble member(s)"],
          linestyle = [:solid :dash],
             xlabel = "μ t",
             ylabel = "E",
             legend = :bottomright,
              title = titlestring)

plot_Ebudget = plot(μ*T[1:nsteps-1], [dEdt_str[1:nsteps-1] work_str[1:nsteps-1]-diss_str[1:nsteps-1] dEdt_theory[1:nsteps-1]],
                linestyle = [:dash :dashdot :solid],
                linewidth = [2 1 3],
                    label = ["numerical 𝖽⟨E⟩/𝖽t" "⟨work - dissipation⟩" "theoretical 𝖽⟨E⟩/𝖽t"],
                   legend = :bottomleft,
                   xlabel = "μ t")

plot(plot_E, plot_Ebudget, layout=(2, 1))

savefig("energy_budgets_Stratonovich.png")
