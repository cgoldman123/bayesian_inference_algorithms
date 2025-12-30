##################################################
# Neural sampling in discrete time with absolute refractory period
##################################################
using StatsFuns   # for logistic
using Random
using Plots

# reproducible RNG
rng = MersenneTwister(13)

function simulate_refractory(num_neurons::Integer,
                             W::AbstractMatrix,
                             tau::Integer,
                             total_time::Integer,
                             bias::AbstractVector{<:Real},
                             clamp::AbstractVector{<:Real};
                             rng = MersenneTwister(13))
    # --- state storage ---
    # zeta is the current refractory countdown for each neuron (updated within loop)
    zeta = zeros(num_neurons)
    has_spiked = falses(num_neurons)      # boolean current spike state (per neuron)
    u = zeros(num_neurons, total_time)    # membrane potential over time

    # to store histories for plotting
    zeta_hist = zeros(num_neurons, total_time)
    binary_hist = zeros(Int, num_neurons, total_time)   # 1 when zeta>0 else 0
    running_avg = zeros(num_neurons, total_time)

    # --- simulation loop ---
    for t in 1:total_time
        # Replace has_spiked according to clamp
        has_spiked .= ifelse.(.!isnan.(clamp), clamp .== 1.0, has_spiked)
        for i in 1:num_neurons
            # input sum uses current has_spiked (matches your original code)
            input_sum = sum(W[:, i] .* Float64.(has_spiked))
            u[i, t] += bias[i] + input_sum

            # If zeta >= 1, mark as having spiked (still within refractory countdown)
            if zeta[i] >= 1
                has_spiked[i] = true
            end

            # If not in refractory countdown (zeta <= 1), allow new spike
            if zeta[i] <= 1
                # if clamp is set, use it
                if !isnan(clamp[i])
                    has_spiked[i] = clamp[i] == 1.0
                else
                    p_spike = logistic(u[i, t] - log(tau))
                    has_spiked[i] = rand(rng) < p_spike
                end

                if has_spiked[i]
                    # set refractory timer to tau+1 to match your +1 countdown in Julia code
                    zeta[i] = tau + 1
                end
            end

            # record current zeta for plotting before countdown step
            zeta_hist[i, t] = zeta[i]
            # binary activity is whether zeta>0
            binary_hist[i, t] = zeta[i] > 0 ? 1 : 0

            # countdown zeta (ensure non-negative)
            zeta[i] = max(zeta[i] - 1, 0)
        end

        # update running averages (cumulative mean) at time t
        for i in 1:num_neurons
            # cumulative sum up to t divided by t
            running_avg[i, t] = sum(binary_hist[i, 1:t]) / t
        end
    end
    final_rates = running_avg[:, end]


    return (zeta_hist = zeta_hist,
            binary_hist = binary_hist,
            running_avg = running_avg,
            u = u,
            final_rates = final_rates)
end

function make_plots(sim)
    # --- plotting ---
    zeta_hist = sim.zeta_hist
    binary_hist = sim.binary_hist
    running_avg = sim.running_avg
    final_rates = sim.final_rates
    membrane_potentials = sim.u
    println("Final running-average firing rates: ", final_rates)

    default(titlefontsize = 14, guidefontsize = 12, tickfontsize = 10, legendfontsize = 10)

    time = 1:total_time

    # --- Plot 1: zeta over time ---
    p1 = plot(time, zeta_hist[1, :], label = "neuron 1", lw = 1.5)
    for i in 2:num_neurons
        plot!(p1, time, zeta_hist[i, :], label = "neuron $(i)", lw = 1.5)
    end
    xlabel!(p1, "Time")
    ylabel!(p1, "zeta (refractory timer)")
    title!(p1, "Refractory timer (zeta) over time")

    # --- Plot 2: binary activity ---
    p2 = plot()
    for i in 1:num_neurons
        plot!(p2, time, binary_hist[i, :], label = "neuron $(i)",
            lw = 1.2, seriestype = :steppost)
    end
    xlabel!(p2, "Time")
    ylabel!(p2, "binary (zeta>0)")
    ylims!(p2, (-0.1, 1.1))    # ✅ correct form
    title!(p2, "Binary activity (1 if zeta>0, else 0) over time")

    # --- Plot 3: running average ---
    p3 = plot()
    for i in 1:num_neurons
        plot!(p3, time, running_avg[i, :], label = "neuron $(i)", lw = 1.5)
    end
    xlabel!(p3, "Time")
    ylabel!(p3, "Running average (cumulative mean)")
    title!(p3, "Running average (cumulative mean) of binary activity")

    # Combine vertically
    plot(p1, p2, p3, layout = (3, 1), size = (1000, 1100))


    # ------- coincidence detection between neuron indices a and b -------
    a, b = 1, 2                     # which two neurons to test (z1 and z2)
    coincidence = (binary_hist[a, :] .== 1) .& (binary_hist[b, :] .== 1)
    coincidence_int = Int.(coincidence)   # 1 when both 1, else 0

    # running (cumulative) average of coincidence
    counts = collect(1:total_time)
    cumsum_coinc = cumsum(coincidence_int)
    running_avg_coinc = cumsum_coinc ./ counts

    # ------- Plot 4: coincidence (binary) ------
    p4 = plot(title = "Coincidence (z$(a) & z$(b)) over time",
            xlabel = "Time", ylabel = "coincidence (both=1)")
    plot!(p4, time, coincidence_int, seriestype = :steppost, label = "z$(a)&z$(b)", lw = 1.2)
    ylims!(p4, (-0.1, 1.1))

    # ------- Plot 5: running average of coincidence ------
    p5 = plot(title = "Running average (cumulative mean) of coincidence z$(a)&z$(b)",
            xlabel = "Time", ylabel = "running average")
    plot!(p5, time, running_avg_coinc, label = "running avg (z$(a)&z$(b))", lw = 1.5)
    ylims!(p5, (0.0, 1.0))

    # ------- Combine everything: p1..p5 stacked vertically -------
    # If you already have p1,p2,p3 from before, combine them with p4,p5:
    plot(p1, p2, p3, p4, p5, layout = (5, 1), size = (1000, 1400))
end
## Two neurons

# --- parameters ---
num_neurons = 2
tau = 40                   # refractory period length
total_time = 5000
bias = 2
b = fill(bias, num_neurons)

# weight matrix (columns are inputs to neuron i, as in your code)
W = [0.0  1.0;
     -1.0 0.0]

# clamp: to clamp neuron i to 0 or 1 set the value of clamp[i] to 0 or 1 respectively
clamp = fill(NaN, num_neurons)
clamp[1] = 0

sim = simulate_refractory(num_neurons, W, tau, total_time, b, clamp; rng = MersenneTwister(13))
# extract outputs
make_plots(sim)


## Three neurons

# --- parameters ---
num_neurons = 3
tau = 8                   # refractory period length
total_time = 5000
bias = .5
b = fill(bias, num_neurons)

# weight matrix (columns are inputs to neuron i, as in your code)
W = [0.0  1.0 -1.0;
     1.0 0.0  1.0;
      1.0 -1.0 0.0]

# clamp: to clamp neuron i to 0 or 1 set the value of clamp[i] to 0 or 1 respectively
clamp = fill(NaN, num_neurons)
# clamp[1] = 0

sim = simulate_refractory(num_neurons, W, tau, total_time, b, clamp; rng = MersenneTwister(13))
# extract outputs

