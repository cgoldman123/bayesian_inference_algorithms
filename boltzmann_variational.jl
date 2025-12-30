##################################################
# Variational Inference for Boltzmann Machines
##################################################
using StatsFuns   # for logistic
using Random
using Plots

# reproducible RNG
rng = MersenneTwister(13)

# Each neuron is modeled as a Bernoulli random variable with mean activation mu[i]
# Under mean field approximation, all neurons are independent:
# Q(s_i) = product over i neurons: Bernoulli(s_i) = product over i neurons: mu[i]^(s_i) * (1-mu[i])^(1-s_i)

# D(Q||P) = E_Q(s_i) [ log Q(s_i) - log P(s_i) ]




## Two neurons

# --- parameters ---
num_neurons = 2
total_time = 10
# Bias
b = [.2;.1]

# weight matrix (columns are inputs to neuron i, as in your code)
W = [0.0  .4;
     -.8 0.0]

# clamp: to clamp neuron i to 0 or 1 set the value of clamp[i] to 0 or 1 respectively
clamp = fill(NaN, num_neurons)

function simulate_variational_inf(num_neurons::Integer,
                             W::AbstractMatrix,
                             total_time::Integer,
                             bias::AbstractVector{<:Real},
                             clamp::AbstractVector{<:Real};
                             rng = MersenneTwister(13))

    # mu = zeros(num_neurons,total_time)  # mean activations over time
    mu = ones(num_neurons,total_time).*.5  # mean activations over time; initialize at 0.5

    for t in 1:total_time-1
        for i in 1:num_neurons
            if !isnan(clamp[i])
                mu[i,t] = clamp[i]
            else
                input_sum = sum(W[:, i] .* mu[:, t])
                # What if mu[:,t] above gets communicated with a series of samples??
                mu[i,t+1] = logistic(bias[i] + input_sum)
            end
        end
        
    end
    return mu
end

mu = simulate_variational_inf(num_neurons, W, total_time, b, clamp; rng = MersenneTwister(13))
# plot
plot()
plot!(mu[1, :], label="Mean neuron 1")
plot!(mu[2, :], label="Mean neuron 2")
xlabel!("Time Point")
ylabel!("Mean")
title!("Mean of neural activation (Variational Inference)")
display(current())
