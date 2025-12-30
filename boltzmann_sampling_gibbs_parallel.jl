##################################################
# Parallel Gibbs Sampling for Boltzmann Machines
# Parallel means that samples are drawn for each node at the same time,
# and updates only take effect at the next timestep. Not sure if this is correct 
# Gibbs but it seems to work
##################################################
using StatsFuns   # for logistic
using Random
using Plots
using Statistics

# reproducible RNG
rng = MersenneTwister(13)

# Each neuron is modeled as a Bernoulli random variable 


## Two neurons

# --- parameters ---
num_neurons = 2
num_samples = 10000
# Bias
b = [.9;.1]

# weight matrix (columns are inputs to neuron i, as in your code)
W = [0.0  .4;
     .4 0.0]

# clamp: to clamp neuron i to 0 or 1 set the value of clamp[i] to 0 or 1 respectively
clamp = fill(NaN, num_neurons)

function simulate_sampling_inf_parallel(num_neurons::Integer,
                             W::AbstractMatrix,
                             num_samples::Integer,
                             bias::AbstractVector{<:Real},
                             clamp::AbstractVector{<:Real};
                             rng = MersenneTwister(13))

    # Initialize activation array with neurons off
    activation = fill(NaN, num_neurons,num_samples)
    activation[:,1]=fill(0,num_neurons)

    for t in 1:num_samples-1
        for i in 1:num_neurons
            if !isnan(clamp[i])
                activation[i,t] = clamp[i]
            else
                input_sum = sum(W[:, i] .* activation[:, t])
                # Get the probability of activation=1
                activation_prob = logistic(bias[i] + input_sum)
                # Sample the activation value
                activation[i,t+1] = rand() < activation_prob
            end
        end
        
    end
    return activation

end


activation = simulate_sampling_inf_parallel(num_neurons, W, num_samples, b, clamp; rng = MersenneTwister(13))

## PLOT ##
# running means
running_means = cumsum(activation, dims=2) ./ (collect(1:num_samples)')

# plot
plot()
plot!(running_means[1, :], label="Running mean neuron 1")
plot!(running_means[2, :], label="Running mean neuron 2")
xlabel!("Sample index")
ylabel!("Running mean")
title!("Running average of activation means (Gibbs sampler)")
display(current())

# Calculate means and covariance after burn-in
# Let's get the means 
burn_in = 500
mu1 = mean(activation[1,burn_in:end])
mu2 = mean(activation[2,burn_in:end])

C = cov(activation[:,burn_in:end], dims=2)

println("Boltzman Gibbs' Parallel Sampling\nmu1: $mu1 \nmu2: $mu2")
