###########
# Exact Inference for Boltzmann Machines
##############
using StatsFuns   # for logistic

function exact_inf_boltzmann(weights::AbstractMatrix,
                             bias::AbstractVector{<:Real},
                             condition::AbstractVector{<:Real}=fill(NaN, size(weights, 1)),
                             ;signed::Bool=false)
    num_neurons = size(weights, 1)
    num_states = 2^num_neurons
    states = zeros(Int, num_neurons, num_states)

    # Generate all possible binary states
    for s in 0:num_states-1
        for i in 1:num_neurons
            states[i, s+1] = (s >> (i-1)) & 1
        end
    end
    if signed
        # Convert from {0,1} to {-1,1}
        states = states .* 2 .- 1
    end

    # Compute unnormalized probabilities
    unnorm_probs = zeros(Float64, num_states)
    for s in 1:num_states
        energy = -0.5 * states[:, s]' * weights * states[:, s] - bias' * states[:, s]
        unnorm_probs[s] = exp(-energy)
    end


    # Normalize to get probabilities over the full joint
    Z = sum(unnorm_probs)
    probs = unnorm_probs / Z

    # Condition on clamped neurons if needed
    mask = all((isnan.(condition) .| (states .== condition)), dims=1)
    conditioned_states = states[:, vec(mask)]
    conditioned_probs = probs[vec(mask)]
    conditioned_probs /= sum(conditioned_probs)  # renormalize

    return conditioned_states, conditioned_probs
end

function marg_neurons(states::AbstractMatrix,
                              probs::AbstractVector{<:Real},
                              marginalize_neurons::AbstractVector{Bool};
                              signed::Bool=false)
    num_neurons_remaining = size(marginalize_neurons,1) - sum(marginalize_neurons)

    num_states_marg = 2^num_neurons_remaining
    states_marg = zeros(Int, num_neurons_remaining, num_states_marg)

    # Generate all possible binary states
    for s in 0:num_states_marg-1
        for i in 1:num_neurons_remaining
            states_marg[i, s+1] = (s >> (i-1)) & 1
        end
    end

    if signed
        # Convert from {0,1} to {-1,1}
        states_marg = states_marg .* 2 .- 1
    end

    full_states_for_neurons_not_marg = states[.!marginalize_neurons, :]
    unnorm_probs = zeros(Float64, num_states_marg)
    for s in 1:num_states_marg
        M = states_marg[:,s] .== full_states_for_neurons_not_marg
        cols = findall(col -> all(col), eachcol(M))
        unnorm_probs[s] = sum(probs[cols])
    end
    Z = sum(unnorm_probs)
    probs_marg = unnorm_probs / Z
    return states_marg, probs_marg
end




function print_marginals(states, probs; signed::Bool=false)
    num_neurons = size(states, 1)
    for i in 1:num_neurons
        marginalize_neurons = trues(num_neurons)
        marginalize_neurons[i] = false
        states_marg, probs_marg = marg_neurons(states, probs, marginalize_neurons, signed=signed)
        # print result
        println(
            "Neuron $i marginal probability of being 1: ",
            probs_marg[2]
        )
    end
end
