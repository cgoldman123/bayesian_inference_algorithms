###########
# Exact Inference for Boltzmann Machines
##############
using StatsFuns   # for logistic

function exact_inf_boltzmann(weights::AbstractMatrix,
                             bias::AbstractVector{<:Real},
                             condition::AbstractVector{<:Real}=fill(NaN, size(weights, 1)))
    num_neurons = size(weights, 1)
    num_states = 2^num_neurons
    states = zeros(Int, num_neurons, num_states)

    # Generate all possible binary states
    for s in 0:num_states-1
        for i in 1:num_neurons
            states[i, s+1] = (s >> (i-1)) & 1
        end
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
                              marginalize_neurons::AbstractVector{Bool})
    num_neurons_remaining = size(marginalize_neurons,1) - sum(marginalize_neurons)

    num_states_marg = 2^num_neurons_remaining
    states_marg = zeros(Int, num_neurons_remaining, num_states_marg)

    # Generate all possible binary states
    for s in 0:num_states_marg-1
        for i in 1:num_neurons_remaining
            states_marg[i, s+1] = (s >> (i-1)) & 1
        end
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


#### 4 Node Boltzmann Machine Example
# weight matrix (columns are inputs to neuron i, as in your code)
weights = [ 0.         -1  0   -1;
        -1         0   1  0   ;
        0         1  0   1  ;
        -1         0   1  0   ;]
weights = weights * -4 * log(exp(2)+exp(-2))

num_neurons = size(weights, 1)

bias = [0;0;0;0;]


# Can leave this NaN to get full joint distribution (no clamping).
# Otherwise, set to 0 or 1 to clamp that neuron.
condition = fill(NaN, num_neurons)
# condition = [NaN, 1.0]  # example: clamp neuron 2 to 1
states, probs = exact_inf_boltzmann(weights, bias,condition)

# Marginalize over neurons
# Specify which neurons to marginalize over in a num_neurons x 1 vector
marginalize_neurons = [true; false; true;true]  # example: marginal
states_marg, probs_marg = marg_neurons(states, probs, marginalize_neurons)






#### 2 Node Boltzmann Machine Example

weights = [ 0.         -2*log(2);
        -2*log(2)         0   ;]
weights = [log(cosh(2))         -log(cosh(2));
        -log(cosh(2))         log(cosh(2));]

num_neurons = size(weights, 1)

bias = [0;0;]


# Can leave this NaN to get full joint distribution (no clamping).
# Otherwise, set to 0 or 1 to clamp that neuron.
condition = fill(NaN, num_neurons)
# condition = [NaN, 1.0]  # example: clamp neuron 2 to 1
states, probs = exact_inf_boltzmann(weights, bias,condition)

# Marginalize over neurons
# Specify which neurons to marginalize over in a num_neurons x 1 vector
marginalize_neurons = [false; true]  # example: marginal
states_marg, probs_marg = marg_neurons(states, probs, marginalize_neurons)




#### 16 Node Boltzmann Machine Example
# weight matrix (columns are inputs to neuron i, as in your code)
weights16 = zeros(Int, 12, 12)

# Bottom -1 couplings
weights16[1, 2] = -1; weights16[2, 1] = -1
weights16[2, 3] = -1; weights16[3, 2] = -1
weights16[1, 4] = -1; weights16[4, 1] = -1
weights16[4, 3] = -1; weights16[3, 4] = -1
weights16[1, 11] = -1; weights16[11, 1] = -1
weights16[10, 11] = -1; weights16[11, 10] = -1
weights16[10, 12] = -1; weights16[12, 10] = -1
weights16[12, 1] = -1; weights16[1, 12] = -1


# Top +1 couplings
weights16[3, 5] = +1; weights16[5, 3] = +1
weights16[3, 6] = +1; weights16[6, 3] = +1
weights16[5, 7] = +1; weights16[7, 5] = +1
weights16[7, 6] = +1; weights16[6, 7] = +1
weights16[7, 8] = +1; weights16[8, 7] = +1
weights16[7, 9] = +1; weights16[9, 7] = +1
weights16[8, 10] = +1; weights16[10, 8] = +1
weights16[9, 10] = +1; weights16[10, 9] = +1

weights = weights16

num_neurons = size(weights, 1)

bias = [0;0;0;0;0;0;0;0;0;0;0;0;]


# Can leave this NaN to get full joint distribution (no clamping).
# Otherwise, set to 0 or 1 to clamp that neuron.
condition = fill(NaN, num_neurons)
# condition = [NaN, 1.0]  # example: clamp neuron 2 to 1
states, probs = exact_inf_boltzmann(weights, bias,condition)

# Marginalize over neurons
# Specify which neurons to marginalize over in a num_neurons x 1 vector
marginalize_neurons = trues(12)  # trues(n) creates a Bool array of length n filled with true

# Set the 1st, 3rd, 7th, and 10th elements to false
marginalize_neurons[[1, 3, 7, 10]] .= false
states_marg, probs_marg = marg_neurons(states, probs, marginalize_neurons)






#### 4 Node Boltzmann Machine Example
# weight matrix (columns are inputs to neuron i, as in your code)
weights = [ 0.         1  0   1;
        1         0   1  0   ;
        0         1  0   1  ;
        1         0   1  0   ;]
weights = weights*2
num_neurons = size(weights, 1)

bias = [0;0;0;0;]


# Can leave this NaN to get full joint distribution (no clamping).
# Otherwise, set to 0 or 1 to clamp that neuron.
condition = fill(NaN, num_neurons)
# condition = [NaN, 1.0]  # example: clamp neuron 2 to 1
states, probs = exact_inf_boltzmann(weights, bias,condition)

# Marginalize over neurons
# Specify which neurons to marginalize over in a num_neurons x 1 vector
marginalize_neurons = [true; true; false;true]  # example: marginal
states_marg, probs_marg = marg_neurons(states, probs, marginalize_neurons)






#### 2 Node Boltzmann Machine Example

weights = [ 0.         log(cosh(4*2)) ;
        log(cosh(4*2))         0   ;]


num_neurons = size(weights, 1)

bias = [0;0;]


# Can leave this NaN to get full joint distribution (no clamping).
# Otherwise, set to 0 or 1 to clamp that neuron.
condition = fill(NaN, num_neurons)
# condition = [NaN, 1.0]  # example: clamp neuron 2 to 1
states, probs = exact_inf_boltzmann(weights, bias,condition)

# Marginalize over neurons
# Specify which neurons to marginalize over in a num_neurons x 1 vector
marginalize_neurons = [false; true]  # example: marginal
states_marg, probs_marg = marg_neurons(states, probs, marginalize_neurons)

