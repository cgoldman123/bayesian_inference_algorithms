include("boltzmann_exact.jl")
include("weight_matrix_generators.jl")

using LinearAlgebra

#### 12 Node Diamond Lattice
weight_constant = 1
# weight matrix (columns are inputs to neuron i, as in your code)
big_lattice_matrix = diamond_weight_matrix(2)
big_lattice_weights = big_lattice_matrix .* weight_constant
spectral_radius = maximum(abs.(eigvals(big_lattice_weights)))
big_lattice_weights = big_lattice_weights ./ spectral_radius 
big_lattice_weight_constant = unique(big_lattice_weights[big_lattice_weights .!= 0])[1]

num_neurons = size(big_lattice_weights, 1)

bias = zeros(num_neurons)



# Can leave this NaN to get full joint distribution (no clamping).
# Otherwise, set to 0 or 1 to clamp that neuron.
condition = fill(NaN, num_neurons)
# condition = [NaN, 1.0]  # example: clamp neuron 2 to 1
states, probs = exact_inf_boltzmann(big_lattice_weights, bias,condition)

# Marginalize over neurons
print_marginals(states, probs)
println("\n\n")


#### 4 Node Diamond Lattice
# weight matrix (columns are inputs to neuron i, as in your code)
small_lattice_matrix = diamond_weight_matrix(1)
small_lattice_weights = small_lattice_matrix * .5*log(cosh(4*big_lattice_weight_constant))

num_neurons = size(small_lattice_weights, 1)

bias = zeros(num_neurons)
# Can leave this NaN to get full joint distribution (no clamping).
# Otherwise, set to 0 or 1 to clamp that neuron.
condition = fill(NaN, num_neurons)
# condition = [NaN, 1.0]  # example: clamp neuron 2 to 1
states, probs = exact_inf_boltzmann(small_lattice_weights, bias,condition)

# Marginalize over neurons
print_marginals(states, probs)
