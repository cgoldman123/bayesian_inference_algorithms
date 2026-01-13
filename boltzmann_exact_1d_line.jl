include("boltzmann_exact.jl")
include("weight_matrix_generators.jl")
using LinearAlgebra

normalize_with_spectral_radius = false
#### 4 Node Line
weight_constant = 1.5
# weight matrix (columns are inputs to neuron i, as in your code)
big_lattice_matrix = one_dimensional_line_weight_matrix(5;ring=false)
big_lattice_weights = big_lattice_matrix .* weight_constant

if normalize_with_spectral_radius
    spectral_radius = maximum(abs.(eigvals(big_lattice_weights)))
    big_lattice_weights = big_lattice_weights ./ spectral_radius 
end

big_lattice_weight_constant = unique(big_lattice_weights[big_lattice_weights .!= 0])[1]

num_neurons = size(big_lattice_weights, 1)

bias = zeros(num_neurons)



# Can leave this NaN to get full joint distribution (no clamping).
# Otherwise, set to 0 or 1 to clamp that neuron.
condition = fill(NaN, num_neurons)
# condition = [NaN, 1.0]  # example: clamp neuron 2 to 1
states, probs = exact_inf_boltzmann(big_lattice_weights, bias,condition,signed=true)
println("Configuration probabilities:")
display(probs)
# Marginalize over neurons
print_marginals(states, probs, signed=true)
println("\n\n")


#### 4 Node Diamond Lattice
# weight matrix (columns are inputs to neuron i, as in your code)
small_lattice_matrix = one_dimensional_line_weight_matrix(3;ring=false)
small_lattice_weights = small_lattice_matrix * .5*log(cosh(2*big_lattice_weight_constant))

num_neurons = size(small_lattice_weights, 1)

bias = zeros(num_neurons)
# Can leave this NaN to get full joint distribution (no clamping).
# Otherwise, set to 0 or 1 to clamp that neuron.
condition = fill(NaN, num_neurons)
# condition = [NaN, 1.0]  # example: clamp neuron 2 to 1
states, probs = exact_inf_boltzmann(small_lattice_weights, bias,condition,signed=true)
println("Configuration probabilities:")
display(probs)
# Marginalize over neurons
print_marginals(states, probs, signed=true)