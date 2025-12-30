using RxInfer, Distributions, Random

# Random number generator for reproducibility
rng            = MersenneTwister(42)
# Number of coin flips (observations)
n_observations = 10
# The bias of a coin used in the demonstration
coin_bias      = 0.75
# We assume that the outcome of each coin flip is
# distributed as the `Bernoulli` distrinution
distribution   = Bernoulli(coin_bias)
# Simulated coin flips
dataset        = rand(rng, distribution, n_observations)

# GraphPPL.jl export `@model` macro for model specification
# It accepts a regular Julia function and builds an FFG under the hood
@model function coin_model(y, a, b)
    # We endow θ parameter of our model with some prior
    θ ~ Beta(a, b)
    # or, in this particular case, the `Uniform(0.0, 1.0)` prior also works:
    # θ ~ Uniform(0.0, 1.0)

    # We assume that outcome of each coin flip is governed by the Bernoulli distribution
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end

@model function coin_model(y, a, b)
    θ  ~ Beta(a, b)
    y .~ Bernoulli(θ)
end

result = infer(
    model = coin_model(a = 2.0, b = 7.0),
    data  = (y = dataset, )
)