# Given prior: p(a,c)=1 for (a,c > 0)
# Given likelihood: p(x|a,c)= a*(c^a)/(x^(a+1)) for (x > c)
using Plots



# Generate data
a = 40
c = 30
n = 1000
# Set random seed
rng = Xoshiro(1234)

# For the likelihood, we can derive the CDF and then use the inverse to sample from a U(0,1)
u = rand(rng,n)
x = c./(u.^((1/a)))

histogram(x)

# Let's see if we can recover these parameters using Gibb's Sampling

