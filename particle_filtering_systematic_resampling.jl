### SEQUENTIAL IMPORTANCE RESAMPLING ###

using Random, Distributions, Plots, Statistics

T = 30
x = fill(NaN,T)
y = fill(NaN,T)
x[1] = 5 

process_noise = 5
obs_noise = 5
for i in 1:(T-1)
    x[i+1] = rand(Normal(x[i],sqrt(process_noise)))
    y[i+1] = rand(Normal(x[i+1],sqrt(obs_noise)))
end

# p(x_t | x_{t-1}) = N(x_{t-1}, 10)
# p(y_t | x_t) = N(x_t, 5)

importance_function = "optimal" # Specify importance function as "prior" or "optimal"


num_samples = 1000
prior_var = 5
likelihood_var = 5

# Initialize samples 
samples = fill(NaN,num_samples,T)
samples[:,1] = rand(Normal(y[2], sqrt(prior_var)), num_samples)
# Initialize weight functions
p_y_given_x = fill(NaN,num_samples,T)
p_y_given_x_t_minus_1 = fill(NaN,num_samples,T)
# Initialize weights
weights = fill(NaN,num_samples,T)
weights[:,1] .= 1/num_samples
# Initialized resampled x
resampled_x = fill(NaN,num_samples,T)

for t in 2:T
    for i in 1:num_samples
        #### Step 1: Sample from importance function ####
        if importance_function == "optimal"
            # Importance function: q(x_t | x_{t-1}^i)
            importance_var = prior_var
            importance_mean = samples[i,t-1]
            samples[i,t] = rand(Normal(importance_mean,sqrt(importance_var)))
            #### Step 3: Update the weight of the sample ####
            # Calculate the probability of the data given the sample: p(y_t | x_t)
            p_y_given_x[i,t] = pdf(Normal(samples[i,t],sqrt(likelihood_var)), y[t])
            weights[i,t] = p_y_given_x[i,t]*weights[i,t-1]
        else
            # Importance function:  q(x_t | x_{t-1},y_{t})
            importance_var = 1/((1/prior_var)+(1/likelihood_var))
            importance_mean = importance_var*((samples[i,t-1]/prior_var)+(y[t]/likelihood_var))
            samples[i,t] = rand(Normal(importance_mean,sqrt(importance_var)))
            #### Step 3: Update the weight of the sample ####
            # Calculate the probability of the data given the previous sample: p(y_t | x_{t-1})
            #p_y_given_x_t_minus_1[i,t] = exp(-.5*(((samples[i,t-1]^2)/prior_var) + ((y[t]^2)/likelihood_var) - ((((samples[i,t-1]/prior_var) + (y[t]/likelihood_var))^2)/importance_var)))
            p_y_given_x_t_minus_1[i,t] = exp(-.5*((y[t]-samples[i,t-1])^2)*((prior_var + likelihood_var)^(-1)))

            weights[i,t] = p_y_given_x_t_minus_1[i,t]*weights[i,t-1]
        end
    end
    # Renormalize weights
    weight_sum = sum(weights[:,t])
    weights[:,t] = weights[:,t]./weight_sum

    # Resample if the effective sample size is really small
    effective_sample_size = 1/(sum(weights[:,t].^2))
    #### Systematic Resampling ####
    if effective_sample_size < (num_samples/2)
        # Systematic resampling
        # Get cumulative sum of weights
        cum_weights = cumsum(weights[:,t])
        # resample x from weighted distribution
        for i in 1:num_samples
            rand_num = rand()
            # Get index of weights greater than or equal to rand
            weight_indices = findall(cum_weights .>= rand_num)
            # Identify the particle that was selected 
            weight_index = weight_indices[1]
            # Get resampled x
            resampled_x[i,t] = samples[weight_index,t]
        end

        n = 0
        m = 1
        u0 = rand()*(1/num_samples)
        while(n < num_samples)
            u = u0 + n/num_samples
            while (cum_weights[m] < u)
                m = m+1
            end
            n = n+1
            resampled_x[n,t] = samples[m,t]
        end

        # Reassign samples
        samples[:,t] = resampled_x[:,t]
    end
end



#### PLOTTING

time = 1:T
means = [ sum(weights[:,t].*samples[:,t]) for t in 1:T]
vars  = [ sum(weights[:,t].*((samples[:,t] .- fill(means[t],num_samples)).^2)) for t in 1:T]
stds = vars.^.5

plot(time, means;
     ribbon = stds,
     lw = 2, color = :blue,
     xlabel = "t", ylabel = "x",
     title = "Inferred vs True x",
     label = "Inferred ±1SD",
     legend = :topright)

plot!(time, x[1:T];
      lw = 2, ls = :dash, marker = :circle, color = :red,
      label = "True x")

