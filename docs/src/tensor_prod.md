# Tensor product function
The tensor product function is defined as:
``\[ f(x) = ∏ᵢ=₁ᵈ cos(aπxᵢ) \]``

Where\
d: Represents the dimensionality of the input vector x\
xi: Represents the ith components of the input vector\
a: A constant parameter

Let's import Surrogates and Plots
```
using Surrogates
using Plots
default()
```

# Generating Data and Plotting

```@example tensor
function tensor_product_function(x, a)
    return prod(cos.(a * π * xi) for xi in x)
end

# Generate training and test data
function generate_data(n, lb, ub, a)
    x_train = sample(n, lb, ub, SobolSample())
    y_train = tensor_product_function(x_train, a)
    
    x_test = sample(1000, lb, ub, RandomSample())  # Generating test data
    y_test = tensor_product_function(x_test, a)  # Generating test labels
    
    return x_train, y_train, x_test, y_test
end

# Visualize training data and the true function
function plot_data_and_true_function(x_train, y_train, x_test, y_test, a, lb, ub)
    xs = range(lb, ub, length=1000)
    plot(xs, tensor_product_function.(xs, a), label="True Function", legend=:top)
    scatter!(x_train, repeat([y_train], length(x_train)), label="Training Points", xlims=(lb,ub), ylims=(-1,1))
    scatter!(x_test, repeat([y_test], length(x_test)), label="Test Points")
end

# Generate data and plot
n = 30
lb = -5.0
ub = 5.0
a = 0.5

x_train, y_train, x_test, y_test = generate_data(n, lb, ub, a)
plot_data_and_true_function(x_train, y_train, x_test, y_test, a, lb, ub)
```

# Training various Surrogates
Now let's train various surrogate models and evaluate their performance on the test data

```@example tensor
# Train different surrogate models
function train_surrogates(x_train, y_train, lb, ub, alpha=2.0, n=6)
    loba = LobachevskySurrogate(x_train, y_train, lb, ub, alpha=alpha, n=n)
    krig = Kriging(x_train, y_train, lb, ub)
    return loba, krig
end

# Evaluate and compare surrogate model performances
function evaluate_surrogates(loba, krig, x_test)
    loba_pred = loba.(x_test)
    krig_pred = krig.(x_test)
    return loba_pred, krig_pred
end

# Plot surrogate predictions against the true function
function plot_surrogate_predictions(loba_pred, krig_pred, x_test, y_test, a, lb, ub)
    xs = collect(x_test)  # Convert x_test to an array
    plot(xs, tensor_product_function.(xs, a), label="True Function", legend=:top)
    plot!(collect(x_test), loba_pred, seriestype=:scatter, label="Lobachevsky")
    plot!(collect(x_test), krig_pred, seriestype=:scatter, label="Kriging")
    plot!(collect(x_test), fill(y_test, length(x_test)), seriestype=:scatter, label="Sampled points")  # Use fill to create an array of the same length as x_test
end

# Train surrogates and evaluate their performance
lb, ub = minimum(x_train), maximum(x_train)
loba, krig = train_surrogates(x_train, y_train, lb, ub)
loba_pred, krig_pred = evaluate_surrogates(loba, krig, x_test)

# Plotting Results
plot_surrogate_predictions(loba_pred, krig_pred, x_test, y_test, 2.0, lb, ub)
```

# Reporting the best Surrogate Model
To determine the best surrogate, you can compare their accuracy and performance metrics on the test data. For instance, you can calculate and compare the mean squared error (MSE) or any other relevant metric

```@example tensor
using Statistics

# Evaluate performance metrics
function calculate_performance_metrics(pred, true_vals)
    return mean((pred .- true_vals).^2)
end

# Compare surrogate model performances
mse_loba = calculate_performance_metrics(loba_pred, y_test)
mse_krig = calculate_performance_metrics(krig_pred, y_test)

if mse_loba < mse_krig
    println("Lobachevsky Surrogate is the best with MSE: ", mse_loba)
else
    println("Kriging Surrogate is the best with MSE: ", mse_krig)
end
```

This structure provides a framework for generating data, training various 
surrogate models, evaluating their performance on test data, and reporting 
the best surrogate based on performance metrics like MSE. Adjustments can made to suit the specific evaluation criteria or additional surrogate models.
