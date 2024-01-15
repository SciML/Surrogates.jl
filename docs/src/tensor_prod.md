# Tensor product function
The tensor product function is defined as:
``\[ f(x) = ∏ᵢ=₁ᵈ cos(aπxᵢ) \]``

Where\
d: Represents the dimensionality of the input vector x\
xi: Represents the ith components of the input vector\
a: A constant parameter

Let's import Surrogates and Plots
```@example tensor
using Surrogates
using Plots
using Statistics
default()
```

Generating Data and Plotting

```@example tensor
function tensor_product_function(x)
    a = 0.5
    return prod(cos.(a*pi*x))
end
```

Sampling parameters for training and test data
```@example tensor
lb = -5.0  # Lower bound of sampling range
ub = 5.0  # Upper bound of sampling range
n = 30  # Number of training points
```

Generate training and test data
```@example tensor
x_train = sample(n, lb, ub, SobolSample())  # Sample training data points
y_train = tensor_product_function.(x_train)  # Calculate corresponding function values
x_test = sample(1000, lb, ub, RandomSample())  # Sample larger test data set
y_test = tensor_product_function.(x_test)  # Calculate corresponding true function values
```

Train two surrogates: Lobachevsky and Kriging
```@example tensor
loba_surrogate = LobachevskySurrogate(x_train, y_train, lb, ub)  # Train Lobachevsky surrogate
krig_surrogate = Kriging(x_train, y_train, lb, ub)  # Train Kriging surrogate
```

Obtain predictions from both surrogates for the test data
```@example tensor
loba_pred = loba_surrogate.(x_test)  # Predict using Lobachevsky surrogate
krig_pred = krig_surrogate.(x_test)  # Predict using Kriging surrogate
```

Define a function to calculate Mean Squared Error (MSE)
```@example tensor
function calculate_mse(predictions, true_values)
    return mean((predictions .- true_values).^2)  # Calculate mean of squared errors
end
```

Calculate MSE for both surrogates
```@example tensor
mse_loba = calculate_mse(loba_pred, y_test)  # Calculate Lobachevsky's MSE
mse_krig = calculate_mse(krig_pred, y_test)  # Calculate Kriging's MSE
```

Compare performance and print best-performing surrogate based on MSE
```@example tensor
if mse_loba < mse_krig
    println("Lobachevsky Surrogate is the best with MSE: ", mse_loba)
else
    println("Kriging Surrogate is the best with MSE: ", mse_krig)
end
```

Plot true function vs. model predictions
```@example tensor
xs = lb:0.01:ub
plot(xs, tensor_product_function.(xs), label="True function", legend=:top, color=:black)
plot!(xs, loba_surrogate.(xs), label="Lobachevsky", legend=:top, color=:red)
plot!(xs, krig_surrogate.(xs), label="Kriging", legend=:top, color=:blue)
```
This structure provides a framework for generating data, training various surrogate models, evaluating their performance on test data, and reporting the best surrogate based on performance metrics like MSE. Adjustments can made to suit the specific evaluation criteria or additional surrogate models.
