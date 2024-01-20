# Tensor product function
The tensor product function is defined as:
``\[ f(x) = ∏ᵢ=₁ᵈ cos(aπxᵢ) \]``

Where\
d: Represents the dimensionality of the input vector x\
xi: Represents the ith components of the input vector\
a: A constant parameter

# Let's import Surrogates and Plots
```@example tensor
using Surrogates
using Plots
using Statistics
using SurrogatesPolyChaos
using SurrogatesRandomForest
default()
```

# Define the function

```@example tensor
function tensor_product_function(x)
    a = 0.5
    return prod(cos.(a*pi*x))
end
```

# Sampling parameters for training and test data
```@example tensor
lb = -5.0  # Lower bound of sampling range
ub = 5.0  # Upper bound of sampling range
n = 30  # Number of training points
```

# Generate training and test data
```@example tensor
x_train = sample(n, lb, ub, SobolSample())  # Sample training data points
y_train = tensor_product_function.(x_train)  # Calculate corresponding function values
x_test = sample(1000, lb, ub, RandomSample())  # Sample larger test data set
y_test = tensor_product_function.(x_test)  # Calculate corresponding true function values
```

# Plot training and testing points
```@example tensor
scatter(x_train, y_train, label="Training Points", xlabel="X-axis", ylabel="Y-axis", legend=:topright)
scatter!(x_test, y_test, label="Testing Points")
```

# Train the following Surrogates: 
## Kriging | Lobachevsky | Radial Basis | RandomForest | Polynomial Chaos
```@example tensor
num_round = 2
alpha = 2.0
n = 6
randomforest_surrogate = RandomForestSurrogate(x_train ,y_train ,lb, ub, num_round = 2)
radial_surrogate = RadialBasis(x_train, y_train, lb, ub)
lobachevsky_surrogate = LobachevskySurrogate(x_train, y_train, lb, ub, alpha = 2.0, n = 6)
kriging_surrogate = Kriging(x_train, y_train, lb, ub)
poly1 = PolynomialChaosSurrogate(x_train,y_train,lb,ub)
poly2 = PolynomialChaosSurrogate(x_train,y_train,lb,ub, op = SurrogatesPolyChaos.GaussOrthoPoly(5))
```

# Obtain predictions from all surrogates for the test data
```@example tensor
loba_pred = lobachevsky_surrogate.(x_test)  
radial_pred = radial_surrogate.(x_test)
kriging_pred = kriging_surrogate.(x_test)
random_forest_pred = randomforest_surrogate.(x_test)
poly1_pred = poly1.(x_test)
poly2_pred = poly2.(x_test)
```

# Define a function to calculate Mean Squared Error (MSE)
```@example tensor
function calculate_mse(predictions, true_values)
    return mean((predictions .- true_values).^2)  # Calculate mean of squared errors
end
```

# Calculate MSE for all Surrogate Models
```@example tensor
mse_loba = calculate_mse(loba_pred, y_test)
mse_krig = calculate_mse(kriging_pred, y_test)
mse_radial = calculate_mse(radial_pred, y_test)
mse_rf = calculate_mse(random_forest_pred, y_test)
mse_poly1 = calculate_mse(poly1_pred, y_test)
mse_poly2 = calculate_mse(poly2_pred, y_test)
```

# Compare the performance of all Surrogate Models
```@example tensor
mse_values = Dict("loba" => mse_loba, "krig" => mse_krig, "radial" => mse_radial, "rf" => mse_rf, "poly1" => mse_poly1, "poly2" => mse_poly2)

# Sort the MSE values in ascending order and display them
sorted_mse = sort(collect(mse_values), by=x->x[2])
for (model, mse) in sorted_mse
    println("$model : $mse")
end
```

# Plot true function vs. model predictions
```@example tensor
xs = lb:0.01:ub
plot(xs, tensor_product_function.(xs), label="True function", legend=:top, color=:black)
plot!(xs, lobachevsky_surrogate.(xs), label="Lobachevsky", legend=:top, color=:red)
plot!(xs, kriging_surrogate.(xs), label="Kriging", legend=:top, color=:blue)
plot!(xs, randomforest_surrogate.(xs), label="Random Forest", legend=:top, color=:green)
plot!(xs, poly1.(xs), label="Polynomial Chaos", legend=:top, color=:purple)
plot!(xs, poly2.(xs), label="Polynomial Chaos", legend=:top, color=:purple)
plot!(xs, radial_surrogate.(xs), label="Radials", legend=:top, color=:orange)
```

# Tabular Representation of all Surrogates and their MSE Scores

| Surrogate Model   | MSE Score            |
|-------------------|----------------------|
| Kriging           | 4.70493378010316e-5 |
| Lobachevsky       | 7.967792682690972e-5|
| Radial Basis      | 0.004972603698976124 |
| RandomForest      | 0.2023233139232778   |
| Poly1             | 0.4124881232761028   |
| Poly2             | 0.42166909818265136  |


This structure provides a framework for generating data, training various surrogate models, evaluating their performance on test data, and reporting the best surrogate based on performance metrics like MSE. Adjustments can made to suit the specific evaluation criteria or additional surrogate models.
