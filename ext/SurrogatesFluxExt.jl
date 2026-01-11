module SurrogatesFluxExt

using Surrogates: NeuralSurrogate, Surrogates, GENNSurrogate
using SurrogatesBase
using Flux: Flux.Optimisers
using Flux
using Zygote
using LinearAlgebra
using Statistics

export predict_derivative

"""
    NeuralSurrogate(x, y, lb, ub; model = Chain(Dense(length(x[1]), 1), first), 
                                 loss = Flux.mse, 
                                 opt = Optimisers.Adam(1e-3), 
                                 n_epochs = 10)

## Arguments

  - `x`: Input data points.
  - `y`: Output data points.
  - `lb`: Lower bound of input data points.
  - `ub`: Upper bound of output data points.

# Keyword Arguments

  - `model`: Flux Chain
  - `loss`: loss function from minimization
  - `opt`: Optimiser defined using Optimisers.jl
  - `n_epochs`: number of epochs for training
"""
function Surrogates.NeuralSurrogate(
        x, y, lb, ub; model = Chain(Dense(length(x[1]), 1)),
        loss = Flux.mse, opt = Optimisers.Adam(1.0e-3),
        n_epochs::Int = 10
    )
    if x isa Tuple
        x = reduce(hcat, x)'
    elseif x isa Vector{<:Tuple}
        x = reduce(hcat, collect.(x))
    elseif x isa Vector
        if size(x) == (1,) && size(x[1]) == ()
            x = hcat(x)
        else
            x = reduce(hcat, x)
        end
    end
    y = reduce(hcat, y)
    opt_state = Flux.setup(opt, model)
    for _ in 1:n_epochs
        grads = Flux.gradient(model) do m
            result = m(x)
            loss(result, y)
        end
        Flux.update!(opt_state, model, grads[1])
    end
    ps = Flux.trainable(model)
    return NeuralSurrogate(x, y, model, loss, opt, ps, n_epochs, lb, ub)
end

function (my_neural::NeuralSurrogate)(val)
    out = my_neural.model(val)
    return out
end

function (my_neural::NeuralSurrogate)(val::Tuple)
    out = my_neural.model(collect(val))
    return out
end

function (my_neural::NeuralSurrogate)(val::Number)
    out = my_neural(reduce(hcat, [[val]]))
    return out
end

function SurrogatesBase.update!(my_n::NeuralSurrogate, x_new, y_new)
    if x_new isa Tuple
        x_new = reduce(hcat, x_new)'
    elseif x_new isa Vector{<:Tuple}
        x_new = reduce(hcat, collect.(x_new))
    elseif x_new isa Vector
        if size(x_new) == (1,) && size(x_new[1]) == ()
            x_new = hcat(x_new)
        else
            x_new = reduce(hcat, x_new)
        end
    elseif x_new isa Number
        x_new = reduce(hcat, [[x_new]])
    end
    y_new = reduce(hcat, y_new)
    opt_state = Flux.setup(my_n.opt, my_n.model)
    for _ in 1:(my_n.n_epochs)
        grads = Flux.gradient(my_n.model) do m
            result = m(x_new)
            my_n.loss(result, y_new)
        end
        Flux.update!(opt_state, my_n.model, grads[1])
    end
    my_n.ps = Flux.trainable(my_n.model)
    my_n.x = hcat(my_n.x, x_new)
    my_n.y = hcat(my_n.y, y_new)
    return nothing
end

# Helper functions for data normalization
function _normalize_x(x)
    """Convert various input formats to matrix (n_features × n_samples)"""
    if x isa Tuple
        return reduce(hcat, x)'
    elseif x isa Vector{<:Tuple}
        return reduce(hcat, collect.(x))
    elseif x isa Vector
        if size(x) == (1,) && size(x[1]) == ()
            return hcat(x)
        else
            return reduce(hcat, x)
        end
    elseif x isa Matrix
        return x
    else
        throw(ArgumentError("Unsupported input format for x"))
    end
end

function _normalize_y(y)
    """Convert y to matrix (n_outputs × n_samples)"""
    if y isa Vector
        # Vector of scalars: create row vector (1 × n_samples)
        return reshape(y, 1, length(y))
    elseif y isa Matrix
        # Already a matrix, assume it's (n_outputs × n_samples) or (n_samples × n_outputs)
        if size(y, 1) > size(y, 2) && size(y, 2) == 1
            # Column vector: transpose to row vector
            return y'
        else
            # Assume already in correct format
            return y
        end
    else
        throw(ArgumentError("Unsupported input format for y"))
    end
end

function _normalize_dydx(dydx, n_inputs, n_outputs)
    """Convert dydx to matrix (n_samples × n_inputs) for single output or (n_samples × n_inputs × n_outputs) for multi-output"""
    if dydx === nothing
        return nothing
    elseif dydx isa Matrix
        # Assume (n_samples × n_inputs) for single output
        return dydx
    elseif dydx isa Vector
        if eltype(dydx) <: Number
            # 1D case: vector of gradients
            return reshape(dydx, length(dydx), 1)
        else
            # Vector of vectors: (n_samples × n_inputs)
            return reduce(hcat, dydx)'
        end
    else
        throw(ArgumentError("Unsupported input format for dydx"))
    end
end

function _compute_gradient_loss(model, x_normalized, dydx_true, n_inputs, n_outputs, is_normalize, x_std, y_std)
    """Compute gradient loss using batch processing where possible"""
    n_samples = size(x_normalized, 2)
    gradient_loss = 0.0
    
    # For each sample, compute predicted gradient
    for i in 1:n_samples
        x_i = x_normalized[:, i]
        
        # Scale true gradients if normalization is used
        if is_normalize
            # For normalized inputs/outputs: d(normalized_y)/d(normalized_x) = (dy/dx) * (x_std / y_std)
            if n_outputs == 1
                scale = x_std ./ y_std[1]
                dydx_true_i = dydx_true[i, :] .* vec(scale)
            else
                # Multi-output: scale each output dimension
                scale = x_std ./ y_std
                dydx_true_i = dydx_true[i, :] .* vec(scale)
            end
        else
            dydx_true_i = dydx_true[i, :]
        end
        
        # Compute predicted gradient using automatic differentiation
        if n_outputs == 1
            dydx_pred_i = Zygote.gradient(x -> model(reshape(x, n_inputs, 1))[1], x_i)[1]
            gradient_loss += sum((dydx_pred_i .- dydx_true_i).^2)
        else
            # Multi-output: compute gradient for each output
            for out_idx in 1:n_outputs
                dydx_pred_i = Zygote.gradient(x -> model(reshape(x, n_inputs, 1))[out_idx], x_i)[1]
                # For multi-output, dydx_true should have shape (n_inputs,) per output
                # Currently assuming same gradient for all outputs (limitation)
                gradient_loss += sum((dydx_pred_i .- dydx_true_i).^2)
            end
        end
    end
    
    # Average over samples and outputs
    normalization_factor = n_samples * (n_outputs == 1 ? 1 : n_outputs)
    return gradient_loss / normalization_factor
end

function _train_genn!(model, x_normalized, y_normalized, dydx_processed, opt, n_epochs, gamma, 
                     n_inputs, n_outputs, is_normalize, x_std, y_std)
    """Shared training function for GENN"""
    opt_state = Flux.setup(opt, model)
    
    for _ in 1:n_epochs
        grads = Flux.gradient(model) do m
            y_pred = m(x_normalized)
            value_loss = Flux.mse(y_pred, y_normalized)
            
            gradient_loss = 0.0
            if dydx_processed !== nothing
                gradient_loss = _compute_gradient_loss(m, x_normalized, dydx_processed, 
                                                      n_inputs, n_outputs, is_normalize, x_std, y_std)
            end
            
            return value_loss + gamma * gradient_loss
        end
        
        Flux.update!(opt_state, model, grads[1])
    end
    
    return Flux.trainable(model)
end

"""
    GENNSurrogate(x, y, lb, ub; dydx = nothing,
                                  model = Chain(Dense(length(x[1]), 12, relu), Dense(12, 12, relu), Dense(12, 1)),
                                  opt = Optimisers.Adam(0.05),
                                  n_epochs = 1000,
                                  gamma = 1.0,
                                  lambda = 0.01,
                                  is_normalize = false)

Gradient-Enhanced Neural Network (GENN) surrogate model.

## Arguments

  - `x`: Input data points.
  - `y`: Output data points.
  - `lb`: Lower bound of input data points.
  - `ub`: Upper bound of input data points.

# Keyword Arguments

  - `dydx`: Optional gradients of y with respect to x. For single output: matrix of shape (n_samples, n_inputs) or vector of vectors. For multiple outputs, currently assumes same gradient applies to all outputs (can be extended in future).
  - `model`: Flux Chain model (default: 2 hidden layers with 12 neurons each)
  - `opt`: Optimiser defined using Optimisers.jl (default: Adam with learning rate 0.05)
  - `n_epochs`: Number of epochs for training (default: 1000)
  - `gamma`: Gradient-enhancement coefficient (default: 1.0). Higher values weight gradient errors more.
  - `lambda`: L2 regularization coefficient (default: 0.01)
  - `is_normalize`: Whether to normalize inputs/outputs (default: false)

# Example

```julia
using Surrogates
using Flux

x = [[1.0], [2.0], [3.0]]
y = [1.0, 4.0, 9.0]
dydx = [[2.0], [4.0], [6.0]]  # gradients of y = x^2

genn = GENNSurrogate(x, y, 0.0, 10.0, dydx = dydx)
pred = genn(2.5)
grad_pred = predict_derivative(genn, [2.5])
```
"""
function GENNSurrogate(x, y, lb, ub;
        dydx = nothing,
        model = nothing,
        opt = Optimisers.Adam(0.05),
        n_epochs::Int = 1000,
        gamma::Real = 1.0,
        lambda::Real = 0.01,
        is_normalize::Bool = false)
    
    # Normalize input data formats
    x_mat = _normalize_x(x)
    y_mat = _normalize_y(y)
    
    n_inputs = size(x_mat, 1)
    n_outputs = size(y_mat, 1)
    
    # Normalize gradients
    dydx_processed = _normalize_dydx(dydx, n_inputs, n_outputs)
    
    # Create default model if not provided
    if model === nothing
        model = Chain(
            Dense(n_inputs, 12, relu),
            Dense(12, 12, relu),
            Dense(12, n_outputs)
        )
    end
    
    # Add L2 regularization if specified
    if lambda > 0.0
        opt = Optimisers.OptimiserChain(Optimisers.WeightDecay(lambda), opt)
    end
    
    # Normalize data if requested
    if is_normalize
        x_mean = mean(x_mat, dims=2)
        x_std = std(x_mat, dims=2) .+ 1e-8
        y_mean = mean(y_mat, dims=2)
        y_std = std(y_mat, dims=2) .+ 1e-8
        x_normalized = (x_mat .- x_mean) ./ x_std
        y_normalized = (y_mat .- y_mean) ./ y_std
    else
        x_normalized = x_mat
        y_normalized = y_mat
        x_std = ones(size(x_mat, 1), 1)
        y_std = ones(size(y_mat, 1), 1)
    end
    
    # Train the model
    ps = _train_genn!(model, x_normalized, y_normalized, dydx_processed, opt, n_epochs, gamma,
                     n_inputs, n_outputs, is_normalize, x_std, y_std)
    
    return GENNSurrogate(x_mat, y_mat, dydx_processed, model, opt, ps, n_epochs, lb, ub, gamma)
end

function (genn::GENNSurrogate)(val)
    if val isa Tuple
        val = collect(val)
    elseif val isa Number
        val = [val]
    end
    
    expected_dim = size(genn.x, 1)
    input_dim = length(val)
    if input_dim != expected_dim
        throw(ArgumentError("Expected $expected_dim-dimensional input, got $input_dim-dimensional input."))
    end
    
    val_matrix = reshape(val, expected_dim, 1)
    out = genn.model(val_matrix)
    
    if size(out, 1) == 1 && size(out, 2) == 1
        return out[1]
    elseif size(out, 1) == 1
        return vec(out)
    else
        return out
    end
end

function (genn::GENNSurrogate)(val::Tuple)
    return genn(collect(val))
end

function (genn::GENNSurrogate)(val::Number)
    return genn([val])
end

"""
    predict_derivative(genn::GENNSurrogate, val)

Predict the derivative of the GENN surrogate at the given point.

## Arguments

  - `genn`: GENNSurrogate model
  - `val`: Input point(s) at which to predict derivatives

## Returns

  - For 1D input: derivative value(s)
  - For multi-dimensional input: gradient vector(s)
"""
function predict_derivative(genn::GENNSurrogate, val)
    # Normalize input
    if val isa Tuple
        val = collect(val)
    elseif val isa Number
        val = [val]
    end
    
    expected_dim = size(genn.x, 1)  # Fixed: x is (n_features × n_samples)
    input_dim = length(val)
    if input_dim != expected_dim
        throw(ArgumentError("Expected $expected_dim-dimensional input, got $input_dim-dimensional input."))
    end
    
    val_matrix = reshape(val, expected_dim, 1)
    n_inputs = size(val_matrix, 1)
    n_outputs = size(genn.y, 1)  # Fixed: y is (n_outputs × n_samples)
    
    # Compute Jacobian: dy/dx for each output
    jac = Zygote.jacobian(x -> vec(genn.model(x)), val_matrix)[1]
    # jac has shape (n_outputs, n_inputs)
    
    if n_outputs == 1
        return vec(jac)  # Return as vector for single output
    else
        return jac  # Return as matrix for multi-output
    end
end

function SurrogatesBase.update!(genn::GENNSurrogate, x_new, y_new; dydx_new = nothing)
    # Normalize new data to match stored format
    x_new_mat = _normalize_x(x_new)
    y_new_mat = _normalize_y(y_new)
    
    # Ensure dimensions match
    if size(x_new_mat, 1) != size(genn.x, 1)
        throw(ArgumentError("Input dimension mismatch: expected $(size(genn.x, 1)), got $(size(x_new_mat, 1))"))
    end
    if size(y_new_mat, 1) != size(genn.y, 1)
        throw(ArgumentError("Output dimension mismatch: expected $(size(genn.y, 1)), got $(size(y_new_mat, 1))"))
    end
    
    # Process new gradients
    n_inputs = size(genn.x, 1)
    n_outputs = size(genn.y, 1)
    dydx_new_processed = _normalize_dydx(dydx_new, n_inputs, n_outputs)
    
    # Combine with existing data
    if genn.dydx === nothing && dydx_new_processed !== nothing
        genn.dydx = dydx_new_processed
    elseif genn.dydx !== nothing && dydx_new_processed !== nothing
        genn.dydx = vcat(genn.dydx, dydx_new_processed)
    end
    
    x_combined = hcat(genn.x, x_new_mat)
    y_combined = hcat(genn.y, y_new_mat)
    
    # For update, we don't normalize (assume data is in same scale)
    # In practice, you might want to re-normalize, but that would require storing normalization params
    x_normalized = x_combined
    y_normalized = y_combined
    x_std = ones(size(x_combined, 1), 1)
    y_std = ones(size(y_combined, 1), 1)
    
    # Retrain on combined data
    genn.ps = _train_genn!(genn.model, x_normalized, y_normalized, genn.dydx, genn.opt, 
                          genn.n_epochs, genn.gamma, n_inputs, n_outputs, false, x_std, y_std)
    
    # Update stored data
    genn.x = x_combined
    genn.y = y_combined
    nothing
end

end # module
