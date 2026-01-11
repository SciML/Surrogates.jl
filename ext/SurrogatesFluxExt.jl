module SurrogatesFluxExt

using Surrogates: NeuralSurrogate, Surrogates, GENNSurrogate, predict_derivative
using SurrogatesBase
using Flux: Flux.Optimisers
using Flux
using Zygote
using LinearAlgebra
using Statistics

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
    """
    Convert y to matrix (n_outputs × n_samples).
    
    Required input formats:
    - Single output: vector of scalars [y1, y2, ...] or matrix of shape (1, n_samples) or (n_samples, 1)
    - Multi-output: matrix of shape (n_outputs, n_samples) where n_outputs > 1
    
    For single output, vectors and column vectors are converted to row vector (1 × n_samples).
    For multi-output, the matrix must already be in (n_outputs × n_samples) format and is kept as-is.
    Note: For multi-output with 1 sample, provide as (n_outputs, 1) matrix, not as a column vector.
    """
    if y isa Vector
        # Vector of scalars: create row vector (1 × n_samples)
        return reshape(y, 1, length(y))
    elseif y isa Matrix
        n_rows, n_cols = size(y)
        if n_rows == 1
            # Already (1 × n_samples) - correct format for single output
            return y
        elseif n_cols == 1 && n_rows > 1
            # Column vector: assume single output (n_samples, 1) - transpose to (1, n_samples)
            # For multi-output with 1 sample, user must provide (n_outputs, n_samples) with n_samples > 1
            # or reshape to avoid this ambiguity
            return y'
        else
            # (n_rows × n_cols) where n_rows > 1 and n_cols > 1
            # Assume (n_outputs × n_samples) format - keep as-is
            return y
        end
    else
        throw(ArgumentError("y must be a Vector (for single output) or Matrix. For multi-output, matrix must be (n_outputs × n_samples) with n_samples > 1 to avoid ambiguity with column vectors."))
    end
end

function _normalize_dydx(dydx, n_inputs, n_outputs, n_samples)
    """
    Convert dydx to internal format (n_outputs × n_inputs × n_samples).
    
    Required input shapes:
    - Single output: matrix of shape (n_samples, n_inputs)
    - Multi-output: 3D array of shape (n_outputs, n_inputs, n_samples)
    """
    if dydx === nothing
        return nothing
    end

    if n_outputs == 1
        # Single output: expect (n_samples, n_inputs) matrix
        dydx isa AbstractMatrix || throw(ArgumentError("For single output, dydx must be a matrix of shape (n_samples, n_inputs), got $(typeof(dydx))"))
        mat = Array(dydx)
        if size(mat) != (n_samples, n_inputs)
            throw(ArgumentError("For single output, dydx must have shape (n_samples=$n_samples, n_inputs=$n_inputs), got $(size(mat))"))
        end
        # Convert to internal format: (1, n_inputs, n_samples)
        # mat[i, j] = gradient of sample i, input j
        # result[1, j, i] = gradient of output 1, input j, sample i
        # So: result[1, j, i] = mat[i, j]
        result = Array{eltype(mat), 3}(undef, 1, n_inputs, n_samples)
        result[1, :, :] = permutedims(mat, (2, 1))  # (n_inputs, n_samples) -> (1, n_inputs, n_samples)
        return result
    else
        # Multi-output: expect (n_outputs, n_inputs, n_samples) 3D array
        dydx isa AbstractArray && ndims(dydx) == 3 || throw(ArgumentError("For multi-output, dydx must be a 3D array of shape (n_outputs, n_inputs, n_samples), got $(typeof(dydx)) with $(ndims(dydx)) dimensions"))
        arr = Array(dydx)
        if size(arr) != (n_outputs, n_inputs, n_samples)
            throw(ArgumentError("For multi-output, dydx must have shape (n_outputs=$n_outputs, n_inputs=$n_inputs, n_samples=$n_samples), got $(size(arr))"))
        end
        return arr
    end
end

function _compute_gradient_loss(model, x_normalized, dydx_true, n_inputs, n_outputs, is_normalize, x_std, y_std)
    """Compute gradient loss using batch processing where possible"""
    n_samples = size(x_normalized, 2)
    ndims(dydx_true) == 3 || throw(ArgumentError("dydx must have dimensions (n_outputs, n_inputs, n_samples)"))
    size(dydx_true, 3) == n_samples || throw(ArgumentError("Gradient sample count $(size(dydx_true, 3)) does not match input sample count $n_samples"))
    gradient_loss = 0.0
    x_scale = vec(x_std)
    
    for i in 1:n_samples
        x_i = x_normalized[:, i]

        for out_idx in 1:n_outputs
            grad_true = view(dydx_true, out_idx, :, i)
            # Scale gradients if normalization is used: d(normalized_y)/d(normalized_x) = (dy/dx) * (x_std / y_std)
            if is_normalize
                grad_true_scaled = grad_true .* (x_scale ./ y_std[out_idx])
            else
                grad_true_scaled = grad_true
            end

            dydx_pred_i = Zygote.gradient(x -> model(reshape(x, n_inputs, 1))[out_idx], x_i)[1]
            gradient_loss += sum((dydx_pred_i .- grad_true_scaled).^2)
        end
    end
    
    return gradient_loss / (n_samples * n_outputs)
end

function _train_genn!(model, x_normalized, y_normalized, dydx_processed, opt, n_epochs, gamma, 
                     n_inputs, n_outputs, is_normalize, x_std, y_std)
    """Shared training function for GENN"""
    opt_state = Flux.setup(opt, model)
    
    for i in 1:n_epochs
        @info "Epoch $i"
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
    - **Single output**: vector of scalars `[y1, y2, ...]` or matrix of shape `(1, n_samples)` or `(n_samples, 1)`.
    - **Multi-output**: matrix of shape `(n_outputs, n_samples)` where each column is one sample's output vector.
  - `lb`: Lower bound of input data points.
  - `ub`: Upper bound of input data points.

# Keyword Arguments

  - `dydx`: Optional gradients of y with respect to x. 
    - **Single output**: matrix of shape `(n_samples, n_inputs)` where each row is the gradient for one sample.
    - **Multi-output**: 3D array of shape `(n_outputs, n_inputs, n_samples)` where `dydx[out_idx, :, sample_idx]` is the gradient of output `out_idx` with respect to inputs for sample `sample_idx`.
  - `model`: Flux Chain model (default: 2 hidden layers with 12 neurons each)
  - `opt`: Optimiser defined using Optimisers.jl (default: Adam with learning rate 0.05)
  - `n_epochs`: Number of epochs for training (default: 1000)
  - `gamma`: Gradient-enhancement coefficient (default: 1.0). Higher values weight gradient errors more.
  - `lambda`: L2 regularization coefficient (default: 0.01)
  - `is_normalize`: Whether to normalize inputs/outputs (default: false)

# Examples

## Single output with gradients

```julia
using Surrogates
using Flux

x = [[1.0], [2.0], [3.0]]
y = [1.0, 4.0, 9.0]
# dydx must be (n_samples, n_inputs) = (3, 1) for single output
dydx = reshape([2.0, 4.0, 6.0], :, 1)  # gradients of y = x^2

genn = GENNSurrogate(x, y, 0.0, 10.0, dydx = dydx)
pred = genn(2.5)
grad_pred = predict_derivative(genn, [2.5])
```

## Multi-output with gradients

```julia
x = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
y = hcat([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]...)  # 2 outputs, 3 samples
# dydx must be (n_outputs, n_inputs, n_samples) = (2, 2, 3) for multi-output
# dydx[out_idx, input_idx, sample_idx] = gradient of output out_idx w.r.t. input input_idx for sample sample_idx
dydx = Array{Float64, 3}(undef, 2, 2, 3)
dydx[:, :, 1] = [1.0 0.0; 0.0 1.0]  # gradients for sample 1
dydx[:, :, 2] = [1.0 0.0; 0.0 1.0]  # gradients for sample 2
dydx[:, :, 3] = [1.0 0.0; 0.0 1.0]  # gradients for sample 3

genn = GENNSurrogate(x, y, [0.0, 0.0], [1.0, 1.0], dydx = dydx)
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
    n_samples = size(x_mat, 2)
    
    # Normalize gradients (validates shape and converts to internal format)
    dydx_processed = _normalize_dydx(dydx, n_inputs, n_outputs, n_samples)
    
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
        x_mean = nothing
        x_std = nothing
        y_mean = nothing
        y_std = nothing
    end
    
    # Train the model
    x_std_for_training = is_normalize ? x_std : ones(size(x_mat, 1), 1)
    y_std_for_training = is_normalize ? y_std : ones(size(y_mat, 1), 1)
    ps = _train_genn!(model, x_normalized, y_normalized, dydx_processed, opt, n_epochs, gamma,
                     n_inputs, n_outputs, is_normalize, x_std_for_training, y_std_for_training)
    
    return GENNSurrogate(x_mat, y_mat, dydx_processed, model, opt, ps, n_epochs, lb, ub, gamma,
                        x_mean, x_std, y_mean, y_std, is_normalize)
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
    
    # Normalize input if normalization was used during training
    val_matrix = reshape(val, expected_dim, 1)
    if genn.is_normalize && genn.x_mean !== nothing
        val_matrix = (val_matrix .- genn.x_mean) ./ genn.x_std
    end
    
    out = genn.model(val_matrix)
    
    # Denormalize output if normalization was used during training
    if genn.is_normalize && genn.y_mean !== nothing
        out = out .* genn.y_std .+ genn.y_mean
    end
    
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
function Surrogates.predict_derivative(genn::GENNSurrogate, val)
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
    
    # Normalize input if normalization was used during training
    val_matrix = reshape(val, expected_dim, 1)
    if genn.is_normalize && genn.x_mean !== nothing
        val_matrix = (val_matrix .- genn.x_mean) ./ genn.x_std
    end
    
    n_inputs = size(val_matrix, 1)
    n_outputs = size(genn.y, 1)  # Fixed: y is (n_outputs × n_samples)
    
    # Compute Jacobian: dy/dx for each output (on normalized space)
    jac_normalized = Zygote.jacobian(x -> vec(genn.model(x)), val_matrix)[1]
    # jac_normalized has shape (n_outputs, n_inputs) in normalized space
    
    # Convert Jacobian from normalized space to original space
    # d(normalized_y)/d(normalized_x) = (dy/dx) * (x_std / y_std)
    # Therefore: dy/dx = d(normalized_y)/d(normalized_x) * (y_std / x_std)
    if genn.is_normalize && genn.x_std !== nothing && genn.y_std !== nothing
        x_scale = vec(genn.x_std)
        for out_idx in 1:n_outputs
            jac_normalized[out_idx, :] = jac_normalized[out_idx, :] .* (genn.y_std[out_idx] ./ x_scale)
        end
    end
    
    if n_outputs == 1
        return vec(jac_normalized)  # Return as vector for single output
    else
        return jac_normalized  # Return as matrix for multi-output
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
    n_new_samples = size(x_new_mat, 2)
    dydx_new_processed = _normalize_dydx(dydx_new, n_inputs, n_outputs, n_new_samples)
    
    # Combine with existing data
    if genn.dydx === nothing && dydx_new_processed !== nothing
        genn.dydx = dydx_new_processed
    elseif genn.dydx !== nothing && dydx_new_processed !== nothing
        genn.dydx = cat(genn.dydx, dydx_new_processed; dims=3)
    end
    
    x_combined = hcat(genn.x, x_new_mat)
    y_combined = hcat(genn.y, y_new_mat)
    total_samples = size(x_combined, 2)
    if genn.dydx !== nothing && size(genn.dydx, 3) != total_samples
        throw(ArgumentError("Gradient sample count $(size(genn.dydx, 3)) does not match combined samples $total_samples"))
    end
    
    # Normalize data if normalization was used during initial training
    if genn.is_normalize && genn.x_mean !== nothing
        # Recompute normalization parameters from combined data
        x_mean_new = mean(x_combined, dims=2)
        x_std_new = std(x_combined, dims=2) .+ 1e-8
        y_mean_new = mean(y_combined, dims=2)
        y_std_new = std(y_combined, dims=2) .+ 1e-8
        x_normalized = (x_combined .- x_mean_new) ./ x_std_new
        y_normalized = (y_combined .- y_mean_new) ./ y_std_new
        # Update stored normalization parameters
        genn.x_mean = x_mean_new
        genn.x_std = x_std_new
        genn.y_mean = y_mean_new
        genn.y_std = y_std_new
    else
        x_normalized = x_combined
        y_normalized = y_combined
    end
    
    x_std_for_training = genn.is_normalize && genn.x_std !== nothing ? genn.x_std : ones(size(x_combined, 1), 1)
    y_std_for_training = genn.is_normalize && genn.y_std !== nothing ? genn.y_std : ones(size(y_combined, 1), 1)
    
    # Retrain on combined data
    genn.ps = _train_genn!(genn.model, x_normalized, y_normalized, genn.dydx, genn.opt, 
                          genn.n_epochs, genn.gamma, n_inputs, n_outputs, genn.is_normalize, x_std_for_training, y_std_for_training)
    
    # Update stored data
    genn.x = x_combined
    genn.y = y_combined
    nothing
end

end # module
