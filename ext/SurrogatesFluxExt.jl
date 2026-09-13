module SurrogatesFluxExt

using Flux: Chain, Dense
using NNlib: relu
using Statistics: mean, std
using Surrogates: NeuralSurrogate, Surrogates, _is_single_sample, _match_stored

import Flux
import Optimisers
import Surrogates: GENNSurrogate
import SurrogatesBase
import Zygote

# A non-positive epoch count skips training altogether and leaves the network at
# its random initialization, which is a silently useless surrogate rather than an
# error.
function _check_n_epochs(name, n_epochs)
    n_epochs < 1 && throw(
        ArgumentError(
            "$name needs at least one training epoch! Got: n_epochs = $(n_epochs)."
        )
    )
    return nothing
end

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
    _check_n_epochs("NeuralSurrogate", n_epochs)
    # Flux wants features-by-samples matrices; the surrogate stores the samples
    # themselves. See the note on the struct fields below for why.
    X = _design_matrix(x)
    Y = _response_matrix(y)
    opt_state = Flux.setup(opt, model)
    for _ in 1:n_epochs
        grads = Flux.gradient(model) do m
            result = m(X)
            loss(result, Y)
        end
        Optimisers.update!(opt_state, model, grads[1])
    end
    ps = Optimisers.trainables(model)
    # Stored as a vector of points and a vector of responses, the layout every
    # other surrogate uses and the one the optimizers assume: `length(surr.x)`
    # is the sample count and `surr.x[i]` is a point.
    return NeuralSurrogate(
        _columns_as_points(X), _columns_as_responses(Y),
        model, loss, opt, ps, n_epochs, lb, ub
    )
end

# One query point as a features-by-1 matrix, the shape Flux expects for a single
# sample. `reshape` rather than a copy, so reverse-mode AD can push a gradient
# back through a call.
_query_matrix(val::Number) = reshape([val], 1, 1)
_query_matrix(val::Tuple) = reshape(collect(val), length(val), 1)
_query_matrix(val::AbstractVector) = reshape(val, length(val), 1)
_query_matrix(val::AbstractMatrix) = val

"""
A prediction in the same shape as the responses the surrogate was fitted to: a
scalar for a single-output model, a vector for a multi-output one.

Flux hands back a `k x 1` matrix even when `k == 1`, and the optimizers compare
responses with `<`. A model chain ending in `first` already returns a `Number`;
that case passes straight through.
"""
# Dispatched on the stored response type rather than branching on the model's
# output shape at runtime: a surrogate's output count is fixed when it is fitted,
# and `Y` already records it. Branching instead inferred `Union{Float32,
# Vector{Float32}}`, which every optimizer call then had to resolve dynamically.
#
# `first` covers both a `k x 1` matrix and the bare `Number` a chain ending in
# `first` returns.
_predict(my_neural::NeuralSurrogate{X, Y}, val) where {X, Y <: AbstractVector{<:Number}} =
    first(my_neural.model(_query_matrix(val)))
_predict(my_neural::NeuralSurrogate, val) = vec(my_neural.model(_query_matrix(val)))

(my_neural::NeuralSurrogate)(val::Number) = _predict(my_neural, val)
(my_neural::NeuralSurrogate)(val::Tuple) = _predict(my_neural, val)
(my_neural::NeuralSurrogate)(val) = _predict(my_neural, val)

# The two conversions at the Flux boundary. Everything above the boundary keeps
# the package's own layout — a vector of points, a vector of responses — and only
# these turn it into the features-by-samples / outputs-by-samples matrices Flux
# consumes.

# The columns of a training matrix, back in the package's sample layout.
#
# Storage is derived from the normalized matrices rather than from whatever the
# caller passed, so that `first(y)` is one sample's response whichever of the
# accepted input forms `y` arrived in.
_columns_as_points(X) = size(X, 1) == 1 ? collect(vec(X)) :
    [Tuple(view(X, :, j)) for j in axes(X, 2)]
_columns_as_responses(Y) = size(Y, 1) == 1 ? collect(vec(Y)) :
    [collect(view(Y, :, j)) for j in axes(Y, 2)]

# Bring a new point into the representation the stored design already uses.
#
# Points as a features-by-samples matrix.
function _design_matrix(x)
    x isa Tuple && return reduce(hcat, collect(x))
    x isa Number && return fill(float(x), 1, 1)
    first(x) isa Number && return reshape(collect(float.(x)), 1, length(x))
    return reduce(hcat, collect.(x))
end

# Responses as an outputs-by-samples matrix.
#
# `y` is one entry per sample, so a scalar-response vector gives a 1xn row and a
# multi-output one a kxn matrix. Not safe on a *lone* response — `[y1, y2]` would
# give a 1x2 row where a 2x1 column is meant — so callers must resolve
# single-versus-batch first.
function _response_matrix(y)
    y isa Number && return fill(float(y), 1, 1)
    # An `n_outputs x n_samples` matrix is already in the layout Flux wants.
    # `first(y)` is a number for a matrix too, so without this the branch below
    # flattened it to a `1 x (k*n)` row and the loss saw the wrong shape.
    y isa AbstractMatrix && return float.(y)
    first(y) isa Number && return reshape(collect(float.(y)), 1, length(y))
    return reduce(hcat, collect.(y))
end

function SurrogatesBase.update!(my_n::NeuralSurrogate, x_new, y_new)
    # `_is_single_sample` is the core's own rule for telling one new sample from
    # a batch of them — it decides from the *inputs*, which is what makes a lone
    # multi-output response unambiguous: `[y1, y2]` against a single new point is
    # one two-output response, not two scalar ones.
    reference = first(my_n.x)
    added_x, added_y = if _is_single_sample(x_new, reference)
        ([_match_stored(reference, x_new)], [y_new])
    else
        ([_match_stored(reference, p) for p in x_new], collect(y_new))
    end
    x_all = vcat(my_n.x, added_x)
    y_all = vcat(my_n.y, added_y)

    # Training continues on the newly added samples only, as it did before.
    X = _design_matrix(added_x)
    Y = _response_matrix(added_y)

    opt_state = Flux.setup(my_n.opt, my_n.model)
    for _ in 1:(my_n.n_epochs)
        grads = Flux.gradient(my_n.model) do m
            result = m(X)
            my_n.loss(result, Y)
        end
        Optimisers.update!(opt_state, my_n.model, grads[1])
    end
    my_n.ps = Optimisers.trainables(my_n.model)
    my_n.x = x_all
    my_n.y = y_all
    return nothing
end

# Helper functions for data normalization
# Convert the accepted input formats to an n_features x n_samples matrix.
function _normalize_x(x)
    if x isa Number
        # One sample on a scalar domain: the ordinary case for a
        # one-dimensional problem, since the optimizers add points one at a time.
        return fill(float(x), 1, 1)
    elseif x isa Tuple
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

"""
    _normalize_y(y, n_samples = nothing)

Convert `y` to an `n_outputs x n_samples` matrix.

Accepted forms are a scalar, a vector of scalars, or a `(1, n_samples)` matrix
for a single output, and an `(n_outputs, n_samples)` matrix for several.

An `n x 1` matrix is ambiguous on its own — `n` single-output samples, or one
`n`-output sample? `n_samples`, which both callers know from the already
normalized inputs, resolves it; without it the first reading is taken.
"""
function _normalize_y(y, n_samples = nothing)
    if y isa Number
        # A single response for a single new sample.
        return fill(float(y), 1, 1)
    elseif y isa Vector
        # A vector of per-sample response vectors: the layout every other
        # surrogate takes for multi-output. One column per sample.
        first(y) isa Number || return reduce(hcat, collect.(float.(y)))
        # One sample's multi-output response, not several scalar ones. Only
        # `n_samples` separates the two readings, exactly as `_append_samples`
        # decides single-versus-batch from the inputs; without it a two-output
        # response `[y1, y2]` became two one-output samples and `update!`
        # rejected it as a dimension mismatch.
        if n_samples == 1 && length(y) > 1
            return reshape(float.(y), length(y), 1)
        end
        # Vector of scalars: create row vector (1 x n_samples)
        return reshape(y, 1, length(y))
    elseif y isa Matrix
        n_rows, n_cols = size(y)
        # When the sample count is known and the columns already match it, the
        # matrix is in (n_outputs, n_samples) form — including the (k, 1) case
        # that is otherwise indistinguishable from k scalar samples.
        n_samples !== nothing && n_cols == n_samples && return y
        if n_rows == 1
            # Already (1 x n_samples) - correct format for single output
            return y
        elseif n_cols == 1 && n_rows > 1
            # Column vector: assume single output (n_samples, 1) - transpose to (1, n_samples)
            # For multi-output with 1 sample, user must provide (n_outputs, n_samples) with n_samples > 1
            # or reshape to avoid this ambiguity
            return transpose(y)
        else
            # (n_rows x n_cols) where n_rows > 1 and n_cols > 1
            # Assume (n_outputs x n_samples) format - keep as-is
            return y
        end
    else
        throw(ArgumentError("y must be a Vector (for single output) or Matrix. For multi-output, matrix must be (n_outputs x n_samples) with n_samples > 1 to avoid ambiguity with column vectors."))
    end
end

# `GENNSurrogate` stores its samples the way every other surrogate does — a
# vector of points and a vector of responses — so `length(genn.x)` is the sample
# count and `genn.x[i]` is a point. The dimensions the training code needs are
# derived from a sample.
_n_inputs(genn::GENNSurrogate) = (p = first(genn.x); p isa Number ? 1 : length(p))
_n_outputs(genn::GENNSurrogate) = (r = first(genn.y); r isa Number ? 1 : length(r))

"""
    _normalize_dydx(dydx, n_inputs, n_outputs, n_samples)

Convert supplied gradients to the internal `(n_outputs, n_inputs, n_samples)`
layout. They must arrive as an `(n_samples, n_inputs)` matrix for a single
output and an `(n_outputs, n_inputs, n_samples)` array for several.
"""
function _normalize_dydx(dydx, n_inputs, n_outputs, n_samples)
    # `update!` takes `dydx_new` as an optional keyword and its caller branches
    # on the result being `nothing`, so absent gradients pass through.
    dydx === nothing && return nothing

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
    n_samples = size(x_normalized, 2)
    ndims(dydx_true) == 3 || throw(ArgumentError("dydx must have dimensions (n_outputs, n_inputs, n_samples)"))
    size(dydx_true, 3) == n_samples || throw(ArgumentError("Gradient sample count $(size(dydx_true, 3)) does not match input sample count $n_samples"))
    gradient_loss = 0.0
    σx_in = vec(x_std)
    σy_out = is_normalize ? vec(y_std) : nothing

    # Samples are independent under the model, so ∂/∂x of the summed output over the
    # whole batch yields every per-sample input gradient in one reverse pass. Doing this
    # per-sample instead costs n_samples nested Zygote calls per epoch.
    for out_idx in 1:n_outputs
        grad_true = @view dydx_true[out_idx, :, :]        # (n_inputs, n_samples)
        if is_normalize
            grad_true_scaled = grad_true .* (σx_in ./ σy_out[out_idx])
        else
            grad_true_scaled = grad_true
        end

        dydx_pred = Zygote.gradient(z -> sum(model(z)[out_idx, :]), x_normalized)[1]
        gradient_loss += sum((dydx_pred .- grad_true_scaled) .^ 2)
    end

    # Mean over (samples × outputs × inputs) to keep loss dimension-invariant
    return gradient_loss / (n_samples * n_outputs * n_inputs)
end

function _train_genn!(
        model, x_normalized, y_normalized, dydx_processed, opt, n_epochs, gamma,
        n_inputs, n_outputs, is_normalize, x_std, y_std
    )
    opt_state = Flux.setup(opt, model)

    for _ in 1:n_epochs
        grads = Flux.gradient(model) do m
            y_pred = m(x_normalized)
            value_loss = Flux.mse(y_pred, y_normalized)

            gradient_loss = 0.0
            if dydx_processed !== nothing
                gradient_loss = _compute_gradient_loss(
                    m, x_normalized, dydx_processed,
                    n_inputs, n_outputs, is_normalize, x_std, y_std
                )
            end

            return value_loss + gamma * gradient_loss
        end

        Optimisers.update!(opt_state, model, grads[1])
    end

    return Optimisers.trainables(model)
end

"""
    GENNSurrogate(x, y, lb, ub, dydx;
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
  - `dydx`: Gradients of y with respect to x (required). 
    - **Single output**: matrix of shape `(n_samples, n_inputs)` where each row is the gradient for one sample.
    - **Multi-output**: 3D array of shape `(n_outputs, n_inputs, n_samples)` where `dydx[out_idx, :, sample_idx]` is the gradient of output `out_idx` with respect to inputs for sample `sample_idx`.
  - `lb`: Lower bound of input data points.
  - `ub`: Upper bound of input data points.

# Keyword Arguments

  - `model`: Flux Chain model (default: 2 hidden layers with 12 neurons each)
  - `opt`: Optimiser defined using Optimisers.jl (default: Adam with learning rate 0.05)
  - `n_epochs`: Number of epochs for training (default: 1000)
  - `gamma`: Gradient-enhancement coefficient (default: 1.0). Higher values weight gradient errors more.
  - `lambda`: L2 regularization coefficient (default: 0.01)
  - `is_normalize`: Whether to normalize inputs/outputs (default: false)
"""
function GENNSurrogate(
        x, y, lb, ub, dydx;
        model = nothing,
        opt = Optimisers.Adam(0.05),
        n_epochs::Int = 1000,
        gamma::Real = 1.0,
        lambda::Real = 0.01,
        is_normalize::Bool = false
    )

    _check_n_epochs("GENNSurrogate", n_epochs)
    # `dydx` is what makes this model gradient-enhanced. Without it the fit is an
    # ordinary `NeuralSurrogate` under another name, and `predict_derivative`
    # reports slopes no gradient observation ever constrained.
    dydx === nothing && throw(
        ArgumentError(
            "GENNSurrogate is gradient-enhanced and needs per-sample gradients; " *
                "got `dydx = nothing`. Use `NeuralSurrogate` when there are no " *
                "gradients to supply."
        )
    )

    # Normalize input data formats
    x_mat = _normalize_x(x)
    y_mat = _normalize_y(y, size(x_mat, 2))

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
        x_mean = mean(x_mat, dims = 2)
        x_std = std(x_mat, dims = 2) .+ 1.0e-8
        y_mean = mean(y_mat, dims = 2)
        y_std = std(y_mat, dims = 2) .+ 1.0e-8
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
    ps = _train_genn!(
        model, x_normalized, y_normalized, dydx_processed, opt, n_epochs, gamma,
        n_inputs, n_outputs, is_normalize, x_std_for_training, y_std_for_training
    )

    return GENNSurrogate(
        _columns_as_points(x_mat), _columns_as_responses(y_mat), dydx_processed,
        model, opt, ps, n_epochs, lb, ub, gamma,
        x_mean, x_std, y_mean, y_std, is_normalize
    )
end

function (genn::GENNSurrogate)(val)
    if val isa Tuple
        val = collect(val)
    elseif val isa Number
        val = [val]
    end

    expected_dim = _n_inputs(genn)
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

    # Same shape contract as `NeuralSurrogate`, and settled the same way: on the
    # stored response type, so the call infers concretely.
    return _genn_output(genn, out)
end

# A scalar for a single-output model, a vector for a multi-output one. `first`
# covers both a `k x 1` matrix and a bare `Number`.
_genn_output(::GENNSurrogate{X, Y}, out) where {X, Y <: AbstractVector{<:Number}} =
    first(out)
_genn_output(::GENNSurrogate, out) = vec(out)

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

    expected_dim = _n_inputs(genn)
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
    n_outputs = _n_outputs(genn)

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

"""
    update!(genn::GENNSurrogate, x_new, y_new, dydx_new)

Add one observation and its gradient, with the gradient passed positionally.

`GENNSurrogate` is gradient-enhanced: a sample without a gradient leaves
`genn.dydx` short of the design and trips the sample-count check. The optimizers
supply one — `_update_with_sample!` and the virtual-point strategies both call
`update!(surr, x, y, gradient)` positionally, as they do for `GEK` — so this
method exists to receive it and reshape it into the `(n_samples, n_inputs)`
layout `_normalize_dydx` expects for a single-output model.

Without it, every optimization method failed on `GENNSurrogate`: the keyword-only
form was never matched, and the gradientless path raised
`ArgumentError: For single output, dydx must be a matrix ...`.
"""
function SurrogatesBase.update!(genn::GENNSurrogate, x_new, y_new, dydx_new)
    n_inputs = _n_inputs(genn)
    # Zygote hands back a scalar in one dimension and a coordinate vector in
    # several; both describe one sample, so both become a 1 x n_inputs row.
    gradient_row = dydx_new isa Number ? fill(float(dydx_new), 1, 1) :
        reshape(collect(float.(dydx_new)), 1, n_inputs)
    return SurrogatesBase.update!(genn, x_new, y_new; dydx_new = gradient_row)
end

function SurrogatesBase.update!(genn::GENNSurrogate, x_new, y_new; dydx_new = nothing)
    # `_normalize_x` judges shape from the argument alone, so it cannot tell one
    # `d`-dimensional point written as a coordinate vector from `d` separate
    # one-dimensional points. `_is_single_sample` settles it against the stored
    # design, as it does for every other surrogate.
    reference = first(genn.x)
    x_new = _is_single_sample(x_new, reference) ?
        _match_stored(reference, x_new) :
        [_match_stored(reference, p) for p in x_new]

    # Normalize new data to match stored format
    x_new_mat = _normalize_x(x_new)
    y_new_mat = _normalize_y(y_new, size(x_new_mat, 2))

    # Ensure dimensions match
    if size(x_new_mat, 1) != _n_inputs(genn)
        throw(ArgumentError("Input dimension mismatch: expected $(_n_inputs(genn)), got $(size(x_new_mat, 1))"))
    end
    if size(y_new_mat, 1) != _n_outputs(genn)
        throw(ArgumentError("Output dimension mismatch: expected $(_n_outputs(genn)), got $(size(y_new_mat, 1))"))
    end

    # Process new gradients
    n_inputs = _n_inputs(genn)
    n_outputs = _n_outputs(genn)
    n_new_samples = size(x_new_mat, 2)
    dydx_new_processed = _normalize_dydx(dydx_new, n_inputs, n_outputs, n_new_samples)

    # Combine with existing data
    if genn.dydx === nothing && dydx_new_processed !== nothing
        genn.dydx = dydx_new_processed
    elseif genn.dydx !== nothing && dydx_new_processed !== nothing
        genn.dydx = cat(genn.dydx, dydx_new_processed; dims = 3)
    end

    # Training runs on matrices rebuilt from the stored samples.
    # `_design_matrix`/`_response_matrix` convert the *stored* layout;
    # `_normalize_y` is for user input and would mis-shape a multi-output design.
    x_combined = hcat(_design_matrix(genn.x), x_new_mat)
    y_combined = hcat(_response_matrix(genn.y), y_new_mat)
    total_samples = size(x_combined, 2)
    if genn.dydx !== nothing && size(genn.dydx, 3) != total_samples
        throw(ArgumentError("Gradient sample count $(size(genn.dydx, 3)) does not match combined samples $total_samples"))
    end

    # Normalize data if normalization was used during initial training
    if genn.is_normalize && genn.x_mean !== nothing
        # Recompute normalization parameters from combined data
        x_mean_new = mean(x_combined, dims = 2)
        x_std_new = std(x_combined, dims = 2) .+ 1.0e-8
        y_mean_new = mean(y_combined, dims = 2)
        y_std_new = std(y_combined, dims = 2) .+ 1.0e-8
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
    genn.ps = _train_genn!(
        genn.model, x_normalized, y_normalized, genn.dydx, genn.opt,
        genn.n_epochs, genn.gamma, n_inputs, n_outputs, genn.is_normalize, x_std_for_training, y_std_for_training
    )

    # Update stored data, in the package's own layout, derived from the very
    # matrices just trained on so the two cannot drift apart.
    genn.x = _columns_as_points(x_combined)
    genn.y = _columns_as_responses(y_combined)
    return nothing
end


# ---- SurrogatesBase parameter interface -----------------------------------
#
# The trained network weights are the learned state; everything that governs how
# it was trained is configuration. Neither model has a hyperparameter-fitting
# routine, so neither gets `update_hyperparameters!` — a silent no-op would be
# worse than a `MethodError`.

SurrogatesBase.parameters(n::NeuralSurrogate) = (; ps = n.ps, model = n.model)
SurrogatesBase.hyperparameters(n::NeuralSurrogate) = (;
    loss = n.loss, opt = n.opt, n_epochs = n.n_epochs,
)

SurrogatesBase.parameters(g::GENNSurrogate) = (;
    ps = g.ps, model = g.model,
    x_mean = g.x_mean, x_std = g.x_std, y_mean = g.y_mean, y_std = g.y_std,
)
SurrogatesBase.hyperparameters(g::GENNSurrogate) = (;
    opt = g.opt, n_epochs = g.n_epochs, gamma = g.gamma,
    is_normalize = g.is_normalize,
)

end # module
