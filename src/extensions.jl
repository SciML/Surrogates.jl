"""
    AbstractGPSurrogate(x, y; gp = GP(Matern52Kernel()), Σy = 0.1)

Gaussian-process surrogate backed by AbstractGPs.jl.

This type is available when the `AbstractGPs` extension is loaded. It is a
stochastic surrogate: `surrogate(x)` evaluates the posterior mean, and
[`std_error_at_point`](@ref) returns the posterior standard deviation.

# Fields

  - `x`: training inputs.
  - `y`: training responses.
  - `gp`: prior Gaussian process.
  - `gp_posterior`: posterior process fitted to `x` and `y`.
  - `Σy`: observation-noise covariance or scale.

# Arguments

  - `x`: sample locations.
  - `y`: observed values at `x`.

# Keywords

  - `gp`: AbstractGPs prior process.
  - `Σy`: observation noise passed to the finite GP posterior.

# Returns

An `AbstractGPSurrogate` satisfying the stochastic surrogate interface.
"""
mutable struct AbstractGPSurrogate{X, Y, GP, GP_P, S} <: AbstractStochasticSurrogate
    x::X
    y::Y
    gp::GP
    gp_posterior::GP_P
    Σy::S
end

"""
    logpdf_surrogate(surrogate)

Return the log marginal likelihood or log density associated with a stochastic
surrogate.

# Arguments

  - `surrogate`: stochastic surrogate implementation.

# Returns

A scalar log-density value. Methods are supplied by extensions that can compute
the value for their backing model.
"""
function logpdf_surrogate end

"""
    NeuralSurrogate(x, y, lb, ub; model, loss, opt, n_epochs)

Neural-network surrogate backed by Flux.jl.

This type is available when the `Flux` extension is loaded. The fitted
surrogate is callable as `surrogate(x_new)` and supports `update!` by retraining
with appended observations.

# Fields

  - `x`: training inputs.
  - `y`: training responses.
  - `model`: Flux model.
  - `loss`: training loss.
  - `opt`: optimizer state or optimizer object.
  - `ps`: vector of trainable model parameter arrays returned by
    `Optimisers.trainables(model)`.
  - `n_epochs`: number of training epochs.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Arguments

  - `x`: sample locations.
  - `y`: observed values at `x`.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Keywords

  - `model`: Flux model used for prediction.
  - `loss`: loss function used during training.
  - `opt`: optimizer used during training.
  - `n_epochs`: number of training epochs.

# Returns

A `NeuralSurrogate` satisfying the generic surrogate interface.
"""
mutable struct NeuralSurrogate{X, Y, M, L, O, P, N, A, U} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    model::M
    loss::L
    opt::O
    ps::P
    n_epochs::N
    lb::A
    ub::U
end

"""
    PolynomialChaosSurrogate(x, y, lb, ub; orthopolys)

Polynomial-chaos surrogate backed by PolyChaos.jl.

This type is available when the `PolyChaos` extension is loaded. It fits
polynomial-chaos coefficients and evaluates the resulting expansion through the
generic surrogate call interface.

# Fields

  - `x`: training inputs.
  - `y`: training responses.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.
  - `coeff`: fitted polynomial-chaos coefficients.
  - `orthopolys`: orthogonal-polynomial basis.
  - `num_of_multi_indexes`: number of multi-index terms in the expansion.

# Arguments

  - `x`: sample locations.
  - `y`: observed values at `x`.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Returns

A `PolynomialChaosSurrogate` satisfying the generic surrogate interface.
"""
mutable struct PolynomialChaosSurrogate{X, Y, L, U, C, O, N} <:
    AbstractDeterministicSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    coeff::C
    orthopolys::O
    num_of_multi_indexes::N
end

"""
    XGBoostSurrogate(x, y, lb, ub; num_round = 1)

Tree-boosted surrogate backed by XGBoost.jl.

This type is available when the `XGBoost` extension is loaded. The fitted model
is callable through the generic surrogate interface.

# Fields

  - `x`: training inputs.
  - `y`: training responses.
  - `bst`: fitted XGBoost booster.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.
  - `num_round`: number of boosting rounds used for fitting.

# Arguments

  - `x`: sample locations.
  - `y`: observed values at `x`.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Keywords

  - `num_round`: number of boosting rounds.

# Returns

An `XGBoostSurrogate` satisfying the generic surrogate interface.
"""
mutable struct XGBoostSurrogate{X, Y, B, L, U, N} <:
    SurrogatesBase.AbstractDeterministicSurrogate
    x::X
    y::Y
    bst::B
    lb::L
    ub::U
    num_round::N
end

"""
    SVMSurrogate(x, y, lb, ub)

Support-vector-machine surrogate backed by LIBSVM.jl.

This type is available when the `LIBSVM` extension is loaded. The fitted model
is callable through the generic surrogate interface and can be updated by
refitting after adding samples.

# Fields

  - `x`: training inputs.
  - `y`: training responses.
  - `model`: fitted LIBSVM model.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Arguments

  - `x`: sample locations.
  - `y`: observed values at `x`.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Returns

An `SVMSurrogate` satisfying the generic surrogate interface.
"""
mutable struct SVMSurrogate{X, Y, M, L, U} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    model::M
    lb::L
    ub::U
end

"""
    MOE(x, y, expert_types; ndim = 1, n_clusters = 2, quantile = 10)

Mixture-of-experts surrogate backed by GaussianMixtures.jl.

This type is available when the `GaussianMixtures` extension is loaded. It
clusters the input data, fits one local surrogate per cluster, and dispatches
evaluation to the selected expert.

# Fields

  - `x`: training inputs.
  - `y`: training responses.
  - `c`: fitted Gaussian-mixture clusters.
  - `d`: frozen distributions corresponding to the clusters.
  - `m`: fitted local surrogate models.
  - `e`: expert surrogate configurations.
  - `nd`: number of input dimensions.
  - `nc`: number of clusters.

# Arguments

  - `x`: sample locations.
  - `y`: observed values at `x`.
  - `expert_types`: surrogate configurations used to build local experts.

# Keywords

  - `ndim`: number of input dimensions.
  - `n_clusters`: number of mixture clusters.
  - `quantile`: quantile used when assigning local training data.

# Returns

An `MOE` surrogate satisfying the generic surrogate interface.
"""
mutable struct MOE{X, Y, C, D, M, E, ND, NC} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    c::C #clusters (C) - vector of gaussian mixture clusters
    d::D #distributions (D) - vector of frozen multivariate distributions
    m::M # models (M) - vector of trained models correspnoding to clusters (C) and distributions (D)
    e::E #expert types
    nd::ND #number of dimensions
    nc::NC #number of clusters
end

"""
    GENNSurrogate(x, y, dydx, lb, ub; model, opt, n_epochs, gamma,
        is_normalize = true)

Gradient-enhanced neural-network surrogate backed by Flux.jl.

This type is available when the `Flux` extension is loaded. It trains on both
function observations and derivative observations. The fitted surrogate is
callable as `surrogate(x_new)`, supports [`predict_derivative`](@ref), and can
be updated with new samples.

# Fields

  - `x`: training inputs.
  - `y`: training responses.
  - `dydx`: derivative observations stored as
    `(n_outputs, n_inputs, n_samples)`.
  - `model`: Flux model.
  - `opt`: optimizer state or optimizer object.
  - `ps`: vector of trainable model parameter arrays returned by
    `Optimisers.trainables(model)`.
  - `n_epochs`: number of training epochs.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.
  - `gamma`: derivative-loss weight.
  - `x_mean`: input normalization mean, or `nothing`.
  - `x_std`: input normalization scale, or `nothing`.
  - `y_mean`: output normalization mean, or `nothing`.
  - `y_std`: output normalization scale, or `nothing`.
  - `is_normalize`: whether normalization is enabled.

# Arguments

  - `x`: sample locations.
  - `y`: observed values at `x`.
  - `dydx`: derivative observations.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Keywords

  - `model`: Flux model used for prediction.
  - `opt`: optimizer used during training.
  - `n_epochs`: number of training epochs.
  - `gamma`: derivative-loss weight.
  - `is_normalize`: whether to normalize inputs and outputs during training.

# Returns

A `GENNSurrogate` satisfying the generic surrogate interface.
"""
mutable struct GENNSurrogate{X, Y, D, M, O, P, N, A, U, G, XM, XS, YM, YS, IN} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    dydx::D  # gradients (n_outputs x n_inputs x n_samples)
    model::M
    opt::O
    ps::P
    n_epochs::N
    lb::A
    ub::U
    gamma::G  # gradient-enhancement coefficient
    x_mean::XM  # normalization parameters (nothing if not normalized)
    x_std::XS
    y_mean::YM
    y_std::YS
    is_normalize::IN  # whether normalization is enabled
end

"""
    predict_derivative(genn::GENNSurrogate, x)

Evaluate the derivative predicted by a gradient-enhanced neural surrogate.

# Arguments

  - `genn::GENNSurrogate`: fitted gradient-enhanced neural surrogate.
  - `x`: input location where the derivative should be predicted.

# Returns

The derivative prediction at `x`, using the normalization convention stored in
`genn`.
"""
function predict_derivative end
