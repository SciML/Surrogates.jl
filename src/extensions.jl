using SurrogatesBase

mutable struct AbstractGPSurrogate{X, Y, GP, GP_P, S} <: AbstractStochasticSurrogate
    x::X
    y::Y
    gp::GP
    gp_posterior::GP_P
    Σy::S
end

function logpdf_surrogate end
function std_error_at_point end

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

mutable struct XGBoostSurrogate{X, Y, B, L, U, N} <:
    SurrogatesBase.AbstractDeterministicSurrogate
    x::X
    y::Y
    bst::B
    lb::L
    ub::U
    num_round::N
end

mutable struct SVMSurrogate{X, Y, M, L, U} <: AbstractDeterministicSurrogate
    x::X
    y::Y
    model::M
    lb::L
    ub::U
end

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
