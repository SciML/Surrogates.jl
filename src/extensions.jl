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

