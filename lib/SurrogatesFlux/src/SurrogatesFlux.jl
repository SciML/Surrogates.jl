module SurrogatesFlux

using SurrogatesBase
using Optimisers
using Flux

export NeuralSurrogate, update!

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
    NeuralSurrogate(x, y, lb, ub; model = Chain(Dense(length(x[1]), 1), first), 
                                 loss = (x, y) -> Flux.mse(model(x), y), 
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
function NeuralSurrogate(x, y, lb, ub; model = Chain(Dense(length(x[1]), 1)),
        loss = Flux.mse, opt = Optimisers.Adam(1e-3),
        n_epochs::Int = 10)
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
    nothing
end

end # module
