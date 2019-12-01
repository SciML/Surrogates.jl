using Flux
using Flux: @epochs
mutable struct NeuralSurrogate{X,Y,M,L,O,P,N,A,U} <: AbstractSurrogate
    x::X
    y::Y
    model::M
    loss::L
    opt::O
    ps::P
    n_echos::N
    lb::A
    ub::U
 end

function NeuralSurrogate(x,y,lb::Number,ub::Number,model,loss,opt,n_echos)
    X = reshape(x,length(x),1)
    data = Tuple{Array{eltype(X[1]),1},eltype(y[1])}[]
    for i in 1:length(x)
        push!(data, ([X[i]], y[i]))
    end
    ps = Flux.params(model)
    @epochs n_echos Flux.train!(loss, ps, data, opt)
    return NeuralSurrogate(x,y,model,loss,opt,ps,n_echos,lb,ub)
end

function (my_neural::NeuralSurrogate)(val::Number)
    return first(my_neural.model([val]))
end

function add_point!(my_n::NeuralSurrogate,x_new,y_new)
    if length(my_n.lb) == 1
        #1D
        X = reshape(my_n.x,length(my_n.x),1)
        data = Tuple{Array{eltype(X[1]),1},eltype(my_n.y[1])}[]
        for i in 1:length(my_n.x)
            push!(data, ([X[i]], my_n.y[i]))
        end
        for j = 1:length(x_new)
            push!(data,([x_new[j]],y_new[j]))
        end
        my_n.x = vcat(my_n.x,x_new)
        my_n.y = vcat(my_n.y,y_new)
        @epochs my_n.n_echos Flux.train!(my_n.loss, my_n.ps, data, my_n.opt)
    else
        #ND

        n_previous = length(my_n.x)
        a = vcat(my_n.x,x_new)
        n_after = length(a)
        dim_new = n_after - n_previous
        n = length(my_n.x)
        d = length(my_n.x[1])
        tot_dim = n + dim_new
        X = Array{eltype(my_n.x[1]),2}(undef,tot_dim,d)
        data = Tuple{Array{eltype(X[1]),1},eltype(my_n.y[1])}[]
        for j = 1:n
            X[j,:] = vec(collect(my_n.x[j]))
            push!(data, (X[j,:], my_n.y[j]))
        end
        if dim_new == 1
            X[n+1,:] = vec(collect(x_new))
            push!(data, (X[n+1,:], y_new))
        else
            i = 1
            for j = n+1:tot_dim
                X[j,:] = vec(collect(x_new[i]))
                push!(data,(X[j,:], y_new[i]))
                i = i + 1
            end
        end
        my_n.x = vcat(my_n.x,x_new)
        my_n.y = vcat(my_n.y,y_new)
        @epochs my_n.n_echos Flux.train!(my_n.loss, my_n.ps, data, my_n.opt)
    end
    nothing
end

"""
NeuralSurrogate(x,y,lb,ub,model,loss,opt,n_echos)

- model: Flux layers
- loss: loss function
- opt: optimization function

"""
function NeuralSurrogate(x,y,lb,ub,model,loss,opt,n_echos)
    X = Array{Float64,2}(undef,length(x),length(x[1]))
    for j = 1:length(x)
        X[j,:] = vec(collect(x[j]))
    end
    data = Tuple{Array{eltype(X[1]),1},eltype(y[1])}[]
    for i in 1:size(x,1)
        push!(data, (X[i,:], y[i]))
    end
    ps = Flux.params(model)
    @epochs n_echos Flux.train!(loss, ps, data, opt)
    return NeuralSurrogate(x,y,model,loss,opt,ps,n_echos,lb,ub)

end

function (my_neural::NeuralSurrogate)(val)
    v = [val...]
    first(my_neural.model(v))
end
