using Flux
using Flux: @epochs
mutable struct NeuralSurrogate{D,M,L,O,P,N,A,U} <: AbstractSurrogate
    data::D
    model::M
    loss::L
    opt::O
    ps::P
    n_echos::N
    lb::A
    ub::U
 end

function NeuralSurrogate(x,y,lb::Number,ub::Number,model,loss,opt,n_echos)
    x = reshape(x,length(x),1)
    data = []
    for i in 1:length(x)
        push!(data, ([x[i]], y[i]))
    end
    ps = Flux.params(model)
    @epochs n_echos Flux.train!(loss, ps, data, opt)
    return NeuralSurrogate(data,model,loss,opt,ps,n_echos,lb,ub)
end

function (my_neural::NeuralSurrogate)(val::Number)
    return my_neural.model([val])
end

function add_point!(my_n::NeuralSurrogate,x_new,y_new)
    if length(my_n.lb) == 1
        for j = 1:length(x_new)
            push!(my_n.data,([x_new[j]],y_new[j]))
        end
        println(my_n.data)
        @epochs my_n.n_echos Flux.train!(my_n.loss, my_n.ps, my_n.data, my_n.opt)
    else
        #ND
        #TODO
    end
    nothing
end


function NeuralSurrogate(x,y,lb,ub,model,loss,opt,n_echos)
    
end
