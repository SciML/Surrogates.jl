mutable struct LinearSurrogate{X,Y,C,L,U} <: AbstractSurrogate
    x::X
    y::Y
    coeff::C
    lb::L
    ub::U
end

function LinearSurrogate(x,y,lb::Number,ub::Number)
    ols = lm(reshape(x,length(x),1),y)
    LinearSurrogate(x,y,coef(ols),lb,ub)
end

function add_point!(my_linear::LinearSurrogate,new_x,new_y)
    if length(my_linear.lb) == 1
        #1D
        my_linear.x = vcat(my_linear.x,new_x)
        my_linear.y = vcat(my_linear.y,new_y)
        md = lm(reshape(my_linear.x,length(my_linear.x),1),my_linear.y)
        my_linear.coeff = coef(md)
    else
        #ND
        n_previous = length(my_linear.x)
        a = vcat(my_linear.x,new_x)
        n_after = length(a)
        dim_new = n_after - n_previous
        n = length(my_linear.x)
        d = length(my_linear.x[1])
        tot_dim = n + dim_new
        X = Array{Float64,2}(undef,tot_dim,d)
        for j = 1:n
            X[j,:] = vec(collect(my_linear.x[j]))
        end
        if dim_new == 1
            X[n+1,:] = vec(collect(new_x))
        else
            i = 1
            for j = n+1:tot_dim
                X[j,:] = vec(collect(new_x[i]))
                i = i + 1
            end
        end
        my_linear.x = vcat(my_linear.x,new_x)
        my_linear.y = vcat(my_linear.y,new_y)
        md = lm(X,my_linear.y)
        my_linear.coeff = coef(md)
    end
    nothing
end

function (lin::LinearSurrogate)(val)
    return vec(collect(val))'*lin.coeff
end

"""
LinearSurrogate(x,y,lb,ub)

Builds a linear surrogate using GLM.jl

"""
function LinearSurrogate(x,y,lb,ub)
    X = Array{Float64,2}(undef,length(x),length(x[1]))
    for j = 1:length(x)
        X[j,:] = vec(collect(x[j]))
    end
    ols = lm(X,y)
    return LinearSurrogate(x,y,coef(ols),lb,ub)
end
