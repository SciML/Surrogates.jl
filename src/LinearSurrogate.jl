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
    if size(my_linear.x,2) == 1
        #1D
        my_linear.x = vcat(my_linear.x,new_x)
        my_linear.y = vcat(my_linear.y,new_y)
        md = lm(reshape(my_linear.x,length(my_linear.x),1),my_linear.y)
        my_linear.coeff = coef(md)
    else
        #ND
        my_linear.x = vcat(my_linear.x,new_x)
        my_linear.y = vcat(my_linear.y,new_y)
        md = lm(my_linear.x,my_linear.y)
        my_linear.coeff = coef(md)
    end
    nothing
end

function (lin::LinearSurrogate)(val)
    return val*lin.coeff
end

function LinearSurrogate(x,y,lb,ub)
    ols = lm(x,y)
    return LinearSurrogate(x,y,coef(ols),lb,ub)
end
