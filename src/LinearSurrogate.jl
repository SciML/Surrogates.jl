mutable struct LinearSurrogate{X,Y,C,L,U} <: AbstractSurrogate
    x::X
    y::Y
    coeff::C
    lb::L
    ub::U
end

function (lin::LinearSurrogate)(val::Number)
    return lin.coeff[1] + val*lin.coeff[2]
end

function LinearSurrogate(x,y,lb::Number,ub::Number)
    df = DataFrame(X=x,Y=y)
    ols = lm(@formula(Y ~ X), df)
    LinearSurrogate(x,y,coef(ols),lb,ub)
end

function add_point!(my_linear::LinearSurrogate,new_x,new_y)
    if size(my_linear.x,2) == 1
        #1D
        if length(new_x) == 1
            push!(my_linear.x,new_x)
            push!(my_linear.y,new_y)
            df = DataFrame(X=my_linear.x,Y=my_linear.y)
            md = lm(@formula(Y ~ X), df)
            my_linear.coeff = coef(md)
        else
            append!(my_linear.x,new_x)
            append!(my_linear.y,new_y)
            df = DataFrame(X=my_linear.x,Y=my_linear.y)
            md = lm(@formula(Y ~ X), df)
            my_linear.coeff = coef(md)
        end
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
