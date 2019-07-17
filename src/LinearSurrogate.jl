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
    if (length(new_x) == 1 && length(new_x[1]) == 1) || ( length(new_x) > 1 && length(new_x[1]) == 1 && length(my_linear.ub[1])>1)
        push!(my_linear.x,new_x)
        push!(my_linear.y,new_y)
        if length(my_linear.ub[1]) == 1
            df = DataFrame(X=my_linear.x,Y=my_linear.y)
            md = lm(@formula(Y ~ X), df)
            my_linear.coeff = coef(md)
        else
            #TODO
        end
    else
        append!(my_linear.x,new_x)
        append!(my_linear.y,new_y)
        if length(my_linear.lb[1]) == 1
            df = DataFrame(X=my_linear.x,Y=my_linear.y)
            md = lm(@formula(Y ~ X), df)
            my_linear.coeff = coef(md)
        else
            #TODO
        end
    end
    nothing
end

function (lin::LinearSurrogate)(val)

end

function LinearSurrogate(x,y,lb,ub)


end
