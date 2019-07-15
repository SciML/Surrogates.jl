mutable struct LinearSurrogate{X,Y,C,L,U} <: AbstractSurrogate
    x::X
    y::Y
    coeff::C
    L::L
    U::U

end


function (lin::LinearSurrogate)(val::Number)
    return coef(lin)[1] + val*coef(lin)[2]
end

function LinearSurrogate(x,y,lb::Number,ub::Number)
    df = DataFrame(X=x,Y=y)
    ols = lm(@formula(Y ~ X), df)
    LinearSurrogate(x,y,coef(ols),lb,ub)
end

function (lin::LinearSurrogate)(val)
    return coef(lin)[1] + val*coef[lin][2]
end

function LinearSurrogate(x,y,lb,ub)
    DF = ??
    ols = ??
    LinearSurrogate(x,y,coef(ols),lb,ub)
end

function add_point!(my_linear::LinearSurrogate,x_new,y_new)

end
