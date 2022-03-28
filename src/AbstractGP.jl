using AbstractGPs

mutable struct AbstractGPStruct{X, Y, GP, GP_P, S} <: AbstractSurrogate
    x::X
    y::Y
    gp::GP 
    gp_posterior::GP_P
    Σy::S
 end

# constructor
function AbstractGPSurrogate(x, y; gp = GP(Matern52Kernel()), Σy = 0.1)
    AbstractGPStruct(x, y, gp, posterior(gp(x, Σy),y), Σy)
 end

# predictor 
function (g::AbstractGPStruct)(val)
    return first(mean(g.gp_posterior([val])))
end

# for add point
# copies of x and y need to be made because we get 
#"Error: cannot resize array with shared data " if we push! directly to x and y  
function add_point!(g::AbstractGPStruct, new_x, new_y)
    if new_x in g.x
        println("Adding a sample that already exists, cannot build AbstracgGPSurrogate.")
        return
    end
    x_copy = copy(g.x)
    push!(x_copy, new_x)
    y_copy = copy(g.y)
    push!(y_copy, new_y) 
    updated_posterior = posterior(g.gp(x_copy, g.Σy), y_copy)
    g.x, g.y, g.gp_posterior = x_copy, y_copy, updated_posterior
    nothing
end


# returns diagonal elements of cov(f)
function var_at_point(g::AbstractGPStruct, val)
    return var(g.gp_posterior(val))
end

function std_error_at_point(g::AbstractGPStruct, val)
    return sqrt(first(var(g.gp_posterior([val]))))
end


# Log marginal posterior predictive probability.
function logpdf_surrogate(g::AbstractGPStruct)
    return logpdf(g.gp_posterior(g.x), g.y)
end
