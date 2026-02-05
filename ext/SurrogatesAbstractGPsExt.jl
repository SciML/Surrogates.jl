module SurrogatesAbstractGPsExt

using Surrogates: AbstractGPSurrogate, logpdf_surrogate, std_error_at_point, Surrogates
using SurrogatesBase, AbstractGPs

# constructor
function Surrogates.AbstractGPSurrogate(x, y; gp = GP(Matern52Kernel()), Σy = 0.1)
    return AbstractGPSurrogate(x, y, gp, posterior(gp(x, Σy), y), Σy)
end

# predictor
function (g::AbstractGPSurrogate)(val)
    return only(mean(g.gp_posterior([val])))
end

function SurrogatesBase.update!(g::AbstractGPSurrogate, new_x, new_y)
    for x in new_x
        in(x, g.x) &&
            error("Adding a sample that already exists, cannot update AbstractGPSurrogate!")
    end
    g.x = vcat(g.x, new_x)
    g.y = vcat(g.y, new_y)
    g.gp_posterior = posterior(g.gp(g.x, g.Σy), g.y)
    return nothing
end

function SurrogatesBase.finite_posterior(g::AbstractGPSurrogate, xs)
    return g.gp_posterior(xs)
end

function Surrogates.std_error_at_point(g::AbstractGPSurrogate, val)
    return sqrt(only(var(g.gp_posterior([val]))))
end

# Log marginal posterior predictive probability.
function Surrogates.logpdf_surrogate(g::AbstractGPSurrogate)
    return logpdf(g.gp_posterior(g.x), g.y)
end

end # module
