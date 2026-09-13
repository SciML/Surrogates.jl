module SurrogatesAbstractGPsExt

using AbstractGPs: GP, posterior
using Distributions: logpdf
using KernelFunctions: Matern52Kernel
using Statistics: mean, var
using Surrogates: AbstractGPSurrogate, Surrogates, _append_samples, _match_stored

import SurrogatesBase

# constructor
function Surrogates.AbstractGPSurrogate(x, y; gp = GP(Matern52Kernel()), Σy = 0.1)
    return AbstractGPSurrogate(x, y, gp, posterior(gp(x, Σy), y), Σy)
end

# predictor
#
# A `d`-dimensional point may be written as a tuple or as a coordinate vector,
# and every other surrogate accepts both. Here the point is handed straight to
# the kernel, which compares it against the stored design: a coordinate vector
# against a tuple-stored design raised `DimensionMismatch`, so the surrogate
# could not be differentiated by `ForwardDiff.gradient`, which supplies a vector.
# `_match_stored` is the same conversion `update!` uses.
function (g::AbstractGPSurrogate)(val)
    return only(mean(g.gp_posterior([_match_stored(first(g.x), val)])))
end

function Surrogates.std_error_at_point(g::AbstractGPSurrogate, val)
    point = _match_stored(first(g.x), val)
    return sqrt(only(var(g.gp_posterior([point]))))
end

function SurrogatesBase.update!(g::AbstractGPSurrogate, new_x, new_y)
    # `_append_samples` is the package's own rule for telling one new sample
    # from a batch, and for writing a point in the representation the stored
    # design uses. Iterating `new_x` directly, as this did, walked the
    # *coordinates* of a single multidimensional point.
    n = length(g.x)
    x_all, y_all = _append_samples(g.x, g.y, new_x, new_y)
    for p in view(x_all, (n + 1):length(x_all))
        in(p, g.x) &&
            error("Adding a sample that already exists, cannot update AbstractGPSurrogate!")
    end
    g.x = x_all
    g.y = y_all
    g.gp_posterior = posterior(g.gp(g.x, g.Σy), g.y)
    return nothing
end

function SurrogatesBase.finite_posterior(g::AbstractGPSurrogate, xs)
    return g.gp_posterior(xs)
end

# Log marginal posterior predictive probability.
function Surrogates.logpdf_surrogate(g::AbstractGPSurrogate)
    return logpdf(g.gp_posterior(g.x), g.y)
end


# ---- SurrogatesBase parameter interface -----------------------------------
#
# The posterior is what conditioning on the data produced; the prior process and
# the observation noise are the configuration that produced it.

SurrogatesBase.parameters(g::AbstractGPSurrogate) = (;
    gp_posterior = g.gp_posterior,
)
SurrogatesBase.hyperparameters(g::AbstractGPSurrogate) = (; gp = g.gp, Sigma_y = g.Σy)

end # module
