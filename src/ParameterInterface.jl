# SurrogatesBase's optional parameter interface.
#
# `parameters` returns learned state — whatever the fit produced from the data.
# `hyperparameters` returns the tunable fitting configuration — whatever the
# caller chose, or a fitting routine chose on their behalf.
# `update_hyperparameters!` refits that configuration in place.
#
# The split follows that contract: `theta` and `p` are configuration even when
# fitted by maximum likelihood, while `mu`, `b`, `sigma` and the coefficient
# vectors are outputs of the fit. Named tuples throughout.

# ---- Kriging family -------------------------------------------------------

SurrogatesBase.parameters(k::Kriging) = (; mu = k.mu, b = k.b, sigma = k.sigma)
SurrogatesBase.hyperparameters(k::Kriging) = (; p = k.p, theta = k.theta)

SurrogatesBase.parameters(g::GEK) = (; mu = g.mu, b = g.b, sigma = g.sigma)
SurrogatesBase.hyperparameters(g::GEK) = (; p = g.p, theta = g.theta)

SurrogatesBase.parameters(k::KPLS) = (;
    beta = k.beta, gamma = k.gamma, sigma2 = k.sigma2,
    reduced_likelihood_function_value = k.reduced_likelihood_function_value,
)
SurrogatesBase.hyperparameters(k::KPLS) = (; theta = k.theta, n_comp = k.n_comp)

SurrogatesBase.parameters(k::KPLSK) = (;
    beta = k.beta, gamma = k.gamma, sigma2 = k.sigma2,
    reduced_likelihood_function_value = k.reduced_likelihood_function_value,
)
SurrogatesBase.hyperparameters(k::KPLSK) = (;
    theta = k.theta, theta_pls = k.theta_pls, n_comp = k.n_comp,
)

SurrogatesBase.parameters(g::GEKPLS) = (;
    beta = g.beta, gamma = g.gamma, sigma2 = g.sigma2,
)
SurrogatesBase.hyperparameters(g::GEKPLS) = (;
    theta = g.theta, n_comp = g.num_components, delta_x = g.delta,
    extra_points = g.extra_points, nugget = g.nugget, noise = g.noise,
)

# ---- Interpolants ---------------------------------------------------------

SurrogatesBase.parameters(r::RadialBasis) = (; coeff = r.coeff)
SurrogatesBase.hyperparameters(r::RadialBasis) = (;
    radial_function = r.phi, dim_poly = r.dim_poly, scale_factor = r.scale_factor,
    sparse = r.sparse, regularization = r.regularization,
)

SurrogatesBase.parameters(w::Wendland) = (; coeff = w.coeff)
SurrogatesBase.hyperparameters(w::Wendland) = (;
    eps = w.eps, maxiters = w.maxiters, tol = w.tol,
)

SurrogatesBase.parameters(l::LobachevskySurrogate) = (; coeff = l.coeff)
SurrogatesBase.hyperparameters(l::LobachevskySurrogate) = (;
    alpha = l.alpha, n = l.n, sparse = l.sparse,
)

SurrogatesBase.parameters(i::InverseDistanceSurrogate) = NamedTuple()
SurrogatesBase.hyperparameters(i::InverseDistanceSurrogate) = (; p = i.p)

SurrogatesBase.parameters(l::LinearSurrogate) = (; coeff = l.coeff)
SurrogatesBase.hyperparameters(::LinearSurrogate) = NamedTuple()

SurrogatesBase.parameters(s::SecondOrderPolynomialSurrogate) = (; beta = s.β)
SurrogatesBase.hyperparameters(::SecondOrderPolynomialSurrogate) = NamedTuple()

# ---- Composites and basis-selection models --------------------------------

# `basis` and `coeff` are what the forward/backward pass produced; the term
# limits and the GCV penalty are the configuration that drove it.
SurrogatesBase.parameters(e::EarthSurrogate) = (;
    basis = e.basis, coeff = e.coeff, intercept = e.intercept,
    rel_res_error = e.rel_res_error, rel_GCV = e.rel_GCV,
)
SurrogatesBase.hyperparameters(e::EarthSurrogate) = (;
    penalty = e.penalty, n_min_terms = e.n_min_terms,
    n_max_terms = e.n_max_terms, maxiters = e.maxiters,
)

# The fitted state of a variable-fidelity model is its two component surrogates;
# the split point and the correction descriptor are the configuration.
SurrogatesBase.parameters(v::VariableFidelitySurrogate) = (;
    low_fid_surr = v.low_fid_surr, eps_surr = v.eps_surr,
)
SurrogatesBase.hyperparameters(v::VariableFidelitySurrogate) = (;
    num_high_fidel = v.num_high_fidel, eps_structure = v.eps_structure,
)

# ---- Refitting ------------------------------------------------------------

# Copy a freshly fitted model over an existing one, field by field.
#
# Listing fields by hand invites omissions — a refit that updated `theta` but
# left the factorization stale would still answer queries, with the old model's
# numbers. Copying every field makes that impossible.
function _overwrite!(dest::T, src::T) where {T}
    for f in fieldnames(T)
        setfield!(dest, f, getfield(src, f))
    end
    return dest
end

"""
    update_hyperparameters!(surrogate, prior = nothing)

Refit the surrogate's hyperparameters in place and rebuild it on its own design.

Every surrogate here that carries a hyperparameter-fitting routine gets a method:
[`Kriging`](@ref) and [`GEK`](@ref) fit `theta` by maximum likelihood, and
[`KPLS`](@ref), [`KPLSK`](@ref) and [`GEKPLS`](@ref) fit theirs over the PLS
components. Each rebuilds from the design the model already holds, so the call
needs nothing but the model.

`prior` is accepted for interface compatibility and ignored — none of these
models takes a prior over its hyperparameters.

Surrogates whose configuration is chosen by the caller and never fitted, such as
[`RadialBasis`](@ref), [`Wendland`](@ref) or [`LobachevskySurrogate`](@ref),
deliberately have no method: there is nothing to optimize, and a silent no-op
would be worse than a `MethodError` because it would look as though something
had been fitted.
"""
function SurrogatesBase.update_hyperparameters!(k::Kriging, prior = nothing)
    return _overwrite!(
        k, Kriging(k.x, k.y, k.lb, k.ub; p = k.p, optimize_theta = true)
    )
end

function SurrogatesBase.update_hyperparameters!(g::GEK, prior = nothing)
    return _overwrite!(
        g, GEK(g.x, g.y, g.lb, g.ub; p = g.p, optimize_theta = true)
    )
end

# The PLS-based models keep their design bounds in `xl`, a `d x 2` matrix of
# lower and upper limits, and their current `theta` is the natural warm start.
_pls_bounds(m) = (m.xl[:, 1], m.xl[:, 2])

function SurrogatesBase.update_hyperparameters!(k::KPLS, prior = nothing)
    lb, ub = _pls_bounds(k)
    return _overwrite!(
        k, KPLS(k.x, k.y, k.n_comp, lb, ub, k.theta; optimize_theta = true)
    )
end

function SurrogatesBase.update_hyperparameters!(k::KPLSK, prior = nothing)
    lb, ub = _pls_bounds(k)
    # `theta_pls` is the stage-1 KPLS fit and the warm start the constructor
    # expects, not the expanded full-dimensional `theta`.
    return _overwrite!(
        k, KPLSK(k.x, k.y, k.n_comp, lb, ub, k.theta_pls; optimize_theta = true)
    )
end

# `GEKPLS` stores its gradients as an `n x d` matrix but its constructor takes
# them in the shape `Zygote.gradient.` returns — one 1-tuple per sample, wrapping
# that sample's coordinate tuple. Refitting has to convert back.
_gekpls_grads_as_vector(grads) = [
    (Tuple(view(grads, i, :)),) for i in axes(grads, 1)
]

function SurrogatesBase.update_hyperparameters!(g::GEKPLS, prior = nothing)
    lb, ub = _pls_bounds(g)
    return _overwrite!(
        g,
        GEKPLS(
            g.x, g.y, _gekpls_grads_as_vector(g.grads), g.num_components,
            g.delta, lb, ub, g.extra_points, g.theta;
            nugget = g.nugget, noise = g.noise, optimize_theta = true
        )
    )
end
