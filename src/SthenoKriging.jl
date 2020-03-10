mutable struct SthenoKriging{X, Y, GP, TΣy, GP_P} <: AbstractSurrogate
    x::X
    y::Y
    gp::GP
    Σy::TΣy
    gp_posterior::GP_P
 end

 """

 SthenoKriging(x::X, y::Y, GP::Stheno.GP=Stheno.GP(Stheno.EQ(), Stheno.GPC()), σ²=1e-18)

Returns a Kriging (or Gaussian process) surrogate conditioned on the data points `x` and `y`.
The `GP` is the base Gaussian process defined with Stheno.

# Arguments
- `x::X`: Vector containing the observation points
- `y::Y`: Vector containing the observed points
- `GP::Stheno.GP`: The base Gaussian process used to condition over. A simple prior is
provided as default. If there are multiple observation dimensions, a `Tuple` of Gaussian
processes can be passed, otherwise the same process is used across all dimensions.
- `σ²`=1e-18: Variance of the observation noise, default is equivalent to no noise
"""
function SthenoKriging(x, y, gp::Stheno.AbstractGP=Stheno.GP(Stheno.EQ(), Stheno.GPC()), σ²=1e-18)
    gpc = gp.gpc
    gps = ntuple(i -> Stheno.GP(Stheno.EQ(), gpc), length(y[1]))

    return SthenoKriging(x, y, gps, σ²)
end
function SthenoKriging(x, y, gps, σ²=1e-18)
    gp_posteriors = _prepare_gps(x, y, gps, σ²)
    return SthenoKriging(x, y, gps, σ², gp_posteriors)
end



"""
    function (k::SthenoKriging)(val)

Gives the mean predicted value for the `SthenoKriging` object.
"""
function (k::SthenoKriging)(x)
    X = _query_to_colvec(x)
    nobs = length(k.y[1])

    ŷ = [first(Stheno.mean_vector(k.gp_posterior[i], X)) for i=1:nobs]

    return _match_container(ŷ, first(k.y))
end

function std_error_at_point(k::SthenoKriging, x)
    X = _query_to_colvec(x)
    nobs = length(k.y[1])

    std_err = [sqrt(abs(first(Stheno.cov(k.gp_posterior[i], X)))) for i=1:nobs]

    return _match_container(std_err, first(k.y))
end
_query_to_colvec(x::Number) = Stheno.ColVecs(reshape([x], length(x), 1))
_query_to_colvec(x::Tuple) = Stheno.ColVecs(reshape([x...], length(x), 1))
_query_to_colvec(x) = Stheno.ColVecs(reshape(x, length(x), 1))


"""
    add_point!(k::SthenoKriging, x_new, y_new)

Adds the new point(s) and its respective value(s) to the sample points, re-conditioning
the surrogate on the updated values.
"""
function add_point!(k::SthenoKriging, x_new, y_new)
    eltype(x_new) == eltype(k.x) ? append!(k.x, x_new) : push!(k.x, x_new)
    eltype(y_new) == eltype(k.y) ? append!(k.y, y_new) : push!(k.y, y_new)

    k.gp_posterior = _prepare_gps(k.x, k.y, k.gp, k.Σy)
    return nothing
end

function _prepare_gps(x, y, gps, σ²)
    X = Stheno.ColVecs([x[j][i] for i = 1:length(x[1]), j = 1:length(x)])
    Y = [[y[i][j] for i in 1:length(y)] for j in 1:length(first(y))]
    gp_posteriors = _condition_gps(X, Y, gps, σ²)

    return gp_posteriors
end
function _condition_gps(X, Y, gp::Stheno.AbstractGP, σ²)
    nobs = length(Y)
    gp_posts = ntuple(i -> gp, nobs) | ntuple(i -> Obs(gp(X, σ²), Y[i]), nobs)

    return gp_posts
end
function _condition_gps(X, Y, gps, σ²)
    nobs = length(Y)
    @assert length(gps) == nobs
    gp_posts = gps | ntuple(i -> Obs(gps[i](X, σ²), Y[i]), nobs)

    return gp_posts
end
