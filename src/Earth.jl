"""
    EarthSurrogate(x, y, lb, ub; penalty = 2.0, n_min_terms = 2,
        n_max_terms = 10, rel_res_error = 1.0e-2, rel_GCV = 1.0e-2,
        maxiters = 100)

Multivariate adaptive regression splines (MARS) surrogate.

The model is a sum of one-dimensional hinge functions about the mean response,

```math
\\hat f(p) = \\bar y + \\sum_{t} c_t \\, h_t(p), \\qquad
h_t(p) = \\max(0, \\pm(p_{j_t} - k_t))
```

fitted by a forward pass that greedily adds *reflected pairs* — a hinge and its
mirror image about the same knot `k_t` in the same coordinate `j_t` — followed
by backward pruning on the generalized cross-validation (GCV) criterion.

Knots are drawn from the sampled coordinates, so the surrogate is piecewise
linear with breakpoints at the data and is continuous but not differentiable
there. It is a regression surrogate, not an interpolant.

# Fields

  - `x`: training inputs.
  - `y`: training responses.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.
  - `basis`: selected hinge basis terms, as `HingeTerm`s.
  - `coeff`: fitted basis coefficients, one per basis term.
  - `penalty`: generalized cross-validation penalty.
  - `n_min_terms`: minimum number of retained basis terms.
  - `n_max_terms`: maximum number of reflected pairs added by the forward pass.
  - `rel_res_error`: relative residual-error threshold for adding terms.
  - `rel_GCV`: relative generalized-cross-validation threshold for pruning.
  - `intercept`: mean response, the value the model takes where every hinge is
    inactive.
  - `maxiters`: maximum number of forward-pass iterations.

# Arguments

  - `x`: sample locations, as numbers for one-dimensional inputs or as
    equal-length tuples or vectors otherwise.
  - `y`: observed values at `x`, one number per sample.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain matching `lb`.

# Keywords

  - `penalty = 2.0`: GCV complexity penalty. Each retained term costs
    `1 + penalty / 2` effective parameters, so a larger penalty prunes harder.
  - `n_min_terms::Int = 2`: minimum number of individual basis functions the
    backward pass will leave in place.
  - `n_max_terms::Int = 10`: maximum number of *reflected pairs* the forward
    pass will add, so at most `2 * n_max_terms` basis functions.
  - `rel_res_error = 1.0e-2`: a candidate pair is added only if it cuts the
    residual sum of squares by at least this *fraction* of the current residual.
  - `rel_GCV = 1.0e-2`: a term is pruned only if dropping it cuts the GCV
    score by at least this fraction of the current score.
  - `maxiters = 100`: maximum forward-selection iterations.

# Returns

A callable `EarthSurrogate` supporting `update!(surrogate, x_new, y_new)`,
which refits the basis and coefficients after adding observations.

!!! note "Additive terms only"

    Both passes select hinges in a single coordinate; products of hinges across
    coordinates — the interaction terms of full MARS — are never formed, so the
    model is additive in the input coordinates.

# Element types

The design matrices are built in `float(eltype)` of the samples, so `Float32`
and `BigFloat` inputs keep their precision and integer or rational inputs are
promoted the way `\\` would promote them. Knots are stored at the samples' own
type, so an integer design keeps exact knots.

# Differentiability

Evaluation is differentiable in the query point with both ForwardDiff and
Zygote. The hinges make the surrogate only piecewise differentiable: at a knot
the derivative jumps, and the value returned there is whichever one-sided
derivative `max` selects.

# Example

```julia
using Surrogates

x = sample(20, 0.0, 5.0, SobolSample())
y = @. 2x + x^2
surrogate = EarthSurrogate(x, y, 0.0, 5.0)
surrogate(3.0)
```
"""
mutable struct EarthSurrogate{X, Y, L, U, B, C, P, M, N, R, G, I, T} <:
    AbstractDeterministicSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    basis::B
    coeff::C
    penalty::P
    n_min_terms::M
    n_max_terms::N
    rel_res_error::R
    rel_GCV::G
    intercept::I
    maxiters::T
end

"""
    HingeTerm(dim, knot, mirror)

One hinge basis function of an [`EarthSurrogate`](@ref): `max(0, p[dim] - knot)`,
or the reflection `max(0, knot - p[dim])` when `mirror` is `true`.

A hinge and its mirror are added as a pair, so together they represent a change
of slope at `knot` rather than a one-sided ramp.
"""
struct HingeTerm{K}
    dim::Int
    knot::K
    mirror::Bool
end

# A one-dimensional sample is its own only coordinate, so reading samples through
# these lets one forward and one backward pass serve both input layouts.
_coord(p::Number, ::Int) = p
_coord(p, dim::Int) = p[dim]
_ndims(p::Number) = 1
_ndims(p) = length(p)

@inline function _eval_term(term::HingeTerm, p)
    z = _coord(p, term.dim)
    return term.mirror ? max(0, term.knot - z) : max(0, z - term.knot)
end

_fill_column!(X, col, term, x) = (X[:, col] .= _eval_term.((term,), x); X)

function _design_matrix(x, basis)
    X = Matrix{float(eltype(first(x)))}(undef, length(x), length(basis))
    for (col, term) in enumerate(basis)
        _fill_column!(X, col, term, x)
    end
    return X
end

# Master rejected a candidate on `cond(X'X) > 1e8`, i.e. `cond(X) > 1e4`. Column
# pivoting leaves `|R[i, i]|` non-increasing, and `|R[1, 1]| / |R[m, m]|`
# lower-bounds `cond(X)`, so the same screen — to within that slack — is a read of
# the factorization that already solves the system, not a fresh SVD.
const _EARTH_COND_TOL = 1.0e-4

"""
    _lstsq(X, yc)

Least-squares coefficients of `yc` on the columns of `X`, or `nothing` when the
design cannot support them.

`nothing` marks the two cases the forward pass must skip rather than select on:
a knot at an extreme sampled coordinate, whose mirrored hinge is an all-zero
column, and fewer samples than columns, where `\\` returns a minimum-norm
solution that fits the samples and says nothing between them.
"""
function _lstsq(X, yc)
    size(X, 1) >= size(X, 2) || return nothing
    F = qr(X, ColumnNorm())
    r = abs.(diag(F.R))
    (isempty(r) || last(r) <= _EARTH_COND_TOL * first(r)) && return nothing
    return F \ yc
end

function _sse(X, coeff, yc)
    s = zero(promote_type(eltype(X), eltype(yc), eltype(coeff)))
    @inbounds for i in axes(X, 1)
        r = yc[i]
        for j in axes(X, 2)
            r -= X[i, j] * coeff[j]
        end
        s += r^2
    end
    return s
end

# The mean is carried separately as `intercept`, so the basis is fitted to what
# that leaves unexplained; regressing raw `y` would count the mean twice.
_center(y) = y .- sum(y) / length(y)

"""
    _gcv(sse, n, m, penalty)

Generalized cross-validation score of a fit with `m` hinge terms and residual
sum of squares `sse` over `n` samples.

Friedman's criterion (MARS, Ann. Statist. 19(1), eq. 30) as the reference
implementations write it: `(sse / n) / (1 - C / n)^2` with
`C = M + penalty * (M - 1) / 2`, where `M` counts every term the model fits, the
constant included. The constant is carried outside `basis`, hence `M = m + 1`;
counting only the hinges would understate every model by `1 + penalty / 2`
effective parameters.

The denominator vanishes as `C` reaches `n` and recovers beyond it, so a basis
that saturates the samples scores `Inf` rather than spuriously well.
"""
function _gcv(sse, n, m, penalty)
    nterms = m + 1
    effective_params = nterms + penalty * (nterms - 1) / 2
    effective_params >= n && return convert(float(typeof(sse)), Inf)
    return sse / (n * (1 - effective_params / n)^2)
end

"""
    _forward_pass(x, y, n_max_terms, rel_res_error, maxiters)

Greedily grow a hinge basis, one reflected pair per iteration.

Candidates are the reflected pairs at every sampled coordinate, `n * d` of them.
Pairing each hinge with a mirror at a *different* knot, as the ND pass used to,
costs `(n * d)^2` candidates and reaches no further: a pair at unrelated knots is
two independent ramps that later iterations can add separately.
"""
function _forward_pass(x, y, n_max_terms, rel_res_error, maxiters)
    n = length(x)
    d = _ndims(first(x))
    yc = _center(y)
    basis = HingeTerm{typeof(_coord(first(x), 1))}[]
    # The intercept-only residual, which the first pair has to improve on.
    best_sse = sum(abs2, yc)
    pairs = 0
    for _ in 1:maxiters
        pairs < n_max_terms || break
        # The retained columns are identical across candidates, so they are
        # filled once and only the two trial columns are rewritten.
        held = length(basis)
        X = Matrix{float(eltype(first(x)))}(undef, n, held + 2)
        for col in 1:held
            _fill_column!(X, col, basis[col], x)
        end
        best_pair = nothing
        for i in 1:n, j in 1:d
            knot = _coord(x[i], j)
            pair = (HingeTerm(j, knot, false), HingeTerm(j, knot, true))
            _fill_column!(X, held + 1, pair[1], x)
            _fill_column!(X, held + 2, pair[2], x)
            coeff = _lstsq(X, yc)
            coeff === nothing && continue
            sse = _sse(X, coeff, yc)
            # Strictly, and by the margin: a response the basis already
            # explains exactly leaves `best_sse` at zero, where a relative
            # margin alone admits everything.
            if sse < best_sse && best_sse - sse >= rel_res_error * best_sse
                best_sse = sse
                best_pair = pair
            end
        end
        best_pair === nothing && break
        push!(basis, best_pair[1], best_pair[2])
        pairs += 1
    end
    isempty(basis) && throw(
        ArgumentError(
            "EarthSurrogate added no basis term, leaving only the intercept: no \
            reflected pair cut the residual sum of squares by the required \
            fraction rel_res_error = $rel_res_error. Lower rel_res_error, or \
            supply samples whose response varies beyond that tolerance."
        )
    )
    return basis
end

"""
    _backward_pass(x, y, n_min_terms, basis, penalty, rel_GCV)

Prune `basis` in place for as long as removal improves the GCV score.

The forward pass adds pairs on residual error alone and so overfits by
construction; pruning one term at a time on GCV trades them back against their
cost.
"""
function _backward_pass(x, y, n_min_terms, basis, penalty, rel_GCV)
    n = length(x)
    yc = _center(y)
    coeff = _lstsq(_design_matrix(x, basis), yc)
    coeff === nothing && return basis
    current_gcv = _gcv(
        _sse(_design_matrix(x, basis), coeff, yc), n, length(basis), penalty
    )
    while length(basis) > max(n_min_terms, 1)
        best_gcv = convert(typeof(current_gcv), Inf)
        best_idx = 0
        for i in eachindex(basis)
            trial = deleteat!(copy(basis), i)
            X = _design_matrix(x, trial)
            trial_coeff = _lstsq(X, yc)
            trial_coeff === nothing && continue
            gcv = _gcv(_sse(X, trial_coeff, yc), n, length(trial), penalty)
            if best_idx == 0 || gcv < best_gcv
                best_gcv = gcv
                best_idx = i
            end
        end
        best_idx == 0 && break
        # Stop unless dropping a term improves GCV by the required margin —
        # except while the basis saturates the samples, where every score is
        # `Inf` and GCV ranks nothing. Pruning has to continue there, or a
        # penalty heavy enough to saturate returns a *larger* basis than a
        # lighter one.
        if isfinite(current_gcv)
            current_gcv - best_gcv >= rel_GCV * current_gcv || break
        end
        deleteat!(basis, best_idx)
        current_gcv = best_gcv
    end
    return basis
end

function _fit_coeff(x, y, basis)
    coeff = _lstsq(_design_matrix(x, basis), _center(y))
    coeff === nothing && throw(
        ArgumentError(
            "EarthSurrogate could not fit its retained basis of \
            $(length(basis)) term(s): the design matrix at the $(length(x)) \
            samples is rank deficient. Spread the samples across the domain."
        )
    )
    return coeff
end

function EarthSurrogate(
        x, y, lb, ub; penalty::Number = 2.0, n_min_terms::Int = 2,
        n_max_terms::Int = 10, rel_res_error::Number = 1.0e-2,
        rel_GCV::Number = 1.0e-2, maxiters = 100
    )
    forward = _forward_pass(x, y, n_max_terms, rel_res_error, maxiters)
    basis = _backward_pass(x, y, n_min_terms, forward, penalty, rel_GCV)
    return EarthSurrogate(
        x, y, lb, ub, basis, _fit_coeff(x, y, basis), penalty, n_min_terms,
        n_max_terms, rel_res_error, rel_GCV, sum(y) / length(y), maxiters
    )
end

# A scalar accumulator, rebound rather than mutated: evaluation does not
# allocate, and both forward and reverse mode differentiate through it.
function (earth::EarthSurrogate)(val)
    _check_dimension(earth, val)
    v = earth.intercept
    for i in eachindex(earth.coeff)
        v += earth.coeff[i] * _eval_term(earth.basis[i], val)
    end
    return v
end

function SurrogatesBase.update!(earth::EarthSurrogate, x_new, y_new)
    earth.x, earth.y = _append_samples(earth.x, earth.y, x_new, y_new)
    forward = _forward_pass(
        earth.x, earth.y, earth.n_max_terms, earth.rel_res_error, earth.maxiters
    )
    earth.basis = _backward_pass(
        earth.x, earth.y, earth.n_min_terms, forward, earth.penalty, earth.rel_GCV
    )
    earth.coeff = _fit_coeff(earth.x, earth.y, earth.basis)
    earth.intercept = sum(earth.y) / length(earth.y)
    return nothing
end
