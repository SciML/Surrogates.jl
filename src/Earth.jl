"""
    EarthSurrogate(x, y, lb, ub; penalty = 2.0, n_min_terms = 2,
        n_max_terms = 10, rel_res_error = 1.0e-2, rel_GCV = 1.0e-2,
        maxiters = 100)

Multivariate adaptive regression splines surrogate.

`EarthSurrogate` fits hinge-function basis terms with a forward pass followed
by backward pruning. The fitted surrogate is callable as `earth(x_new)` and can
be updated with `update!(earth, x_new, y_new)`.

# Fields

  - `x`: training inputs.
  - `y`: training responses.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.
  - `basis`: selected hinge basis terms.
  - `coeff`: fitted basis coefficients.
  - `penalty`: generalized cross-validation penalty.
  - `n_min_terms`: minimum number of retained basis terms.
  - `n_max_terms`: maximum number of basis terms considered during the forward
    pass.
  - `rel_res_error`: relative sum-of-squared-error improvement required to keep
    adding terms in the forward pass.
  - `rel_GCV`: relative generalized-cross-validation improvement required to
    keep pruning terms in the backward pass.
  - `intercept`: intercept fitted jointly with `coeff` by least squares.
  - `maxiters`: maximum number of forward-pass iterations.

# Arguments

  - `x`: sample locations. Use scalars for one-dimensional inputs or tuples for
    multidimensional inputs.
  - `y`: observed values at `x`. Responses must be scalars; vector-valued responses are not supported.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Keywords

  - `penalty`: complexity penalty used during pruning.
  - `n_min_terms`: minimum number of basis terms to keep.
  - `n_max_terms`: maximum number of basis terms to fit.
  - `rel_res_error`: stopping threshold for forward selection.
  - `rel_GCV`: stopping threshold for backward pruning.
  - `maxiters`: maximum forward-selection iterations.

# Returns

An `EarthSurrogate` that satisfies the generic surrogate interface:
`surrogate(x)` evaluates the approximation and `update!(surrogate, x, y)`
refits after adding a sample.
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

_hinge(x::Number, knot::Number) = max(0, x - knot)
_hinge_mirror(x::Number, knot::Number) = max(0, knot - x)

# AD-friendly basis term structure for 1D
struct BasisTerm1D{K}
    knot::K
    is_mirror::Bool  # true for hinge_mirror, false for hinge
end

# Evaluate a basis term at a point (AD-friendly)
@inline function _eval_basis_term_1d(term::BasisTerm1D, x::Number)
    if term.is_mirror
        return _hinge_mirror(x, term.knot)
    else
        return _hinge(x, term.knot)
    end
end

# Least-squares design matrix with a leading intercept column, so the constant
# term is fitted jointly with the basis coefficients rather than pinned to the
# response mean (the hinge columns are not orthogonal to the constant vector,
# so pinning it would not give the least-squares fit).
function _design_1d(x, basis)
    n = length(x)
    d = length(basis)
    X = ones(eltype(x[1]), n, d + 1)
    @inbounds for i in 1:n
        for j in 1:d
            X[i, j + 1] = _eval_basis_term_1d(basis[j], x[i])
        end
    end
    return X
end

# Residual sum of squares of the least-squares fit of `y` on the columns of `X`.
function _lsq_sse(X, y)
    return sum(abs2, y .- X * (X \ y))
end

# Split a fitted parameter vector into its intercept and basis coefficients.
_split_beta(β) = (β[1], β[2:end])

function _coeff_1d(x, y, basis)
    return _split_beta(_design_1d(x, basis) \ y)
end

function _forward_pass_1d(x, y, n_max_terms, rel_res_error, maxiters)
    basis = BasisTerm1D[]
    current_sse = +Inf
    num_terms = 0
    iters = 0
    while num_terms < n_max_terms && iters < maxiters
        # Select the knot that minimizes SSE over all candidates, then accept
        # it only if it improves on this sweep's starting SSE by at least
        # rel_res_error. Selection and acceptance must stay separate: folding
        # the threshold into the candidate comparison would make the winner
        # depend on candidate ordering rather than on SSE alone.
        pos_of_knot = 0
        best_sse = +Inf
        for i in 1:length(x)
            # Candidate: the hinge/mirror pair knotted at x[i]
            var_i = x[i]
            new_basis = copy(basis)
            push!(new_basis, BasisTerm1D(var_i, false))  # hinge
            push!(new_basis, BasisTerm1D(var_i, true))   # hinge_mirror
            X = _design_1d(x, new_basis)
            # Skip knots that make the augmented design ill conditioned
            if cond(X' * X) > 1.0e8
                continue
            end
            new_sse = _lsq_sse(X, y)
            if new_sse < best_sse
                pos_of_knot = i
                best_sse = new_sse
            end
        end
        iters = iters + 1
        if pos_of_knot != 0 && best_sse < current_sse * (1 - rel_res_error)
            push!(basis, BasisTerm1D(x[pos_of_knot], false))  # hinge
            push!(basis, BasisTerm1D(x[pos_of_knot], true))   # hinge_mirror
            current_sse = best_sse
            num_terms = num_terms + 1
        else
            break
        end
    end
    if length(basis) == 0
        error("Earth surrogate did not add any term beyond the intercept; loosen rel_res_error or n_max_terms, or check the data scale.")
    end
    return basis
end

# Generalized cross-validation score with the standard MARS effective-parameter
# count. When the effective parameter count reaches the sample count the GCV
# denominator vanishes; return Inf so such models are never selected.
function _gcv(sse, num_terms, penalty, n)
    effect_num_params = num_terms + penalty * (num_terms - 1) / 2
    effect_num_params >= n && return oftype(sse / n, Inf)
    return sse / (n * (1 - effect_num_params / n)^2)
end

function _backward_pass_1d(x, y, n_min_terms, basis, penalty, rel_GCV)
    n = length(x)
    d = length(basis)
    X = _design_1d(x, basis)
    sse = _lsq_sse(X, y)
    current_gcv = _gcv(sse, d, penalty, n)
    num_terms = d
    while (num_terms > n_min_terms)
        #Basis-> select worst performing element-> eliminate it
        if num_terms <= 1
            break
        end
        found_new_to_eliminate = false
        best_removal_idx = 0
        best_new_gcv = +Inf
        for i in 1:num_terms
            current_basis = [basis[j] for j in 1:num_terms if j != i]
            current_base_len = num_terms - 1
            Xi = _design_1d(x, current_basis)
            new_sse = _lsq_sse(Xi, y)
            i_gcv = _gcv(new_sse, current_base_len, penalty, n)
            if i_gcv < best_new_gcv
                best_removal_idx = i
                best_new_gcv = i_gcv
                found_new_to_eliminate = true
            end
        end
        if !found_new_to_eliminate || best_new_gcv >= current_gcv
            break
        end
        # Stop when the GCV improvement is below the relative threshold
        if current_gcv - best_new_gcv < rel_GCV * abs(current_gcv)
            break
        else
            num_terms = num_terms - 1
            deleteat!(basis, best_removal_idx)
            current_gcv = best_new_gcv
        end
    end
    return basis
end

function EarthSurrogate(
        x, y, lb::Number, ub::Number; penalty::Number = 2.0,
        n_min_terms::Int = 2, n_max_terms::Int = 10,
        rel_res_error::Number = 1.0e-2, rel_GCV::Number = 1.0e-2,
        maxiters = 100
    )
    basis_after_forward = _forward_pass_1d(x, y, n_max_terms, rel_res_error, maxiters)
    basis = _backward_pass_1d(x, y, n_min_terms, basis_after_forward, penalty, rel_GCV)
    intercept, coeff = _coeff_1d(x, y, basis)
    return EarthSurrogate(
        x, y, lb, ub, basis, coeff, penalty, n_min_terms, n_max_terms,
        rel_res_error, rel_GCV, intercept, maxiters
    )
end

function (earth::EarthSurrogate)(val::Number)
    _check_dimension(earth, val)
    return sum(
        earth.coeff[i] * _eval_basis_term_1d(earth.basis[i], val)
            for i in 1:length(earth.coeff)
    ) +
        earth.intercept
end

#ND
# AD-friendly basis term structure for ND
struct BasisTermND{D, K}
    dims::D  # Vector of dimension indices where basis is active (1-based)
    knots::K  # Vector of knot values (one per active dimension)
    is_mirror::Vector{Bool}  # Vector indicating hinge vs hinge_mirror for each active dimension
end

# Evaluate a ND basis term at a point (AD-friendly)
@inline function _eval_basis_term_nd(term::BasisTermND, x)
    result = one(eltype(x[1]))
    for (idx, dim) in enumerate(term.dims)
        knot = term.knots[idx]
        if term.is_mirror[idx]
            result *= _hinge_mirror(x[dim], knot)
        else
            result *= _hinge(x[dim], knot)
        end
    end
    return result
end

# See `_design_1d`: the intercept is a fitted column, not the response mean.
function _design_nd(x, basis)
    n = length(x)
    base_len = length(basis)
    X = ones(eltype(x[1]), n, base_len + 1)
    @inbounds for a in 1:n
        for b in 1:base_len
            X[a, b + 1] = _eval_basis_term_nd(basis[b], x[a])
        end
    end
    return X
end

function _coeff_nd(x, y, basis)
    return _split_beta(_design_nd(x, basis) \ y)
end

function _forward_pass_nd(x, y, n_max_terms, rel_res_error, maxiters)
    n = length(x)
    basis = BasisTermND[]
    current_sse = +Inf
    num_terms = 0
    d = length(x[1])
    iters = 0

    while num_terms < n_max_terms && iters < maxiters
        # As in the 1D pass: pick the SSE-minimizing candidate pair first, then
        # apply the relative acceptance threshold once against the sweep's
        # starting SSE.
        best_term1 = nothing
        best_term2 = nothing
        best_sse = +Inf

        # Candidates are the reflected hinge pair at a single knot (i, j), as in
        # classic MARS: `max(0, x_j - t)` and `max(0, t - x_j)` share the knot
        # `t = x[i][j]`. This is O(n d) least-squares solves per added term.
        for i in 1:n
            for j in 1:d
                knot = x[i][j]
                term1 = BasisTermND([j], [knot], [false])  # hinge
                term2 = BasisTermND([j], [knot], [true])   # hinge_mirror

                new_basis = vcat(basis, [term1, term2])
                X = _design_nd(x, new_basis)
                # Skip pairs that make the augmented design ill conditioned
                if cond(X' * X) > 1.0e8
                    continue
                end
                new_sse = _lsq_sse(X, y)

                if new_sse < best_sse
                    best_term1 = term1
                    best_term2 = term2
                    best_sse = new_sse
                end
            end
        end

        iters = iters + 1
        if best_term1 !== nothing && best_sse < current_sse * (1 - rel_res_error)
            push!(basis, best_term1)
            push!(basis, best_term2)
            current_sse = best_sse
            num_terms = num_terms + 1
        else
            break
        end
    end

    if length(basis) == 0
        error("Earth surrogate did not add any term beyond the intercept; loosen rel_res_error or n_max_terms, or check the data scale.")
    end
    return basis
end

function _backward_pass_nd(x, y, n_min_terms, basis, penalty, rel_GCV)
    n = length(x)
    base_len = length(basis)
    X = _design_nd(x, basis)
    sse = _lsq_sse(X, y)
    current_gcv = _gcv(sse, base_len, penalty, n)
    num_terms = base_len

    while num_terms > n_min_terms
        if num_terms <= 1
            break
        end

        found_new_to_eliminate = false
        best_removal_idx = 0
        best_new_gcv = +Inf

        for i in 1:num_terms
            current_basis = [basis[j] for j in 1:num_terms if j != i]
            current_base_len = num_terms - 1
            Xi = _design_nd(x, current_basis)
            new_sse = _lsq_sse(Xi, y)
            i_gcv = _gcv(new_sse, current_base_len, penalty, n)

            if i_gcv < best_new_gcv
                best_removal_idx = i
                best_new_gcv = i_gcv
                found_new_to_eliminate = true
            end
        end

        if !found_new_to_eliminate || best_new_gcv >= current_gcv
            break
        end

        # Stop when the GCV improvement is below the relative threshold
        if current_gcv - best_new_gcv < rel_GCV * abs(current_gcv)
            break
        end

        # Remove the best candidate
        deleteat!(basis, best_removal_idx)
        num_terms = num_terms - 1
        current_gcv = best_new_gcv
    end

    return basis
end

function EarthSurrogate(
        x, y, lb, ub; penalty::Number = 2.0, n_min_terms::Int = 2,
        n_max_terms::Int = 10, rel_res_error::Number = 1.0e-2,
        rel_GCV::Number = 1.0e-2, maxiters = 100
    )
    basis_after_forward = _forward_pass_nd(x, y, n_max_terms, rel_res_error, maxiters)
    basis = _backward_pass_nd(x, y, n_min_terms, basis_after_forward, penalty, rel_GCV)
    intercept, coeff = _coeff_nd(x, y, basis)
    return EarthSurrogate(
        x, y, lb, ub, basis, coeff, penalty, n_min_terms, n_max_terms,
        rel_res_error, rel_GCV, intercept, maxiters
    )
end

function (earth::EarthSurrogate)(val)
    _check_dimension(earth, val)
    return sum(
        earth.coeff[i] * _eval_basis_term_nd(earth.basis[i], val)
            for i in 1:length(earth.coeff)
    ) +
        earth.intercept
end

function SurrogatesBase.update!(earth::EarthSurrogate, x_new, y_new)
    earth.x, earth.y = _append_samples(earth.x, earth.y, x_new, y_new)
    if first(earth.x) isa Number
        basis_after_forward = _forward_pass_1d(
            earth.x, earth.y, earth.n_max_terms,
            earth.rel_res_error, earth.maxiters
        )
        earth.basis = _backward_pass_1d(
            earth.x, earth.y, earth.n_min_terms,
            basis_after_forward, earth.penalty, earth.rel_GCV
        )
        earth.intercept, earth.coeff = _coeff_1d(earth.x, earth.y, earth.basis)
    else
        basis_after_forward = _forward_pass_nd(
            earth.x, earth.y, earth.n_max_terms,
            earth.rel_res_error, earth.maxiters
        )
        earth.basis = _backward_pass_nd(
            earth.x, earth.y, earth.n_min_terms,
            basis_after_forward, earth.penalty, earth.rel_GCV
        )
        earth.intercept, earth.coeff = _coeff_nd(earth.x, earth.y, earth.basis)
    end
    return nothing
end
