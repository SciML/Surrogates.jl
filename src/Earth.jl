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

function _coeff_1d(x, y, basis)
    n = length(x)
    d = length(basis)
    X = zeros(eltype(x[1]), n, d)
    @inbounds for i in 1:n
        for j in 1:d
            X[i, j] = _eval_basis_term_1d(basis[j], x[i])
        end
    end
    return (X' * X) \ (X' * y)
end

function _forward_pass_1d(x, y, n_max_terms, rel_res_error, maxiters)
    n = length(x)
    basis = BasisTerm1D[]
    current_sse = +Inf
    intercept = sum(y) / length(y)
    num_terms = 0
    pos_of_knot = 0
    iters = 0
    while num_terms < n_max_terms && iters < maxiters
        #Look for best addition:
        new_addition = false
        for i in 1:length(x)
            #Add or not add the knot var_i?
            var_i = x[i]
            new_basis = copy(basis)
            #select best new pair
            push!(new_basis, BasisTerm1D(var_i, false))  # hinge
            push!(new_basis, BasisTerm1D(var_i, true))   # hinge_mirror
            #find coefficients
            d = length(new_basis)
            X = zeros(eltype(x[1]), n, d)
            @inbounds for k in 1:n
                for j in 1:d
                    X[k, j] = _eval_basis_term_1d(new_basis[j], x[k])
                end
            end
            if (cond(X' * X) > 1.0e8)
                condition_number = false
                new_sse = +Inf
            else
                condition_number = true
                coeff = (X' * X) \ (X' * y)
                new_sse = zero(y[1])
                for k in 1:n
                    val_k = sum(
                        coeff[j] * _eval_basis_term_1d(new_basis[j], x[k])
                            for j in 1:d
                    ) + intercept
                    new_sse = new_sse + (y[k] - val_k)^2
                end
            end
            #is the i-esim the best?
            if (
                    (new_sse < current_sse) && (abs(current_sse - new_sse) >= rel_res_error) &&
                        condition_number
                )
                #Add the hinge function to the basis
                pos_of_knot = i
                current_sse = new_sse
                new_addition = true
            end
        end
        iters = iters + 1
        if new_addition
            push!(basis, BasisTerm1D(x[pos_of_knot], false))  # hinge
            push!(basis, BasisTerm1D(x[pos_of_knot], true))   # hinge_mirror
            num_terms = num_terms + 1
        else
            break
        end
    end
    if length(basis) == 0
        throw("Earth surrogate did not add any term, just the intercept. It is advised to double check the parameters.")
    end
    return basis
end

function _backward_pass_1d(x, y, n_min_terms, basis, penalty, rel_GCV)
    n = length(x)
    d = length(basis)
    intercept = sum(y) / length(y)
    coeff = _coeff_1d(x, y, basis)
    sse = zero(y[1])
    for i in 1:n
        val_i = sum(coeff[j] * _eval_basis_term_1d(basis[j], x[i]) for j in 1:d) + intercept
        sse = sse + (y[i] - val_i)^2
    end
    effect_num_params = d + penalty * (d - 1) / 2
    current_gcv = sse / (n * (1 - effect_num_params / n)^2)
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
            coef = _coeff_1d(x, y, current_basis)
            new_sse = zero(y[1])
            current_base_len = num_terms - 1
            for a in 1:n
                val_a = sum(
                    coef[j] * _eval_basis_term_1d(current_basis[j], x[a])
                        for j in 1:current_base_len
                ) + intercept
                new_sse = new_sse + (y[a] - val_a)^2
            end
            effect_num_params = current_base_len + penalty * (current_base_len - 1) / 2
            i_gcv = new_sse / (n * (1 - effect_num_params / n)^2)
            if i_gcv < best_new_gcv
                best_removal_idx = i
                best_new_gcv = i_gcv
                found_new_to_eliminate = true
            end
        end
        if !found_new_to_eliminate || best_new_gcv >= current_gcv
            break
        end
        if abs(current_gcv - best_new_gcv) < rel_GCV
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
    intercept = sum(y) / length(y)
    basis_after_forward = _forward_pass_1d(x, y, n_max_terms, rel_res_error, maxiters)
    basis = _backward_pass_1d(x, y, n_min_terms, basis_after_forward, penalty, rel_GCV)
    coeff = _coeff_1d(x, y, basis)
    return EarthSurrogate(
        x, y, lb, ub, basis, coeff, penalty, n_min_terms, n_max_terms,
        rel_res_error, rel_GCV, intercept, maxiters
    )
end

function (earth::EarthSurrogate)(val::Number)
    # Check to make sure dimensions of input matches expected dimension of surrogate
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

function _coeff_nd(x, y, basis)
    n = length(x)
    base_len = length(basis)
    X = zeros(eltype(x[1]), n, base_len)
    @inbounds for a in 1:n
        for b in 1:base_len
            X[a, b] = _eval_basis_term_nd(basis[b], x[a])
        end
    end
    return (X' * X) \ (X' * y)
end

function _forward_pass_nd(x, y, n_max_terms, rel_res_error, maxiters)
    n = length(x)
    basis = BasisTermND[]
    current_sse = +Inf
    intercept = sum(y) / length(y)
    num_terms = 0
    d = length(x[1])
    iters = 0

    while num_terms < n_max_terms && iters < maxiters
        new_addition = false
        best_term1 = nothing
        best_term2 = nothing

        for i in 1:n
            for j in 1:d
                for k in 1:n
                    for l in 1:d
                        # Create two new basis terms
                        term1 = BasisTermND([j], [x[i][j]], [false])  # hinge
                        term2 = BasisTermND([l], [x[k][l]], [true])  # hinge_mirror

                        new_basis = vcat(basis, [term1, term2])
                        bas_len = length(new_basis)

                        # Build design matrix
                        X = zeros(eltype(x[1]), n, bas_len)
                        @inbounds for a in 1:n
                            for b in 1:bas_len
                                X[a, b] = _eval_basis_term_nd(new_basis[b], x[a])
                            end
                        end

                        # Check condition number
                        XtX = X' * X
                        if cond(XtX) > 1.0e8
                            continue
                        end

                        # Solve for coefficients
                        coeff = XtX \ (X' * y)

                        # Compute SSE
                        new_sse = zero(y[1])
                        @inbounds for a in 1:n
                            val_a = sum(coeff[b] * X[a, b] for b in 1:bas_len) + intercept
                            new_sse = new_sse + (y[a] - val_a)^2
                        end

                        # Check if this is the best so far
                        if (new_sse < current_sse) &&
                                (abs(current_sse - new_sse) >= rel_res_error)
                            best_term1 = term1
                            best_term2 = term2
                            current_sse = new_sse
                            new_addition = true
                        end
                    end
                end
            end
        end

        iters = iters + 1
        if new_addition
            push!(basis, best_term1)
            push!(basis, best_term2)
            num_terms = num_terms + 1
        else
            break
        end
    end

    if length(basis) == 0
        throw("Earth surrogate did not add any term, just the intercept. It is advised to double check the parameters.")
    end
    return basis
end

function _backward_pass_nd(x, y, n_min_terms, basis, penalty, rel_GCV)
    n = length(x)
    d = length(x[1])
    base_len = length(basis)
    intercept = sum(y) / length(y)
    coeff = _coeff_nd(x, y, basis)

    # Compute initial SSE
    sse = zero(y[1])
    @inbounds for a in 1:n
        val_a = sum(coeff[b] * _eval_basis_term_nd(basis[b], x[a]) for b in 1:base_len) +
            intercept
        sse = sse + (y[a] - val_a)^2
    end

    effect_num_params = base_len + penalty * (base_len - 1) / 2
    current_gcv = sse / (n * (1 - effect_num_params / n)^2)
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
            coef = _coeff_nd(x, y, current_basis)

            new_sse = zero(y[1])
            current_base_len = num_terms - 1
            @inbounds for a in 1:n
                val_a = sum(
                    coef[b] * _eval_basis_term_nd(current_basis[b], x[a])
                        for b in 1:current_base_len
                ) + intercept
                new_sse = new_sse + (y[a] - val_a)^2
            end

            curr_effect_num_params = current_base_len + penalty * (current_base_len - 1) / 2
            i_gcv = new_sse / (n * (1 - curr_effect_num_params / n)^2)

            if i_gcv < best_new_gcv
                best_removal_idx = i
                best_new_gcv = i_gcv
                found_new_to_eliminate = true
            end
        end

        if !found_new_to_eliminate || best_new_gcv >= current_gcv
            break
        end

        if abs(current_gcv - best_new_gcv) < rel_GCV
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
    intercept = sum(y) / length(y)
    basis_after_forward = _forward_pass_nd(x, y, n_max_terms, rel_res_error, maxiters)
    basis = _backward_pass_nd(x, y, n_min_terms, basis_after_forward, penalty, rel_GCV)
    coeff = _coeff_nd(x, y, basis)
    return EarthSurrogate(
        x, y, lb, ub, basis, coeff, penalty, n_min_terms, n_max_terms,
        rel_res_error, rel_GCV, intercept, maxiters
    )
end

function (earth::EarthSurrogate)(val)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(earth, val)
    return sum(
        earth.coeff[i] * _eval_basis_term_nd(earth.basis[i], val)
            for i in 1:length(earth.coeff)
    ) +
        earth.intercept
end

function SurrogatesBase.update!(earth::EarthSurrogate, x_new, y_new)
    return if length(earth.x[1]) == 1
        #1D
        earth.x = vcat(earth.x, x_new)
        earth.y = vcat(earth.y, y_new)
        earth.intercept = sum(earth.y) / length(earth.y)
        basis_after_forward = _forward_pass_1d(
            earth.x, earth.y, earth.n_max_terms,
            earth.rel_res_error, earth.maxiters
        )
        earth.basis = _backward_pass_1d(
            earth.x, earth.y, earth.n_min_terms,
            basis_after_forward, earth.penalty, earth.rel_GCV
        )
        earth.coeff = _coeff_1d(earth.x, earth.y, earth.basis)
        nothing
    else
        #ND
        earth.x = vcat(earth.x, x_new)
        earth.y = vcat(earth.y, y_new)
        earth.intercept = sum(earth.y) / length(earth.y)
        basis_after_forward = _forward_pass_nd(
            earth.x, earth.y, earth.n_max_terms,
            earth.rel_res_error, earth.maxiters
        )
        earth.basis = _backward_pass_nd(
            earth.x, earth.y, earth.n_min_terms,
            basis_after_forward, earth.penalty, earth.rel_GCV
        )
        earth.coeff = _coeff_nd(earth.x, earth.y, earth.basis)
        nothing
    end
end
