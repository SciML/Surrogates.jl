using LinearAlgebra

mutable struct EarthSurrogate{X, Y, L, U, B, C, P, M, N, R, G, I, T} <: AbstractSurrogate
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

function _coeff_1d(x, y, basis)
    n = length(x)
    d = length(basis)
    X = zeros(eltype(x[1]), n, d)
    @inbounds for i in 1:n
        for j in 1:d
            X[i, j] = basis[j](x[i])
        end
    end
    return (X' * X) \ (X' * y)
end

function _forward_pass_1d(x, y, n_max_terms, rel_res_error, maxiters)
    n = length(x)
    basis = Array{Function}(undef, 0)
    current_sse = +Inf
    intercept = sum([y[i] for i in 1:length(y)]) / length(y)
    num_terms = 0
    pos_of_knot = 0
    iters = 0
    while num_terms < n_max_terms || iters > maxiters
        #Look for best addition:
        new_addition = false
        for i in 1:length(x)
            #Add or not add the knot var_i?
            var_i = x[i]
            new_basis = copy(basis)
            #select best new pair
            hinge1 = x -> _hinge(x, var_i)
            hinge2 = x -> _hinge_mirror(x, var_i)
            push!(new_basis, hinge1)
            push!(new_basis, hinge2)
            #find coefficients
            d = length(new_basis)
            X = zeros(eltype(x[1]), n, d)
            @inbounds for i in 1:n
                for j in 1:d
                    X[i, j] = new_basis[j](x[i])
                end
            end
            if (cond(X' * X) > 1e8)
                condition_number = false
                new_sse = +Inf
            else
                condition_number = true
                coeff = (X' * X) \ (X' * y)
                new_sse = zero(y[1])
                d = length(new_basis)
                for i in 1:n
                    val_i = sum(coeff[j] * new_basis[j](x[i]) for j in 1:d) + intercept
                    new_sse = new_sse + (y[i] - val_i)^2
                end
            end
            #is the i-esim the best?
            if ((new_sse < current_sse) && (abs(current_sse - new_sse) >= rel_res_error) &&
                condition_number)
                #Add the hinge function to the basis
                pos_of_knot = i
                current_sse = new_sse
                new_addition = true
            end
        end
        iters = iters + 1
        if new_addition
            best_hinge1 = z -> _hinge(z, x[pos_of_knot])
            best_hinge2 = z -> _hinge_mirror(z, x[pos_of_knot])
            push!(basis, best_hinge1)
            push!(basis, best_hinge2)
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
    intercept = sum([y[i] for i in 1:length(y)]) / length(y)
    coeff = _coeff_1d(x, y, basis)
    sse = zero(y[1])
    for i in 1:n
        val_i = sum(coeff[j] * basis[j](x[i]) for j in 1:d) + intercept
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
        for i in 1:n
            current_basis = copy(basis)
            #remove i-esim element from current basis
            deleteat!(current_basis, i)
            coef = _coeff_1d(x, y, current_basis)
            new_sse = zero(y[i])
            for a in 1:n
                val_a = sum(coeff[j] * basis[j](x[a]) for j in 1:num_terms) + intercept
                new_sse = new_sse + (y[a] - val_a)^2
            end
            effect_num_params = num_terms + penalty * (num_terms - 1) / 2
            i_gcv = new_sse / (n * (1 - effect_num_params / n)^2)
            if i_gcv < current_gcv
                basis_to_remove = i
                new_gcv = i_gcv
                found_new_to_eliminate = true
            end
        end
        if !found_new_to_eliminate
            break
        end
        if abs(current_gcv - new_gcv) < rel_GCV
            break
        else
            num_terms = num_terms - 1
            deleteat!(basis, basis_to_remove)
        end
    end
    return basis
end

function EarthSurrogate(x, y, lb::Number, ub::Number; penalty::Number = 2.0,
        n_min_terms::Int = 2, n_max_terms::Int = 10,
        rel_res_error::Number = 1e-2, rel_GCV::Number = 1e-2,
        maxiters = 100)
    intercept = sum([y[i] for i in 1:length(y)]) / length(y)
    basis_after_forward = _forward_pass_1d(x, y, n_max_terms, rel_res_error, maxiters)
    basis = _backward_pass_1d(x, y, n_min_terms, basis_after_forward, penalty, rel_GCV)
    coeff = _coeff_1d(x, y, basis)
    return EarthSurrogate(x, y, lb, ub, basis, coeff, penalty, n_min_terms, n_max_terms,
        rel_res_error, rel_GCV, intercept, maxiters)
end

function (earth::EarthSurrogate)(val::Number)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(earth, val)
    return sum([earth.coeff[i] * earth.basis[i](val) for i in 1:length(earth.coeff)]) +
           earth.intercept
end

#ND
#inside arr_hing I have functions like g(x) = x -> _hinge(x,5.0) or g(x) = one(x)
#_product_hinge(val,arr_hing) = prod([arr_hing[i](val[i]) for i = 1:length(val)])

function _coeff_nd(x, y, basis)
    n = length(x)
    base_len = length(basis)
    d = length(x[1])
    X = zeros(eltype(x[1]), n, base_len)
    @inbounds for a in 1:n
        for b in 1:base_len
            X[a, b] = prod([basis[b][c](x[a][c]) for c in 1:d])
        end
    end
    return (X' * X) \ (X' * y)
end

function _forward_pass_nd(x, y, n_max_terms, rel_res_error, maxiters)
    n = length(x)
    basis = Array{Array{Function, 1}}(undef, 0) #ex of basis: push!(x,[x->1.0,x->1.0,x->_hinge(x,knot)]) so basis is of the form arr_hing and then I can pass my val
    current_sse = +Inf
    const_1 = x -> 1.0
    intercept = sum([y[i] for i in 1:length(y)]) / length(y)
    num_terms = 0
    d = length(x[1])
    best_hinge1 = best_hinge2 = [x -> one(eltype(x[1])) for j in 1:d]
    new_addition = false
    iters = 0
    while num_terms <= n_max_terms || iters > maxiters
        current_basis = copy(basis)
        new_addition = false
        for i in 1:n
            for j in 1:d
                for k in 1:n
                    for l in 1:d
                        new_basis = copy(basis)
                        #model with interaction between x[i][j] and x[k][l]
                        new_hinge1 = [w != j ? x -> one(eltype(x[1])) :
                                      z -> _hinge(z, x[i][j]) for w in 1:d]
                        new_hinge2 = [w != l ? x -> one(eltype(x[1])) :
                                      z -> _hinge_mirror(z, x[k][l]) for w in 1:d]
                        push!(new_basis, new_hinge1)
                        push!(new_basis, new_hinge2)

                        #build the model, find sse and check if it's better than current best, if so save the params
                        bas_len = length(new_basis)
                        X = zeros(eltype(x[1]), n, bas_len)
                        @inbounds for a in 1:n
                            for b in 1:bas_len
                                X[a, b] = prod([new_basis[b][c](x[a][c]) for c in 1:d])
                            end
                        end
                        if (cond(X' * X) > 1e8)
                            condition_number = false
                            new_sse = +Inf
                        else
                            condition_number = true
                            coeff = (X' * X) \ (X' * y)
                            new_sse = zero(y[1])
                            for a in 1:n
                                val_a = sum(coeff[b] *
                                            prod([new_basis[b][c](x[a][c]) for c in 1:d])
                                            for b in 1:bas_len) + intercept
                                new_sse = new_sse + (y[a] - val_a)^2
                            end
                        end
                        #is the i-esim the best?
                        if ((new_sse < current_sse) &&
                            (abs(current_sse - new_sse) >= rel_res_error) &&
                            condition_number)
                            #Add the hinge function to the basis
                            best_hinge1 = new_hinge1
                            best_hinge2 = new_hinge2
                            current_sse = new_sse
                            new_addition = true
                        end
                    end
                end
            end
        end
        iters = iters + 1
        if new_addition
            push!(basis, best_hinge1)
            push!(basis, best_hinge2)
            num_terms = num_terms + 1
        else
            break
        end
    end
    if (length(basis) == 0)
        throw("Earth surrogate did not add any term, just the intercept. It is advised to double check the parameters.")
    end
    return basis
end

function _backward_pass_nd(x, y, n_min_terms, basis, penalty, rel_GCV)
    n = length(x)
    d = length(x[1])
    base_len = length(basis)
    intercept = sum([y[i] for i in 1:length(y)]) / length(y)
    coeff = _coeff_nd(x, y, basis)
    sse = zero(y[1])
    for a in 1:n
        val_a = sum(coeff[b] * prod([basis[b][c](x[a][c]) for c in 1:d])
                    for b in 1:base_len) +
                intercept
        sse = sse + (y[a] - val_a)^2
    end
    effect_num_params = base_len + penalty * (base_len - 1) / 2
    current_gcv = sse / (n * (1 - effect_num_params / n)^2)
    num_terms = base_len
    new_gcv = +Inf
    basis_to_remove = 0
    while (num_terms > n_min_terms)
        #Basis-> select worst performing element-> eliminate it
        if num_terms <= 1
            break
        end
        found_new_to_eliminate = false
        for i in 1:num_terms
            current_basis = copy(basis)
            #remove i-esim element from current basis
            deleteat!(current_basis, i)
            coef = _coeff_nd(x, y, current_basis)
            new_sse = zero(y[i])
            current_base_len = num_terms - 1
            for a in 1:n
                val_a = sum(coeff[b] * prod([current_basis[b][c](x[a][c]) for c in 1:d])
                            for b in 1:current_base_len) + intercept
                new_sse = new_sse + (y[a] - val_a)^2
            end
            curr_effect_num_params = current_base_len + penalty * (current_base_len - 1) / 2
            i_gcv = new_sse / (n * (1 - curr_effect_num_params / n)^2)
            if i_gcv < current_gcv
                basis_to_remove = i
                new_gcv = i_gcv
                found_new_to_eliminate = true
            end
        end
        if !found_new_to_eliminate
            break
        elseif abs(current_gcv - new_gcv) < rel_GCV
            break
        else
            num_terms = num_terms - 1
            deleteat!(basis, basis_to_remove)
        end
    end
    return basis
end

function EarthSurrogate(x, y, lb, ub; penalty::Number = 2.0, n_min_terms::Int = 2,
        n_max_terms::Int = 10, rel_res_error::Number = 1e-2,
        rel_GCV::Number = 1e-2, maxiters = 100)
    intercept = sum([y[i] for i in 1:length(y)]) / length(y)
    basis_after_forward = _forward_pass_nd(x, y, n_max_terms, rel_res_error, maxiters)
    basis = _backward_pass_nd(x, y, n_min_terms, basis_after_forward, penalty, rel_GCV)
    coeff = _coeff_nd(x, y, basis)
    return EarthSurrogate(x, y, lb, ub, basis, coeff, penalty, n_min_terms, n_max_terms,
        rel_res_error, rel_GCV, intercept, maxiters)
end

function (earth::EarthSurrogate)(val)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(earth, val)
    return sum([earth.coeff[i] * prod([earth.basis[i][j](val[j]) for j in 1:length(val)])
                for i in 1:length(earth.coeff)]) + earth.intercept
end

function add_point!(earth::EarthSurrogate, x_new, y_new)
    if length(earth.x[1]) == 1
        #1D
        earth.x = vcat(earth.x, x_new)
        earth.y = vcat(earth.y, y_new)
        earth.intercept = sum([earth.y[i] for i in 1:length(earth.y)]) / length(earth.y)
        basis_after_forward = _forward_pass_1d(earth.x, earth.y, earth.n_max_terms,
            earth.rel_res_error, earth.maxiters)
        earth.basis = _backward_pass_1d(earth.x, earth.y, earth.n_min_terms,
            basis_after_forward, earth.penalty, earth.rel_GCV)
        earth.coeff = _coeff_1d(earth.x, earth.y, earth.basis)
        nothing
    else
        #ND
        earth.x = vcat(earth.x, x_new)
        earth.y = vcat(earth.y, y_new)
        earth.intercept = sum([earth.y[i] for i in 1:length(earth.y)]) / length(earth.y)
        basis_after_forward = _forward_pass_nd(earth.x, earth.y, earth.n_max_terms,
            earth.rel_res_error, earth.maxiters)
        earth.basis = _backward_pass_nd(earth.x, earth.y, earth.n_min_terms,
            basis_after_forward, earth.penalty, earth.rel_GCV)
        earth.coeff = _coeff_nd(earth.x, earth.y, earth.basis)
        nothing
    end
end
