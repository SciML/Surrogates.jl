# Gradient-enhanced surrogates require a gradient alongside every new response,
# so a virtual point has to carry one as well.
_requires_gradient(::AbstractSurrogate) = false
_requires_gradient(::GEK) = true
_requires_gradient(::GEKPLS) = true

# Place a virtual observation on the temporary surrogate. Where a gradient is
# required, the model's own slope at that point is used: the virtual point then
# misstates only the response, and asserts nothing about the slope that the
# model does not already believe.
function _virtual_update!(tmp_surr::AbstractSurrogate, new_x, new_y)
    _requires_gradient(tmp_surr) || return update!(tmp_surr, new_x, new_y)
    return update!(
        tmp_surr, new_x, new_y, only(Zygote.gradient(tmp_surr, new_x))
    )
end

# The constant liars take their virtual value from the observed responses. They
# go through `_sample_responses` because `GEK` stores `[values; gradients]` in
# `y`: reducing over the whole vector would hand the surrogate a directional
# derivative as though it were an objective value.

# Minimum Constant Liar
function calculate_liars(
        ::MinimumConstantLiar,
        tmp_surr::AbstractSurrogate,
        surr::AbstractSurrogate,
        new_x
    )
    new_y = minimum(_sample_responses(surr))
    return _virtual_update!(tmp_surr, new_x, new_y)
end

# Maximum Constant Liar
function calculate_liars(
        ::MaximumConstantLiar,
        tmp_surr::AbstractSurrogate,
        surr::AbstractSurrogate,
        new_x
    )
    new_y = maximum(_sample_responses(surr))
    return _virtual_update!(tmp_surr, new_x, new_y)
end

# Mean Constant Liar
function calculate_liars(
        ::MeanConstantLiar,
        tmp_surr::AbstractSurrogate,
        surr::AbstractSurrogate,
        new_x
    )
    new_y = mean(_sample_responses(surr))
    return _virtual_update!(tmp_surr, new_x, new_y)
end

# The believer strategies read their virtual value off `tmp_k`, the model that
# already carries the beliefs placed at the batch's earlier points. Reading it
# off `k` instead would make every belief in a batch independent of the ones
# before it, which is the opposite of what Ginsbourger, Le Riche and Carraro
# (2010) describe: the metamodel is updated between selections precisely so the
# next belief accounts for them.

# The three believers need a predictive standard deviation, so they accept any
# surrogate that models one rather than `Kriging` alone: `GEK` and
# `AbstractGPSurrogate` answer `std_error_at_point` just as well.

# Kriging Believer
function calculate_liars(
        ::KrigingBeliever, tmp_k::AbstractSurrogate, k::AbstractSurrogate, new_x
    )
    new_y = tmp_k(new_x)
    return _virtual_update!(tmp_k, new_x, new_y)
end

# Kriging Believer Upper Bound
function calculate_liars(
        ::KrigingBelieverUpperBound, tmp_k::AbstractSurrogate,
        k::AbstractSurrogate, new_x
    )
    _require_std_error(tmp_k, "KrigingBelieverUpperBound")
    new_y = tmp_k(new_x) + 3 * std_error_at_point(tmp_k, new_x)
    return _virtual_update!(tmp_k, new_x, new_y)
end

# Kriging Believer Lower Bound
function calculate_liars(
        ::KrigingBelieverLowerBound, tmp_k::AbstractSurrogate,
        k::AbstractSurrogate, new_x
    )
    _require_std_error(tmp_k, "KrigingBelieverLowerBound")
    new_y = tmp_k(new_x) - 3 * std_error_at_point(tmp_k, new_x)
    return _virtual_update!(tmp_k, new_x, new_y)
end
