# Minimum Constant Liar
function calculate_liars(
        ::MinimumConstantLiar,
        tmp_surr::AbstractSurrogate,
        surr::AbstractSurrogate,
        new_x
    )
    new_y = minimum(surr.y)
    return update!(tmp_surr, new_x, new_y)
end

# Maximum Constant Liar
function calculate_liars(
        ::MaximumConstantLiar,
        tmp_surr::AbstractSurrogate,
        surr::AbstractSurrogate,
        new_x
    )
    new_y = maximum(surr.y)
    return update!(tmp_surr, new_x, new_y)
end

# Mean Constant Liar
function calculate_liars(
        ::MeanConstantLiar,
        tmp_surr::AbstractSurrogate,
        surr::AbstractSurrogate,
        new_x
    )
    new_y = mean(surr.y)
    return update!(tmp_surr, new_x, new_y)
end

# The believer strategies read their virtual value off `tmp_k`, the model that
# already carries the beliefs placed at the batch's earlier points. Reading it
# off `k` instead would make every belief in a batch independent of the ones
# before it, which is the opposite of what Ginsbourger, Le Riche and Carraro
# (2010) describe: the metamodel is updated between selections precisely so the
# next belief accounts for them.

# Kriging Believer
function calculate_liars(::KrigingBeliever, tmp_k::Kriging, k::Kriging, new_x)
    new_y = tmp_k(new_x)
    return update!(tmp_k, new_x, new_y)
end

# Kriging Believer Upper Bound
function calculate_liars(::KrigingBelieverUpperBound, tmp_k::Kriging, k::Kriging, new_x)
    new_y = tmp_k(new_x) + 3 * std_error_at_point(tmp_k, new_x)
    return update!(tmp_k, new_x, new_y)
end

# Kriging Believer Lower Bound
function calculate_liars(::KrigingBelieverLowerBound, tmp_k::Kriging, k::Kriging, new_x)
    new_y = tmp_k(new_x) - 3 * std_error_at_point(tmp_k, new_x)
    return update!(tmp_k, new_x, new_y)
end
