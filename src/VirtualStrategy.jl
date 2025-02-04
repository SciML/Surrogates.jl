# Minimum Constant Liar 
function calculate_liars(::MinimumConstantLiar,
        tmp_surr::AbstractSurrogate,
        surr::AbstractSurrogate,
        new_x)
    new_y = minimum(surr.y)
    update!(tmp_surr, new_x, new_y)
end

# Maximum Constant Liar
function calculate_liars(::MaximumConstantLiar,
        tmp_surr::AbstractSurrogate,
        surr::AbstractSurrogate,
        new_x)
    new_y = maximum(surr.y)
    update!(tmp_surr, new_x, new_y)
end

# Mean Constant Liar
function calculate_liars(::MeanConstantLiar,
        tmp_surr::AbstractSurrogate,
        surr::AbstractSurrogate,
        new_x)
    new_y = mean(surr.y)
    update!(tmp_surr, new_x, new_y)
end

# Kriging Believer
function calculate_liars(::KrigingBeliever, tmp_k::Kriging, k::Kriging, new_x)
    new_y = k(new_x)
    update!(tmp_k, new_x, new_y)
end

# Kriging Believer Upper Bound
function calculate_liars(::KrigingBelieverUpperBound, tmp_k::Kriging, k::Kriging, new_x)
    new_y = k(new_x) + 3 * std_error_at_point(k, new_x)
    update!(tmp_k, new_x, new_y)
end

# Kriging Believer Lower Bound
function calculate_liars(::KrigingBelieverLowerBound, tmp_k::Kriging, k::Kriging, new_x)
    new_y = k(new_x) - 3 * std_error_at_point(k, new_x)
    update!(tmp_k, new_x, new_y)
end
