# Minimum Constant Liar 
function CLmin!(tmp_k::Kriging, k::Kriging, new_x)
    new_y = minimum(k.y)
    add_point!(tmp_k, new_x, new_y)
end

# Maximum Constant Liar
function CLmax!(tmp_k::Kriging, k::Kriging, new_x)
    new_y = maximum(k.y)
    add_point!(tmp_k, new_x, new_y)
end

# Mean Constant Liar
function CLmean!(tmp_k::Kriging, k::Kriging, new_x)
    new_y = mean(k.y)
    add_point!(tmp_k, new_x, new_y)
end

# Kriging Believer
function KB!(tmp_k::Kriging, k::Kriging, new_x)
    new_y = k(new_x)
    add_point!(tmp_k, new_x, new_y)
end

# Kriging Believer Upper Bound
function KBUB!(tmp_k::Kriging, k::Kriging, new_x)
    new_y = k(new_x) + 3 * std_error_at_point(k, new_x)
    add_point!(tmp_k, new_x, new_y)
end

# Kriging Believer Lower Bound
function KBLB!(tmp_k::Kriging, k::Kriging, new_x)
    new_y = k(new_x) - 3 * std_error_at_point(k, new_x)
    add_point!(tmp_k, new_x, new_y)
end