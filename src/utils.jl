remove_tracker(x) = x

_match_container(y, y_el::Number) = first(y)
_match_container(y, y_el) = y

function _check_dimension(surr, input)
    expected_dim = length(surr.x[1])
    input_dim = length(input)

    if input_dim != expected_dim
        throw(
            ArgumentError("This surrogate expects $expected_dim-dimensional inputs, but the input had dimension $input_dim.")
        )
    end

    return nothing
end
