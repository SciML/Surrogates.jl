using LinearAlgebra
using Distributions
using Zygote

abstract type SurrogateOptimizationAlgorithm end

#single objective optimization
struct SRBF <: SurrogateOptimizationAlgorithm end
struct LCBS <: SurrogateOptimizationAlgorithm end
struct EI <: SurrogateOptimizationAlgorithm end
struct DYCORS <: SurrogateOptimizationAlgorithm end
struct SOP{P} <: SurrogateOptimizationAlgorithm
    p::P
end

#multi objective optimization
struct SMB <: SurrogateOptimizationAlgorithm end
struct RTEA{K, Z, P, N, S} <: SurrogateOptimizationAlgorithm
    k::K
    z::Z
    p::P
    n_c::N
    sigma::S
end

function merit_function(point, w, surr::AbstractSurrogate, s_max, s_min, d_max, d_min,
                        box_size)
    if length(point) == 1
        D_x = box_size + 1
        for i in 1:length(surr.x)
            distance = norm(surr.x[i] - point)
            if distance < D_x
                D_x = distance
            end
        end
        return w * (surr(point) - s_min) / (s_max - s_min) +
               (1 - w) * ((d_max - D_x) / (d_max - d_min))
    else
        D_x = norm(box_size) + 1
        for i in 1:length(surr.x)
            distance = norm(surr.x[i] .- point)
            if distance < D_x
                D_x = distance
            end
        end
        return w * (surr(point) - s_min) / (s_max - s_min) +
               (1 - w) * ((d_max - D_x) / (d_max - d_min))
    end
end

"""
The main idea is to pick the new evaluations from a set of candidate points where each candidate point is generated as an N(0, sigma^2)
distributed perturbation from the current best solution.
The value of sigma is modified based on progress and follows the same logic as
in many trust region methods: we increase sigma if we make a lot of progress
(the surrogate is accurate) and decrease sigma when we aren’t able to make progress
(the surrogate model is inaccurate).
More details about how sigma is updated is given in the original papers.

After generating the candidate points, we predict their objective function value
and compute the minimum distance to the previously evaluated point.
Let the candidate points be denoted by C and let the function value predictions
be s(x\\_i) and the distance values be d(x\\_i), both rescaled through a
linear transformation to the interval [0,1]. This is done to put the values on
the same scale.
The next point selected for evaluation is the candidate point x that minimizes
the weighted-distance merit function:

``merit(x) = ws(x) + (1-w)(1-d(x))``

where `` 0 \\leq w \\leq 1 ``.
That is, we want a small function value prediction and a large minimum distance
from the previously evaluated points.
The weight w is commonly cycled between
a few values to achieve both exploitation and exploration.
When w is close to zero, we do pure exploration, while w close to 1 corresponds to exploitation.
"""
function surrogate_optimize(obj::Function, ::SRBF, lb, ub, surr::AbstractSurrogate,
                            sample_type::SamplingAlgorithm; maxiters = 100,
                            num_new_samples = 100, needs_gradient = false)
    scale = 0.2
    success = 0
    failure = 0
    w_range = [0.3, 0.5, 0.7, 0.95]

    #Vector containing size in each direction
    box_size = lb - ub
    success = 0
    failures = 0
    dtol = 1e-3 * norm(ub - lb)
    d = length(surr.x)
    num_of_iterations = 0
    for w in Iterators.cycle(w_range)
        num_of_iterations += 1
        if num_of_iterations == maxiters
            index = argmin(surr.y)
            return (surr.x[index], surr.y[index])
        end
        for k in 1:maxiters
            incumbent_value = minimum(surr.y)
            incumbent_x = surr.x[argmin(surr.y)]

            new_lb = incumbent_x .- 3 * scale * norm(incumbent_x .- lb)
            new_ub = incumbent_x .+ 3 * scale * norm(incumbent_x .- ub)

            @inbounds for i in 1:length(new_lb)
                if new_lb[i] < lb[i]
                    new_lb = collect(new_lb)
                    new_lb[i] = lb[i]
                end
                if new_ub[i] > ub[i]
                    new_ub = collect(new_ub)
                    new_ub[i] = ub[i]
                end
            end

            new_sample = sample(num_new_samples, new_lb, new_ub, sample_type)
            s = zeros(eltype(surr.x[1]), num_new_samples)
            for j in 1:num_new_samples
                s[j] = surr(new_sample[j])
            end
            s_max = maximum(s)
            s_min = minimum(s)

            d_min = norm(box_size .+ 1)
            d_max = 0.0
            for r in 1:length(surr.x)
                for c in 1:num_new_samples
                    distance_rc = norm(surr.x[r] .- new_sample[c])
                    if distance_rc > d_max
                        d_max = distance_rc
                    end
                    if distance_rc < d_min
                        d_min = distance_rc
                    end
                end
            end

            #3)Evaluate merit function in the sampled points

            evaluation_of_merit_function = zeros(float(eltype(surr.x[1])), num_new_samples)
            @inbounds for r in 1:num_new_samples
                evaluation_of_merit_function[r] = merit_function(new_sample[r], w, surr,
                                                                 s_max, s_min, d_max, d_min,
                                                                 box_size)
            end
            new_addition = false
            adaptive_point_x = Tuple{}
            diff_x = zeros(eltype(surr.x[1]), d)
            while new_addition == false
                #find minimum
                new_min_y = minimum(evaluation_of_merit_function)
                min_index = argmin(evaluation_of_merit_function)
                new_min_x = new_sample[min_index]
                for l in 1:d
                    diff_x[l] = norm(surr.x[l] .- new_min_x)
                end
                bit_x = diff_x .> dtol
                #new_min_x has to have some distance from krig.x
                if false in bit_x
                    #The new_point is not actually that new, discard it!

                    deleteat!(evaluation_of_merit_function, min_index[1])
                    deleteat!(new_sample, min_index)

                    if length(new_sample) == 0
                        println("Out of sampling points")
                        index = argmin(surr.y)
                        return (surr.x[index], surr.y[index])
                    end
                else
                    new_addition = true
                    adaptive_point_x = Tuple(new_min_x)
                end
            end

            #4) Evaluate objective function at adaptive point
            adaptive_point_y = obj(adaptive_point_x)

            #5) Update surrogate with (adaptive_point,objective(adaptive_point)
            if (needs_gradient)
                adaptive_grad = Zygote.gradient(obj, adaptive_point_x)
                add_point!(surr, adaptive_point_x, adaptive_point_y, adaptive_grad)
            else
                add_point!(surr, adaptive_point_x, adaptive_point_y)
            end

            #6) How to go on?
            if surr(adaptive_point_x) < incumbent_value
                #success
                incumbent_x = adaptive_point_x
                incumbent_value = adaptive_point_y
                if failure == 0
                    success += 1
                else
                    failure = 0
                    success += 1
                end
            else
                #failure
                if success == 0
                    failure += 1
                else
                    success = 0
                    failure += 1
                end
            end

            if success == 3
                scale = scale * 2
                if scale > 0.8 * norm(ub - lb)
                    println("Exiting, scale too big")
                    index = argmin(surr.y)
                    return (surr.x[index], surr.y[index])
                end
                success = 0
                failure = 0
            end

            if failure == 5
                scale = scale / 2
                #check bounds and go on only if > 1e-5*interval
                if scale < 1e-5
                    println("Exiting, too narrow")
                    index = argmin(surr.y)
                    return (surr.x[index], surr.y[index])
                end
                success = 0
                failure = 0
            end
        end
    end
end

"""
SRBF 1D:
surrogate_optimize(obj::Function,::SRBF,lb::Number,ub::Number,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
"""
function surrogate_optimize(obj::Function, ::SRBF, lb::Number, ub::Number,
                            surr::AbstractSurrogate, sample_type::SamplingAlgorithm;
                            maxiters = 100, num_new_samples = 100)
    #Suggested by:
    #https://www.mathworks.com/help/gads/surrogate-optimization-algorithm.html
    scale = 0.2
    success = 0
    failure = 0
    w_range = [0.3, 0.5, 0.7, 0.95]
    box_size = lb - ub
    success = 0
    failures = 0
    dtol = 1e-3 * norm(ub - lb)
    num_of_iterations = 0
    for w in Iterators.cycle(w_range)
        num_of_iterations += 1
        if num_of_iterations == maxiters
            index = argmin(surr.y)
            return (surr.x[index], surr.y[index])
        end
        for k in 1:maxiters
            #1) Sample near incumbent (the 2 fraction is arbitrary here)
            incumbent_value = minimum(surr.y)
            incumbent_x = surr.x[argmin(surr.y)]

            new_lb = incumbent_x - scale * norm(incumbent_x - lb)
            new_ub = incumbent_x + scale * norm(incumbent_x - ub)
            if new_lb < lb
                new_lb = lb
            end
            if new_ub > ub
                new_ub = ub
            end
            new_sample = sample(num_new_samples, new_lb, new_ub, sample_type)

            #2) Create  merit function
            s = zeros(eltype(surr.x[1]), num_new_samples)
            for j in 1:num_new_samples
                s[j] = surr(new_sample[j])
            end
            s_max = maximum(s)
            s_min = minimum(s)

            d_min = box_size + 1
            d_max = 0.0
            for r in 1:length(surr.x)
                for c in 1:num_new_samples
                    distance_rc = norm(surr.x[r] - new_sample[c])
                    if distance_rc > d_max
                        d_max = distance_rc
                    end
                    if distance_rc < d_min
                        d_min = distance_rc
                    end
                end
            end
            #3) Evaluate merit function at the sampled points
            evaluation_of_merit_function = merit_function.(new_sample, w, surr, s_max,
                                                           s_min, d_max, d_min, box_size)

            new_addition = false
            adaptive_point_x = zero(eltype(new_sample[1]))
            while new_addition == false
                #find minimum
                new_min_y = minimum(evaluation_of_merit_function)
                min_index = argmin(evaluation_of_merit_function)
                new_min_x = new_sample[min_index]

                diff_x = abs.(surr.x .- new_min_x)
                bit_x = diff_x .> dtol
                #new_min_x has to have some distance from krig.x
                if false in bit_x
                    #The new_point is not actually that new, discard it!
                    deleteat!(evaluation_of_merit_function, min_index)
                    deleteat!(new_sample, min_index)
                    if length(new_sample) == 0
                        println("Out of sampling points")
                        index = argmin(surr.y)
                        return (surr.x[index], surr.y[index])
                    end
                else
                    new_addition = true
                    adaptive_point_x = new_min_x
                end
            end
            #4) Evaluate objective function at adaptive point
            adaptive_point_y = obj(adaptive_point_x)

            #5) Update surrogate with (adaptive_point,objective(adaptive_point)
            add_point!(surr, adaptive_point_x, adaptive_point_y)

            #6) How to go on?
            if surr(adaptive_point_x) < incumbent_value
                #success
                incumbent_x = adaptive_point_x
                incumbent_value = adaptive_point_y
                if failure == 0
                    success += 1
                else
                    failure = 0
                    success += 1
                end
            else
                #failure
                if success == 0
                    failure += 1
                else
                    success = 0
                    failure += 1
                end
            end

            if success == 3
                scale = scale * 2
                #check bounds cannot go more than [a,b]
                if scale > 0.8 * norm(ub - lb)
                    println("Exiting, scale too big")
                    index = argmin(surr.y)
                    return (surr.x[index], surr.y[index])
                end
                success = 0
                failure = 0
            end

            if failure == 5
                scale = scale / 2
                #check bounds and go on only if > 1e-5*interval
                if scale < 1e-5
                    println("Exiting, too narrow")
                    index = argmin(surr.y)
                    return (surr.x[index], surr.y[index])
                end
                success = 0
                failure = 0
            end
        end
    end
end

# Ask SRBF ND
function Ask(::SRBF, lb, ub, surr::AbstractSurrogate, sample_type::SamplingAlgorithm, n_parallel, strategy!;
    num_new_samples = 500)

    scale = 0.2
    success = 0
    failure = 0
    w_range = [0.3, 0.5, 0.7, 0.95]
    w_cycle = Iterators.cycle(w_range)

    w, state = iterate(w_cycle)

    #Vector containing size in each direction
    box_size = lb - ub
    success = 0
    failures = 0
    dtol = 1e-3 * norm(ub - lb)
    d = length(surr.x)
    incumbent_x = surr.x[argmin(surr.y)]

    new_lb = incumbent_x .- 3 * scale * norm(incumbent_x .- lb)
    new_ub = incumbent_x .+ 3 * scale * norm(incumbent_x .- ub)

    @inbounds for i in 1:length(new_lb)
        if new_lb[i] < lb[i]
            new_lb = collect(new_lb)
            new_lb[i] = lb[i]
        end
        if new_ub[i] > ub[i]
            new_ub = collect(new_ub)
            new_ub[i] = ub[i]
        end
    end

    new_sample = sample(num_new_samples, new_lb, new_ub, sample_type)
    s = zeros(eltype(surr.x[1]), num_new_samples)
    for j in 1:num_new_samples
        s[j] = surr(new_sample[j])
    end
    s_max = maximum(s)
    s_min = minimum(s)

    d_min = norm(box_size .+ 1)
    d_max = 0.0
    for r in 1:length(surr.x)
        for c in 1:num_new_samples
            distance_rc = norm(surr.x[r] .- new_sample[c])
            if distance_rc > d_max
                d_max = distance_rc
            end
            if distance_rc < d_min
                d_min = distance_rc
            end
        end
    end

    tmp_surr = deepcopy(surr)

    
    new_addition = 0
    diff_x = zeros(eltype(surr.x[1]), d)

    evaluation_of_merit_function = zeros(float(eltype(surr.x[1])), num_new_samples)
    proposed_points_x = Vector{typeof(surr.x[1])}(undef, n_parallel)
    merit_of_proposed_points = zeros(Float64, n_parallel)

    while new_addition < n_parallel
        #find minimum

        @inbounds for r in 1:num_new_samples
            evaluation_of_merit_function[r] = merit_function(new_sample[r], w, tmp_surr,
                s_max, s_min, d_max, d_min,
                box_size)
        end

        min_index = argmin(evaluation_of_merit_function)
        new_min_x = new_sample[min_index]
        min_x_merit = evaluation_of_merit_function[min_index]

        for l in 1:d
            diff_x[l] = norm(surr.x[l] .- new_min_x)
        end
        bit_x = diff_x .> dtol
        #new_min_x has to have some distance from krig.x
        if false in bit_x
            #The new_point is not actually that new, discard it!

            deleteat!(evaluation_of_merit_function, min_index)
            deleteat!(new_sample, min_index)

            if length(new_sample) == 0
                println("Out of sampling points")
                index = argmin(surr.y)
                return (surr.x[index], surr.y[index])
            end
        else
            new_addition += 1
            proposed_points_x[new_addition] = new_min_x
            merit_of_proposed_points[new_addition] = min_x_merit

            # Update temporary surrogate using provided strategy
            strategy!(tmp_surr, surr, new_min_x)
        end

        #4) Update w
        w, state = iterate(w_cycle, state)
    end

    return (proposed_points_x, merit_of_proposed_points)
end

# Ask SRBF 1D
function Ask(::SRBF, lb::Number, ub::Number, surr::AbstractSurrogate,
    sample_type::SamplingAlgorithm, n_parallel, strategy!;
    num_new_samples = 500)
    scale = 0.2
    success = 0
    w_range = [0.3, 0.5, 0.7, 0.95]
    w_cycle = Iterators.cycle(w_range)

    w, state = iterate(w_cycle)

    box_size = lb - ub
    success = 0
    failures = 0
    dtol = 1e-3 * norm(ub - lb)
    num_of_iterations = 0

    #1) Sample near incumbent (the 2 fraction is arbitrary here)
    incumbent_x = surr.x[argmin(surr.y)]

    new_lb = incumbent_x - scale * norm(incumbent_x - lb)
    new_ub = incumbent_x + scale * norm(incumbent_x - ub)
    if new_lb < lb
        new_lb = lb
    end
    if new_ub > ub
        new_ub = ub
    end

    new_sample = sample(num_new_samples, new_lb, new_ub, sample_type)

    #2) Create  merit function
    s = zeros(eltype(surr.x[1]), num_new_samples)
    for j in 1:num_new_samples
        s[j] = surr(new_sample[j])
    end
    s_max = maximum(s)
    s_min = minimum(s)

    d_min = box_size + 1
    d_max = 0.0
    for r in 1:length(surr.x)
        for c in 1:num_new_samples
            distance_rc = norm(surr.x[r] - new_sample[c])
            if distance_rc > d_max
                d_max = distance_rc
            end
            if distance_rc < d_min
                d_min = distance_rc
            end
        end
    end

    new_addition = 0
    proposed_points_x = zeros(eltype(new_sample[1]), n_parallel)
    merit_of_proposed_points = zeros(eltype(new_sample[1]), n_parallel)

    # Temporary surrogate for virtual points
    tmp_surr = deepcopy(surr)

    # Loop until we have n_parallel new points
    while new_addition < n_parallel

        #3) Evaluate merit function at the sampled points in parallel 
        evaluation_of_merit_function = merit_function.(new_sample, w, tmp_surr, s_max,
            s_min, d_max, d_min, box_size)

        #find minimum
        min_index = argmin(evaluation_of_merit_function)
        new_min_x = new_sample[min_index]
        min_x_merit = evaluation_of_merit_function[min_index]

        diff_x = abs.(tmp_surr.x .- new_min_x)
        bit_x = diff_x .> dtol
        #new_min_x has to have some distance from krig.x
        if false in bit_x
            #The new_point is not actually that new, discard it!
            deleteat!(evaluation_of_merit_function, min_index)
            deleteat!(new_sample, min_index)
            if length(new_sample) == 0
                println("Out of sampling points")
                return (proposed_points_x[1:new_addition],
                    merit_of_proposed_points[1:new_addition])
            end
        else
            new_addition += 1
            proposed_points_x[new_addition] = new_min_x
            merit_of_proposed_points[new_addition] = min_x_merit

            # Update temporary surrogate using provided strategy
            strategy!(tmp_surr, surr, new_min_x)
        end

        #4) Update w
        w, state = iterate(w_cycle, state)
    end

    return (proposed_points_x, merit_of_proposed_points)
end


"""
This is an implementation of Lower Confidence Bound (LCB),
a popular acquisition function in Bayesian optimization.
Under a Gaussian process (GP) prior, the goal is to minimize:
``LCB(x) := E[x] - k * \\sqrt{(V[x])}``
default value ``k = 2``.
"""
function surrogate_optimize(obj::Function, ::LCBS, lb::Number, ub::Number, krig,
                            sample_type::SamplingAlgorithm; maxiters = 100,
                            num_new_samples = 100, k = 2.0)
    dtol = 1e-3 * norm(ub - lb)
    for i in 1:maxiters
        new_sample = sample(num_new_samples, lb, ub, sample_type)
        evaluations = zeros(eltype(krig.x[1]), num_new_samples)
        for j in 1:num_new_samples
            evaluations[j] = krig(new_sample[j]) +
                             k * std_error_at_point(krig, new_sample[j])
        end

        new_addition = false
        min_add_x = zero(eltype(new_sample[1]))
        min_add_y = zero(eltype(krig.y[1]))
        while new_addition == false
            #find minimum
            new_min_y = minimum(evaluations)
            min_index = argmin(evaluations)
            new_min_x = new_sample[min_index]

            diff_x = abs.(krig.x .- new_min_x)
            bit_x = diff_x .> dtol
            #new_min_x has to have some distance from krig.x
            if false in bit_x
                #The new_point is not actually that new, discard it!
                deleteat!(evaluations, min_index)
                deleteat!(new_sample, min_index)

                if length(new_sample) == 0
                    println("Out of sampling points")
                    index = argmin(krig.y)
                    return (krig.x[index], krig.y[index])
                end
            else
                new_addition = true
                min_add_x = new_min_x
                min_add_y = new_min_y
            end
        end
        if min_add_y < 1e-6 * (maximum(krig.y) - minimum(krig.y))
            return
        else
            if (abs(min_add_y) == Inf || min_add_y == NaN)
                println("New point being added is +Inf or NaN, skipping.\n")
            else
                add_point!(krig, min_add_x, min_add_y)
            end
        end
    end
end

"""
This is an implementation of Lower Confidence Bound (LCB),
a popular acquisition function in Bayesian optimization.
Under a Gaussian process (GP) prior, the goal is to minimize:

``LCB(x) := E[x] - k * \\sqrt{(V[x])}``

default value ``k = 2``.
"""
function surrogate_optimize(obj::Function, ::LCBS, lb, ub, krig,
                            sample_type::SamplingAlgorithm; maxiters = 100,
                            num_new_samples = 100, k = 2.0)
    dtol = 1e-3 * norm(ub - lb)
    for i in 1:maxiters
        d = length(krig.x)
        new_sample = sample(num_new_samples, lb, ub, sample_type)
        evaluations = zeros(eltype(krig.x[1]), num_new_samples)
        for j in 1:num_new_samples
            evaluations[j] = krig(new_sample[j]) +
                             k * std_error_at_point(krig, new_sample[j])
        end

        new_addition = false
        min_add_x = Tuple{}
        min_add_y = zero(eltype(krig.y[1]))
        diff_x = zeros(eltype(krig.x[1]), d)
        while new_addition == false
            #find minimum
            new_min_y = minimum(evaluations)
            min_index = argmin(evaluations)
            new_min_x = new_sample[min_index]
            for l in 1:d
                diff_x[l] = norm(krig.x[l] .- new_min_x)
            end
            bit_x = diff_x .> dtol
            #new_min_x has to have some distance from krig.x
            if false in bit_x
                #The new_point is not actually that new, discard it!
                deleteat!(evaluations, min_index)
                deleteat!(new_sample, min_index)

                if length(new_sample) == 0
                    println("Out of sampling points")
                    index = argmin(krig.y)
                    return (krig.x[index], krig.y[index])
                end
            else
                new_addition = true
                min_add_x = new_min_x
                min_add_y = new_min_y
            end
        end
        if min_add_y < 1e-6 * (maximum(krig.y) - minimum(krig.y))
            index = argmin(krig.y)
            return (krig.x[index], krig.y[index])
        else
            min_add_y = obj(min_add_x) # I actually add the objc function at that point
            if (abs(min_add_y) == Inf || min_add_y == NaN)
                println("New point being added is +Inf or NaN, skipping.\n")
            else
                add_point!(krig, Tuple(min_add_x), min_add_y)
            end
        end
    end
end

"""
Expected improvement method 1D
"""
function surrogate_optimize(obj::Function, ::EI, lb::Number, ub::Number, krig,
                            sample_type::SamplingAlgorithm; maxiters = 100,
                            num_new_samples = 100)
    dtol = 1e-3 * norm(ub - lb)
    eps = 0.01
    for i in 1:maxiters
        # Sample lots of points from the design space -- we will evaluate the EI function at these points
        new_sample = sample(num_new_samples, lb, ub, sample_type)

        # Find the best point so far
        f_min = minimum(krig.y)

        # Allocate some arrays
        evaluations = zeros(eltype(krig.x[1]), num_new_samples)  # Holds EI function evaluations
        point_found = false                                     # Whether we have found a new point to test
        new_x_max = zero(eltype(krig.x[1]))                     # New x point
        new_EI_max = zero(eltype(krig.x[1]))                    # EI at new x point
        while point_found == false
            # For each point in the sample set, evaluate the Expected Improvement function
            for j in 1:length(new_sample)
                std = std_error_at_point(krig, new_sample[j])
                u = krig(new_sample[j])
                if abs(std) > 1e-6
                    z = (f_min - u - eps) / std
                else
                    z = 0
                end
                # Evaluate EI at point new_sample[j]
                evaluations[j] = (f_min - u - eps) * cdf(Normal(), z) +
                                 std * pdf(Normal(), z)
            end
            # find the sample which maximizes the EI function
            index_max = argmax(evaluations)
            x_new = new_sample[index_max]   # x point which maximized EI
            y_new = maximum(evaluations)    # EI at the new point
            diff_x = abs.(krig.x .- x_new)
            bit_x = diff_x .> dtol
            #new_min_x has to have some distance from krig.x
            if false in bit_x
                #The new_point is not actually that new, discard it!
                deleteat!(evaluations, index_max)
                deleteat!(new_sample, index_max)
                if length(new_sample) == 0
                    println("Out of sampling points")
                    index = argmin(krig.y)
                    return (krig.x[index], krig.y[index])
                end
            else
                point_found = true
                new_x_max = x_new
                new_EI_max = y_new
            end
        end
        # if the EI is less than some tolerance times the difference between the maximum and minimum points
        # in the surrogate, then we terminate the optimizer.
        if new_EI_max < 1e-6 * norm(maximum(krig.y) - minimum(krig.y))
            index = argmin(krig.y)
            println("Termination tolerance reached.")
            return (krig.x[index], krig.y[index])
        end
        # Otherwise, evaluate the true objective function at the new point and repeat.
        add_point!(krig, new_x_max, obj(new_x_max))
    end
    println("Completed maximum number of iterations")
end

# Ask EI 1D & ND
function Ask(::EI, lb, ub, krig, sample_type::SamplingAlgorithm, n_parallel::Number, strategy!;
             num_new_samples = 100)

    lb = krig.lb
    ub = krig.ub

    dtol = 1e-3 * norm(ub - lb)
    eps = 0.01

    tmp_krig = deepcopy(krig) # Temporary copy of the kriging model to store virtual points

    new_x_max = Vector{typeof(tmp_krig.x[1])}(undef, n_parallel)             # New x point
    new_EI_max = zeros(eltype(tmp_krig.x[1]), n_parallel)                    # EI at new x point

    for i in 1:n_parallel
        # Sample lots of points from the design space -- we will evaluate the EI function at these points
        new_sample = sample(num_new_samples, lb, ub, sample_type)

        # Find the best point so far
        f_min = minimum(tmp_krig.y)

        # Allocate some arrays
        evaluations = zeros(eltype(tmp_krig.x[1]), num_new_samples)  # Holds EI function evaluations
        point_found = false                                     # Whether we have found a new point to test
        while point_found == false
            # For each point in the sample set, evaluate the Expected Improvement function
            for j in eachindex(new_sample)
                std = std_error_at_point(tmp_krig, new_sample[j])
                u = tmp_krig(new_sample[j])
                if abs(std) > 1e-6
                    z = (f_min - u - eps) / std
                else
                    z = 0
                end
                # Evaluate EI at point new_sample[j]
                evaluations[j] = (f_min - u - eps) * cdf(Normal(), z) +
                                 std * pdf(Normal(), z)
            end
            # find the sample which maximizes the EI function
            index_max = argmax(evaluations)
            x_new = new_sample[index_max]   # x point which maximized EI
            y_new = maximum(evaluations)    # EI at the new point
            diff_x = [abs.(prev_point .- x_new) for prev_point in tmp_krig.x]
            bit_x = [diff_x_point .> dtol for diff_x_point in diff_x]
            #new_min_x has to have some distance from tmp_krig.x
            if false in bit_x
                #The new_point is not actually that new, discard it!
                deleteat!(evaluations, index_max)
                deleteat!(new_sample, index_max)
                if length(new_sample) == 0
                    println("Out of sampling points")
                    index = argmin(tmp_krig.y)
                    return (tmp_krig.x[index], tmp_krig.y[index])
                end
            else
                point_found = true
                new_x_max[i] = x_new
                new_EI_max[i] = y_new
                strategy!(tmp_krig, krig, x_new)
            end
        end
    end

    return (new_x_max, new_EI_max)
end

"""
This is an implementation of Expected Improvement (EI),
arguably the most popular acquisition function in Bayesian optimization.
Under a Gaussian process (GP) prior, the goal is to
maximize expected improvement:

``EI(x) := E[max(f_{best}-f(x),0)``


"""
function surrogate_optimize(obj::Function, ::EI, lb, ub, krig,
                            sample_type::SamplingAlgorithm; maxiters = 100,
                            num_new_samples = 100)
    dtol = 1e-3 * norm(ub - lb)
    eps = 0.01
    for i in 1:maxiters
        d = length(krig.x)
        # Sample lots of points from the design space -- we will evaluate the EI function at these points
        new_sample = sample(num_new_samples, lb, ub, sample_type)

        # Find the best point so far
        f_min = minimum(krig.y)

        # Allocate some arrays
        evaluations = zeros(eltype(krig.x[1]), num_new_samples)  # Holds EI function evaluations
        point_found = false                                     # Whether we have found a new point to test
        new_x_max = zero(eltype(krig.x[1]))                     # New x point
        new_EI_max = zero(eltype(krig.x[1]))                    # EI at new x point
        diff_x = zeros(eltype(krig.x[1]), d)
        while point_found == false
            # For each point in the sample set, evaluate the Expected Improvement function
            for j in 1:length(new_sample)
                std = std_error_at_point(krig, new_sample[j])
                u = krig(new_sample[j])
                if abs(std) > 1e-6
                    z = (f_min - u - eps) / std
                else
                    z = 0
                end
                # Evaluate EI at point new_sample[j]
                evaluations[j] = (f_min - u - eps) * cdf(Normal(), z) +
                                 std * pdf(Normal(), z)
            end
            # find the sample which maximizes the EI function
            index_max = argmax(evaluations)
            x_new = new_sample[index_max]    # x point which maximized EI
            EI_new = maximum(evaluations)    # EI at the new point
            for l in 1:d
                diff_x[l] = norm(krig.x[l] .- x_new)
            end
            bit_x = diff_x .> dtol
            #new_min_x has to have some distance from krig.x
            if false in bit_x
                #The new_point is not actually that new, discard it!
                deleteat!(evaluations, index_max)
                deleteat!(new_sample, index_max)
                if length(new_sample) == 0
                    println("Out of sampling points.")
                    index = argmin(krig.y)
                    return (krig.x[index], krig.y[index])
                end
            else
                point_found = true
                new_x_max = x_new
                new_EI_max = EI_new
            end
        end
        # if the EI is less than some tolerance times the difference between the maximum and minimum points
        # in the surrogate, then we terminate the optimizer.
        if new_EI_max < 1e-6 * norm(maximum(krig.y) - minimum(krig.y))
            index = argmin(krig.y)
            println("Termination tolerance reached.")
            return (krig.x[index], krig.y[index])
        end
        # Otherwise, evaluate the true objective function at the new point and repeat.
        add_point!(krig, Tuple(new_x_max), obj(new_x_max))
    end
    println("Completed maximum number of iterations.")
end

function adjust_step_size(sigma_n, sigma_min, C_success, t_success, C_fail, t_fail)
    if C_success >= t_success
        sigma_n = 2 * sigma_n
        C_success = 0
    end
    if C_fail >= t_fail
        sigma_n = max(sigma_n / 2, sigma_min)
        C_fail = 0
    end
    return sigma_n, C_success, C_fail
end

function select_evaluation_point_1D(new_points1, surr1::AbstractSurrogate, numb_iters,
                                    maxiters)
    v = [0.3, 0.5, 0.8, 0.95]
    k = 4
    n = length(surr1.x)
    if mod(maxiters - 1, 4) != 0
        w_nR = v[mod(maxiters - 1, 4)]
    else
        w_nR = v[4]
    end
    w_nD = 1 - w_nR

    l = length(new_points1)
    evaluations1 = zeros(eltype(surr1.y[1]), l)

    for i in 1:l
        evaluations1[i] = surr1(new_points1[i])
    end
    s_max = maximum(evaluations1)
    s_min = minimum(evaluations1)
    V_nR = zeros(eltype(surr1.y[1]), l)
    for i in 1:l
        if abs(s_max - s_min) <= 10e-6
            V_nR[i] = 1.0
        else
            V_nR[i] = (evaluations1[i] - s_min) / (s_max - s_min)
        end
    end

    #Compute score V_nD
    V_nD = zeros(eltype(surr1.y[1]), l)
    delta_n_x = zeros(eltype(surr1.x[1]), l)
    delta = zeros(eltype(surr1.x[1]), n)
    for j in 1:l
        for i in 1:n
            delta[i] = norm(new_points1[j] - surr1.x[i])
        end
        delta_n_x[j] = minimum(delta)
    end
    delta_n_max = maximum(delta_n_x)
    delta_n_min = minimum(delta_n_x)
    for i in 1:l
        if abs(delta_n_max - delta_n_min) <= 10e-6
            V_nD[i] = 1.0
        else
            V_nD[i] = (delta_n_max - delta_n_x[i]) / (delta_n_max - delta_n_min)
        end
    end

    #Compute weighted score
    W_n = w_nR * V_nR + w_nD * V_nD
    return new_points1[argmin(W_n)]
end

"""
surrogate_optimize(obj::Function,::DYCORS,lb::Number,ub::Number,surr1::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)

DYCORS optimization method in 1D, following closely: Combining radial basis function
surrogates and dynamic coordinate search in high-dimensional expensive black-box optimization".

"""
function surrogate_optimize(obj::Function, ::DYCORS, lb::Number, ub::Number,
                            surr1::AbstractSurrogate, sample_type::SamplingAlgorithm;
                            maxiters = 100, num_new_samples = 100)
    x_best = argmin(surr1.y)
    y_best = minimum(surr1.y)
    sigma_n = 0.2 * norm(ub - lb)
    d = length(lb)
    sigma_min = 0.2 * (0.5)^6 * norm(ub - lb)
    t_success = 3
    t_fail = max(d, 5)
    C_success = 0
    C_fail = 0
    for k in 1:maxiters
        p_select = min(20 / d, 1) * (1 - log(k)) / log(maxiters - 1)
        # In 1D I_perturb is always equal to one, no need to sample
        d = 1
        I_perturb = d
        new_points = zeros(eltype(surr1.x[1]), num_new_samples)
        for i in 1:num_new_samples
            new_points[i] = x_best + rand(Normal(0, sigma_n))
            while new_points[i] < lb || new_points[i] > ub
                if new_points[i] > ub
                    #reflection
                    new_points[i] = max(lb,
                                        maximum(surr1.x) -
                                        norm(new_points[i] - maximum(surr1.x)))
                end
                if new_points[i] < lb
                    #reflection
                    new_points[i] = min(ub,
                                        minimum(surr1.x) +
                                        norm(new_points[i] - minimum(surr1.x)))
                end
            end
        end

        x_new = select_evaluation_point_1D(new_points, surr1, k, maxiters)
        f_new = obj(x_new)

        if f_new < y_best
            C_success = C_success + 1
            C_fail = 0
        else
            C_fail = C_fail + 1
            C_success = 0
        end

        sigma_n, C_success, C_fail = adjust_step_size(sigma_n, sigma_min, C_success,
                                                      t_success, C_fail, t_fail)

        if f_new < y_best
            x_best = x_new
            y_best = f_new
            add_point!(surr1, x_best, y_best)
        end
    end
    index = argmin(surr1.y)
    return (surr1.x[index], surr1.y[index])
end

function select_evaluation_point_ND(new_points, surrn::AbstractSurrogate, numb_iters,
                                    maxiters)
    v = [0.3, 0.5, 0.8, 0.95]
    k = 4
    n = size(surrn.x, 1)
    d = size(surrn.x, 2)
    if mod(maxiters - 1, 4) != 0
        w_nR = v[mod(maxiters - 1, 4)]
    else
        w_nR = v[4]
    end
    w_nD = 1 - w_nR

    l = size(new_points, 1)
    evaluations = zeros(eltype(surrn.y[1]), l)
    for i in 1:l
        evaluations[i] = surrn(Tuple(new_points[i, :]))
    end
    s_max = maximum(evaluations)
    s_min = minimum(evaluations)
    V_nR = zeros(eltype(surrn.y[1]), l)
    for i in 1:l
        if abs(s_max - s_min) <= 10e-6
            V_nR[i] = 1.0
        else
            V_nR[i] = (evaluations[i] - s_min) / (s_max - s_min)
        end
    end

    #Compute score V_nD
    V_nD = zeros(eltype(surrn.y[1]), l)
    delta_n_x = zeros(eltype(surrn.x[1]), l)
    delta = zeros(eltype(surrn.x[1]), n)
    for j in 1:l
        for i in 1:n
            delta[i] = norm(new_points[j, :] - collect(surrn.x[i]))
        end
        delta_n_x[j] = minimum(delta)
    end
    delta_n_max = maximum(delta_n_x)
    delta_n_min = minimum(delta_n_x)
    for i in 1:l
        if abs(delta_n_max - delta_n_min) <= 10e-6
            V_nD[i] = 1.0
        else
            V_nD[i] = (delta_n_max - delta_n_x[i]) / (delta_n_max - delta_n_min)
        end
    end
    #Compute weighted score
    W_n = w_nR * V_nR + w_nD * V_nD
    return new_points[argmin(W_n), :]
end

"""
      surrogate_optimize(obj::Function,::DYCORS,lb::Number,ub::Number,surr1::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)


This is an implementation of the DYCORS strategy by Regis and Shoemaker:
Rommel G Regis and Christine A Shoemaker.
Combining radial basis function surrogates and dynamic coordinate search in high-dimensional expensive black-box optimization.
Engineering Optimization, 45(5): 529–555, 2013.
This is an extension of the SRBF strategy that changes how the
candidate points are generated. The main idea is that many objective
functions depend only on a few directions so it may be advantageous to
perturb only a few directions. In particular, we use a perturbation probability
to perturb a given coordinate and decrease this probability after each function
evaluation so fewer coordinates are perturbed later in the optimization.
"""
function surrogate_optimize(obj::Function, ::DYCORS, lb, ub, surrn::AbstractSurrogate,
                            sample_type::SamplingAlgorithm; maxiters = 100,
                            num_new_samples = 100)
    x_best = collect(surrn.x[argmin(surrn.y)])
    y_best = minimum(surrn.y)
    sigma_n = 0.2 * norm(ub - lb)
    d = length(lb)
    sigma_min = 0.2 * (0.5)^6 * norm(ub - lb)
    t_success = 3
    t_fail = max(d, 5)
    C_success = 0
    C_fail = 0
    for k in 1:maxiters
        p_select = min(20 / d, 1) * (1 - log(k)) / log(maxiters - 1)
        new_points = zeros(eltype(surrn.x[1]), num_new_samples, d)
        for j in 1:num_new_samples
            w = sample(d, 0, 1, sample_type)
            I_perturb = w .< p_select
            if ~(true in I_perturb)
                val = rand(1:d)
                I_perturb = vcat(zeros(Int, val - 1), 1, zeros(Int, d - val))
            end
            I_perturb = Int.(I_perturb)
            for i in 1:d
                if I_perturb[i] == 1
                    new_points[j, i] = x_best[i] + rand(Normal(0, sigma_n))
                else
                    new_points[j, i] = x_best[i]
                end
            end
        end

        for i in 1:num_new_samples
            for j in 1:d
                while new_points[i, j] < lb[j] || new_points[i, j] > ub[j]
                    if new_points[i, j] > ub[j]
                        new_points[i, j] = max(lb[j],
                                               maximum(surrn.x)[j] -
                                               norm(new_points[i, j] - maximum(surrn.x)[j]))
                    end
                    if new_points[i, j] < lb[j]
                        new_points[i, j] = min(ub[j],
                                               minimum(surrn.x)[j] +
                                               norm(new_points[i] - minimum(surrn.x)[j]))
                    end
                end
            end
        end

        #ND version
        x_new = select_evaluation_point_ND(new_points, surrn, k, maxiters)
        f_new = obj(x_new)

        if f_new < y_best
            C_success = C_success + 1
            C_fail = 0
        else
            C_fail = C_fail + 1
            C_success = 0
        end

        sigma_n, C_success, C_fail = adjust_step_size(sigma_n, sigma_min, C_success,
                                                      t_success, C_fail, t_fail)

        if f_new < y_best
            x_best = x_new
            y_best = f_new
            add_point!(surrn, Tuple(x_best), y_best)
        end
    end
    index = argmin(surrn.y)
    return (surrn.x[index], surrn.y[index])
end

function obj2_1D(value, points)
    min = +Inf
    my_p = filter(x -> abs(x - value) > 10^-6, points)
    for i in 1:length(my_p)
        new_val = norm(my_p[i] - value)
        if new_val < min
            min = new_val
        end
    end
    return min
end

function I_tier_ranking_1D(P, surrSOP::AbstractSurrogate)
    #obj1 = objective_function
    #obj2 = obj2_1D
    Fronts = Dict{Int, Array{eltype(surrSOP.x[1]), 1}}()
    i = 1
    while true
        F = []
        j = 1
        for p in P
            n_p = 0
            k = 1
            for q in P
                #I use equality with floats because p and q are in surrSOP.x
                #for sure at this stage
                p_index = j
                q_index = k
                val1_p = surrSOP.y[p_index]
                val2_p = obj2_1D(p, P)
                val1_q = surrSOP.y[q_index]
                val2_q = obj2_1D(q, P)
                p_dominates_q = (val1_p < val1_q || abs(val1_p - val1_q) <= 10^-5) &&
                                (val2_p < val2_q || abs(val2_p - val2_q) <= 10^-5) &&
                                ((val1_p < val1_q) || (val2_p < val2_q))

                q_dominates_p = (val1_p < val1_q || abs(val1_p - val1_q) < 10^-5) &&
                                (val2_p < val2_q || abs(val2_p - val2_q) < 10^-5) &&
                                ((val1_p < val1_q) || (val2_p < val2_q))
                if q_dominates_p
                    n_p += 1
                end
                k = k + 1
            end
            if n_p == 0
                # no individual dominates p
                push!(F, p)
            end
            j = j + 1
        end
        if length(F) > 0
            Fronts[i] = F
            P = setdiff(P, F)
            i = i + 1
        else
            return Fronts
        end
    end
    return F
end

function II_tier_ranking_1D(D::Dict, srg::AbstractSurrogate)
    for i in 1:length(D)
        pos = []
        yn = []
        for j in 1:length(D[i])
            push!(pos, findall(e -> e == D[i][j], srg.x))
            push!(yn, srg.y[pos[j]])
        end
        D[i] = D[i][sortperm(D[i])]
    end
    return D
end

function Hypervolume_Pareto_improving(f1_new, f2_new, Pareto_set)
    if size(Pareto_set, 1) == 1
        area_before = zero(eltype(f1_new))
    else
        my_p = Pareto_set
        #Area before
        v_ref = [maximum(Pareto_set[:, 1]), maximum(Pareto_set[:, 2])]
        my_p = vcat(my_p, v_ref)
        v = sortperm(my_p[:, 2])
        my_p[:, 1] = my_p[:, 1][v]
        my_p[:, 2] = my_p[:, 2][v]
        area_before = zero(eltype(f1_new))
        for j in 1:(length(v) - 1)
            area_before += (my_p[j + 1, 2] - my_p[j, 2]) * (v_ref[1] - my_p[j])
        end
    end
    #Area after
    Pareto_set = vcat(Pareto_set, [f1_new f2_new])
    v_ref = [maximum(Pareto_set[:, 1]) maximum(Pareto_set[:, 2])]
    Pareto_set = vcat(Pareto_set, v_ref)
    v = sortperm(Pareto_set[:, 2])
    Pareto_set[:, 1] = Pareto_set[:, 1][v]
    Pareto_set[:, 2] = Pareto_set[:, 2][v]
    area_after = zero(eltype(f1_new))
    for j in 1:(length(v) - 1)
        area_after += (Pareto_set[j + 1, 2] - Pareto_set[j, 2]) * (v_ref[1] - Pareto_set[j])
    end
    return area_after - area_before
end

"""
surrogate_optimize(obj::Function,::SOP,lb::Number,ub::Number,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)

SOP Surrogate optimization method, following closely the following papers:

    - SOP: parallel surrogate global optimization with Pareto center selection for computationally expensive single objective problems by Tipaluck Krityakierne
    - Multiobjective Optimization Using Evolutionary Algorithms by Kalyan Deb
#Suggested number of new_samples = min(500*d,5000)
"""
function surrogate_optimize(obj::Function, sop1::SOP, lb::Number, ub::Number,
                            surrSOP::AbstractSurrogate, sample_type::SamplingAlgorithm;
                            maxiters = 100, num_new_samples = min(500 * 1, 5000))
    d = length(lb)
    N_fail = 3
    N_tenure = 5
    tau = 10^-5
    num_P = sop1.p
    centers_global = surrSOP.x
    r_centers_global = 0.2 * norm(ub - lb) * ones(length(surrSOP.x))
    N_failures_global = zeros(length(surrSOP.x))
    tabu = []
    N_tenures_tabu = []
    for k in 1:maxiters
        N_tenures_tabu .+= 1
        #deleting points that have been in tabu for too long
        del = N_tenures_tabu .> N_tenure

        if length(del) > 0
            for i in 1:length(del)
                if del[i]
                    del[i] = i
                end
            end
            deleteat!(N_tenures_tabu, del)
            deleteat!(tabu, del)
        end

        ##### P CENTERS ######
        C = []

        #S(x) set of points already evaluated
        #Rank points in S with:
        #1) Non dominated sorting
        Fronts_I = I_tier_ranking_1D(centers_global, surrSOP)
        #2) Second tier ranking
        Fronts = II_tier_ranking_1D(Fronts_I, surrSOP)
        ranked_list = []
        for i in 1:length(Fronts)
            for j in 1:length(Fronts[i])
                push!(ranked_list, Fronts[i][j])
            end
        end
        ranked_list = eltype(surrSOP.x[1]).(ranked_list)

        centers_full = 0
        i = 1
        while i <= length(ranked_list) && centers_full == 0
            flag = 0
            for j in 1:length(ranked_list)
                for m in 1:length(tabu)
                    if abs(ranked_list[j] - tabu[m]) < tau
                        flag = 1
                    end
                end
                for l in 1:length(centers_global)
                    if abs(ranked_list[j] - centers_global[l]) < tau
                        flag = 1
                    end
                end
            end
            if flag == 1
                skip
            else
                push!(C, ranked_list[i])
                if length(C) == num_P
                    centers_full = 1
                end
            end
            i = i + 1
        end

        #I examined all the points in the ranked list but num_selected < num_p
        #I just iterate again using only radius rule
        if length(C) < num_P
            i = 1
            while i <= length(ranked_list) && centers_full == 0
                flag = 0
                for j in 1:length(ranked_list)
                    for m in 1:length(centers_global)
                        if abs(centers_global[j] - ranked_list[m]) < tau
                            flag = 1
                        end
                    end
                end
                if flag == 1
                    skip
                else
                    push!(C, ranked_list[i])
                    if length(C) == num_P
                        centers_full = 1
                    end
                end
                i = i + 1
            end
        end

        #If I still have num_selected < num_P, I double down on some centers iteratively
        if length(C) < num_P
            i = 1
            while i <= length(ranked_list)
                push!(C, ranked_list[i])
                if length(C) == num_P
                    centers_full = 1
                end
                i = i + 1
            end
        end

        #Here I have selected C = [] containing the centers
        r_centers = 0.2 * norm(ub - lb) * ones(num_P)
        N_failures = zeros(num_P)
        #2.3 Candidate search
        new_points = zeros(eltype(surrSOP.x[1]), num_P, 2)
        for i in 1:num_P
            N_candidates = zeros(eltype(surrSOP.x[1]), num_new_samples)
            #Using phi(n) just like DYCORS, merit function = surrogate
            #Like in DYCORS, I_perturb = 1 always
            evaluations = zeros(eltype(surrSOP.y[1]), num_new_samples)
            for j in 1:num_new_samples
                a = lb - C[i]
                b = ub - C[i]
                N_candidates[j] = C[i] + rand(truncated(Normal(0, r_centers[i]), a, b))
                evaluations[j] = surrSOP(N_candidates[j])
            end
            x_best = N_candidates[argmin(evaluations)]
            y_best = minimum(evaluations)
            new_points[i, 1] = x_best
            new_points[i, 2] = y_best
        end

        #new_points[i] now contains:
        #[x_1,y_1; x_2,y_2,...,x_{num_new_samples},y_{num_new_samples}]

        #2.4 Adaptive learning and tabu archive
        for i in 1:num_P
            if new_points[i, 1] in centers_global
                r_centers[i] = r_centers_global[i]
                N_failures[i] = N_failures_global[i]
            end

            f_1 = obj(new_points[i, 1])
            f_2 = obj2_1D(f_1, surrSOP.x)

            l = length(Fronts[1])
            Pareto_set = zeros(eltype(surrSOP.x[1]), l, 2)

            for j in 1:l
                val = obj2_1D(Fronts[1][j], surrSOP.x)
                Pareto_set[j, 1] = obj(Fronts[1][j])
                Pareto_set[j, 2] = val
            end
            if (Hypervolume_Pareto_improving(f_1, f_2, Pareto_set) < tau)
                #failure
                r_centers[i] = r_centers[i] / 2
                N_failures[i] += 1
                if N_failures[i] > N_fail
                    push!(tabu, C[i])
                    push!(N_tenures_tabu, 0)
                end
            else
                #P_i is success
                #Adaptive_learning
                add_point!(surrSOP, new_points[i, 1], new_points[i, 2])
                push!(r_centers_global, r_centers[i])
                push!(N_failures_global, N_failures[i])
            end
        end
    end
    index = argmin(surrSOP.y)
    return (surrSOP.x[index], surrSOP.y[index])
end

function obj2_ND(value, points)
    min = +Inf
    my_p = filter(x -> norm(x .- value) > 10^-6, points)
    for i in 1:length(my_p)
        new_val = norm(my_p[i] .- value)
        if new_val < min
            min = new_val
        end
    end
    return min
end

function I_tier_ranking_ND(P, surrSOPD::AbstractSurrogate)
    #obj1 = objective_function
    #obj2 = obj2_1D
    Fronts = Dict{Int, Array{eltype(surrSOPD.x), 1}}()
    i = 1
    while true
        F = Array{eltype(surrSOPD.x), 1}()
        j = 1
        for p in P
            n_p = 0
            k = 1
            for q in P
                #I use equality with floats because p and q are in surrSOP.x
                #for sure at this stage
                p_index = j
                q_index = k
                val1_p = surrSOPD.y[p_index]
                val2_p = obj2_ND(p, P)
                val1_q = surrSOPD.y[q_index]
                val2_q = obj2_ND(q, P)
                p_dominates_q = (val1_p < val1_q || abs(val1_p - val1_q) <= 10^-5) &&
                                (val2_p < val2_q || abs(val2_p - val2_q) <= 10^-5) &&
                                ((val1_p < val1_q) || (val2_p < val2_q))

                q_dominates_p = (val1_p < val1_q || abs(val1_p - val1_q) < 10^-5) &&
                                (val2_p < val2_q || abs(val2_p - val2_q) < 10^-5) &&
                                ((val1_p < val1_q) || (val2_p < val2_q))
                if q_dominates_p
                    n_p += 1
                end
                k = k + 1
            end
            if n_p == 0
                # no individual dominates p
                push!(F, p)
            end
            j = j + 1
        end
        if length(F) > 0
            Fronts[i] = F
            P = setdiff(P, F)
            i = i + 1
        else
            return Fronts
        end
    end
    return F
end

function II_tier_ranking_ND(D::Dict, srgD::AbstractSurrogate)
    for i in 1:length(D)
        pos = []
        yn = []
        for j in 1:length(D[i])
            push!(pos, findall(e -> e == D[i][j], srgD.x))
            push!(yn, srgD.y[pos[j]])
        end
        D[i] = D[i][sortperm(D[i])]
    end
    return D
end

function surrogate_optimize(obj::Function, sopd::SOP, lb, ub, surrSOPD::AbstractSurrogate,
                            sample_type::SamplingAlgorithm; maxiters = 100,
                            num_new_samples = min(500 * length(lb), 5000))
    d = length(lb)
    N_fail = 3
    N_tenure = 5
    tau = 10^-5
    num_P = sopd.p
    centers_global = surrSOPD.x
    r_centers_global = 0.2 * norm(ub .- lb) * ones(length(surrSOPD.x))
    N_failures_global = zeros(length(surrSOPD.x))
    tabu = []
    N_tenures_tabu = []
    for k in 1:maxiters
        N_tenures_tabu .+= 1
        #deleting points that have been in tabu for too long
        del = N_tenures_tabu .> N_tenure

        if length(del) > 0
            for i in 1:length(del)
                if del[i]
                    del[i] = i
                end
            end
            deleteat!(N_tenures_tabu, del)
            deleteat!(tabu, del)
        end

        ##### P CENTERS ######
        C = Array{eltype(surrSOPD.x), 1}()

        #S(x) set of points already evaluated
        #Rank points in S with:
        #1) Non dominated sorting
        Fronts_I = I_tier_ranking_ND(centers_global, surrSOPD)
        #2) Second tier ranking
        Fronts = II_tier_ranking_ND(Fronts_I, surrSOPD)
        ranked_list = Array{eltype(surrSOPD.x), 1}()
        for i in 1:length(Fronts)
            for j in 1:length(Fronts[i])
                push!(ranked_list, Fronts[i][j])
            end
        end

        centers_full = 0
        i = 1
        while i <= length(ranked_list) && centers_full == 0
            flag = 0
            for j in 1:length(ranked_list)
                for m in 1:length(tabu)
                    if norm(ranked_list[j] .- tabu[m]) < tau
                        flag = 1
                    end
                end
                for l in 1:length(centers_global)
                    if norm(ranked_list[j] .- centers_global[l]) < tau
                        flag = 1
                    end
                end
            end
            if flag == 1
                skip
            else
                push!(C, ranked_list[i])
                if length(C) == num_P
                    centers_full = 1
                end
            end
            i = i + 1
        end

        #I examined all the points in the ranked list but num_selected < num_p
        #I just iterate again using only radius rule
        if length(C) < num_P
            i = 1
            while i <= length(ranked_list) && centers_full == 0
                flag = 0
                for j in 1:length(ranked_list)
                    for m in 1:length(centers_global)
                        if norm(centers_global[j] .- ranked_list[m]) < tau
                            flag = 1
                        end
                    end
                end
                if flag == 1
                    skip
                else
                    push!(C, ranked_list[i])
                    if length(C) == num_P
                        centers_full = 1
                    end
                end
                i = i + 1
            end
        end

        #If I still have num_selected < num_P, I double down on some centers iteratively
        if length(C) < num_P
            i = 1
            while i <= length(ranked_list)
                push!(C, ranked_list[i])
                if length(C) == num_P
                    centers_full = 1
                end
                i = i + 1
            end
        end

        #Here I have selected C = [(1.0,2.0),(3.0,4.0),.....] containing the centers
        r_centers = 0.2 * norm(ub .- lb) * ones(num_P)
        N_failures = zeros(num_P)
        #2.3 Candidate search
        new_points_x = Array{eltype(surrSOPD.x), 1}()
        new_points_y = zeros(eltype(surrSOPD.y[1]), num_P)
        for i in 1:num_P
            N_candidates = zeros(eltype(surrSOPD.x[1]), num_new_samples, d)
            #Using phi(n) just like DYCORS, merit function = surrogate
            #Like in DYCORS, I_perturb = 1 always
            evaluations = zeros(eltype(surrSOPD.y[1]), num_new_samples)
            for j in 1:num_new_samples
                for k in 1:d
                    a = lb[k] - C[i][k]
                    b = ub[k] - C[i][k]
                    N_candidates[j, k] = C[i][k] +
                                         rand(truncated(Normal(0, r_centers[i]), a, b))
                end
                evaluations[j] = surrSOPD(Tuple(N_candidates[j, :]))
            end
            x_best = Tuple(N_candidates[argmin(evaluations), :])
            y_best = minimum(evaluations)
            push!(new_points_x, x_best)
            new_points_y[i] = y_best
        end

        #new_points[i] is splitted in new_points_x and new_points_y now contains:
        #[x_1,y_1; x_2,y_2,...,x_{num_new_samples},y_{num_new_samples}]

        #2.4 Adaptive learning and tabu archive
        for i in 1:num_P
            if new_points_x[i] in centers_global
                r_centers[i] = r_centers_global[i]
                N_failures[i] = N_failures_global[i]
            end

            f_1 = obj(Tuple(new_points_x[i]))
            f_2 = obj2_ND(f_1, surrSOPD.x)

            l = length(Fronts[1])
            Pareto_set = zeros(eltype(surrSOPD.x[1]), l, 2)
            for j in 1:l
                val = obj2_ND(Fronts[1][j], surrSOPD.x)
                Pareto_set[j, 1] = obj(Tuple(Fronts[1][j]))
                Pareto_set[j, 2] = val
            end
            if (Hypervolume_Pareto_improving(f_1, f_2, Pareto_set) < tau)#check this
                #failure
                r_centers[i] = r_centers[i] / 2
                N_failures[i] += 1
                if N_failures[i] > N_fail
                    push!(tabu, C[i])
                    push!(N_tenures_tabu, 0)
                end
            else
                #P_i is success
                #Adaptive_learning
                add_point!(surrSOPD, new_points_x[i], new_points_y[i])
                push!(r_centers_global, r_centers[i])
                push!(N_failures_global, N_failures[i])
            end
        end
    end
    index = argmin(surrSOPD.y)
    return (surrSOPD.x[index], surrSOPD.y[index])
end

#EGO

_dominates(x, y) = all(x .<= y) && any(x .< y)
function _nonDominatedSorting(arr::Array{Float64, 2})
    fronts::Array{Array, 1} = Array[]
    ind::Array{Int64, 1} = collect(1:size(arr, 1))
    while !isempty(arr)
        s = size(arr, 1)
        red = dropdims(sum([_dominates(arr[i, :], arr[j, :]) for i in 1:s, j in 1:s],
                           dims = 1) .== 0, dims = 1)
        a = 1:s
        sel::Array{Int64, 1} = a[red]
        push!(fronts, ind[sel])
        da::Array{Int64, 1} = deleteat!(collect(1:s), sel)
        ind = deleteat!(ind, sel)
        arr = arr[da, :]
    end
    return fronts
end

function surrogate_optimize(obj::Function, sbm::SMB, lb::Number, ub::Number,
                            surrSMB::AbstractSurrogate, sample_type::SamplingAlgorithm;
                            maxiters = 100, n_new_look = 1000)
    #obj contains a function for each output dimension
    dim_out = length(surrSMB.y[1])
    d = 1
    x_to_look = sample(n_new_look, lb, ub, sample_type)
    for iter in 1:maxiters
        index_min = 0
        min_mean = +Inf
        for i in 1:n_new_look
            new_mean = sum(obj(x_to_look[i])) / dim_out
            if new_mean < min_mean
                min_mean = new_mean
                index_min = i
            end
        end

        x_new = x_to_look[index_min]
        deleteat!(x_to_look, index_min)
        n_new_look = n_new_look - 1
        # evaluate the true function at that point
        y_new = obj(x_new)
        #update the surrogate
        add_point!(surrSMB, x_new, y_new)
    end
    #Find and return Pareto
    y = surrSMB.y
    y = permutedims(reshape(hcat(y...), (length(y[1]), length(y)))) #2d matrix
    Fronts = _nonDominatedSorting(y) #this returns the indexes
    pareto_front_index = Fronts[1]
    pareto_set = []
    pareto_front = []
    for i in 1:length(pareto_front_index)
        push!(pareto_set, surrSMB.x[pareto_front_index[i]])
        push!(pareto_front, surrSMB.y[pareto_front_index[i]])
    end
    return pareto_set, pareto_front
end

function surrogate_optimize(obj::Function, smb::SMB, lb, ub, surrSMBND::AbstractSurrogate,
                            sample_type::SamplingAlgorithm; maxiters = 100,
                            n_new_look = 1000)
    #obj contains a function for each output dimension
    dim_out = length(surrSMBND.y[1])
    d = length(lb)
    x_to_look = sample(n_new_look, lb, ub, sample_type)
    for iter in 1:maxiters
        index_min = 0
        min_mean = +Inf
        for i in 1:n_new_look
            new_mean = sum(obj(x_to_look[i])) / dim_out
            if new_mean < min_mean
                min_mean = new_mean
                index_min = i
            end
        end
        x_new = x_to_look[index_min]
        deleteat!(x_to_look, index_min)
        n_new_look = n_new_look - 1
        # evaluate the true function at that point
        y_new = obj(x_new)
        #update the surrogate
        add_point!(surrSMBND, x_new, y_new)
    end
    #Find and return Pareto
    y = surrSMBND.y
    y = permutedims(reshape(hcat(y...), (length(y[1]), length(y)))) #2d matrix
    Fronts = _nonDominatedSorting(y) #this returns the indexes
    pareto_front_index = Fronts[1]
    pareto_set = []
    pareto_front = []
    for i in 1:length(pareto_front_index)
        push!(pareto_set, surrSMBND.x[pareto_front_index[i]])
        push!(pareto_front, surrSMBND.y[pareto_front_index[i]])
    end
    return pareto_set, pareto_front
end

# RTEA (Noisy model based multi objective optimization + standard rtea by fieldsen), use this for very noisy objective functions because there are a lot of re-evaluations

function surrogate_optimize(obj, rtea::RTEA, lb::Number, ub::Number,
                            surrRTEA::AbstractSurrogate, sample_type::SamplingAlgorithm;
                            maxiters = 100, n_new_look = 1000)
    Z = rtea.z
    K = rtea.k
    p_cross = rtea.p
    n_c = rtea.n_c
    sigma = rtea.sigma
    #find pareto set of the first evaluations: (estimated pareto)
    y = surrRTEA.y
    y = permutedims(reshape(hcat(y...), (length(y[1]), length(y)))) #2d matrix
    Fronts = _nonDominatedSorting(y) #this returns the indexes
    pareto_front_index = Fronts[1]
    pareto_set = []
    pareto_front = []
    for i in 1:length(pareto_front_index)
        push!(pareto_set, surrRTEA.x[pareto_front_index[i]])
        push!(pareto_front, surrRTEA.y[pareto_front_index[i]])
    end
    number_of_revaluations = zeros(Int, length(pareto_set))
    iter = 1
    d = 1
    dim_out = length(surrRTEA.y[1])
    while iter < maxiters
        if iter < (1 - Z) * maxiters
            #1) propose new point x_new

            #sample randomly from (estimated) pareto v and u
            if length(pareto_set) < 2
                throw("Starting pareto set is too small, increase number of sampling point of the surrogate")
            end
            u = pareto_set[rand(1:length(pareto_set))]
            v = pareto_set[rand(1:length(pareto_set))]

            #children
            if rand() < p_cross
                mu = rand()
                if mu <= 0.5
                    beta = (2 * mu)^(1 / n_c + 1)
                else
                    beta = (1 / (2 * (1 - mu)))^(1 / n_c + 1)
                end
                x = 0.5 * ((1 + beta) * v + (1 - beta) * u)
            else
                x = v
            end

            #mutation
            x_new = x + rand(Normal(0, sigma))
            y_new = obj(x_new)

            #update pareto
            new_to_pareto = false
            for i in 1:length(pareto_set)
                counter = zeros(Int, dim_out)
                #compare the y_new values to pareto, if there is at least one entry where it dominates all the others, then it can be in pareto
                for l in 1:dim_out
                    if y_new[l] < pareto_front[i][l]
                        counter[l]
                    end
                end
            end
            for j in 1:dim_out
                if counter[j] == dim_out
                    new_to_pareto = true
                end
            end
            if new_to_pareto == true
                push!(pareto_set, x_new)
                push!(pareto_front, y_new)
                push!(number_of_revaluations, 0)
            end
            add_point!(surrRTEA, new_x, new_y)
        end
        for k in 1:K
            val, pos = findmin(number_of_revaluations)
            x_r = pareto_set[pos]
            y_r = obj(x_r)
            number_of_revaluations[pos] = number_of_revaluations + 1
            #check if it is again in the pareto set or not, if not eliminate it from pareto
            still_in_pareto = false
            for i in 1:length(pareto_set)
                counter = zeros(Int, dim_out)
                for l in 1:dim_out
                    if y_r[l] < pareto_front[i][l]
                        counter[l]
                    end
                end
            end
            for j in 1:dim_out
                if counter[j] == dim_out
                    still_in_pareto = true
                end
            end
            if still_in_pareto == false
                #remove from pareto
                deleteat!(pareto_set, pos)
                deleteat!(pareto_front, pos)
                deleteat!(number_of_revaluationsm, pos)
            end
        end
        iter = iter + 1
    end
    return pareto_set, pareto_front
end

function surrogate_optimize(obj, rtea::RTEA, lb, ub, surrRTEAND::AbstractSurrogate,
                            sample_type::SamplingAlgorithm; maxiters = 100,
                            n_new_look = 1000)
    Z = rtea.z
    K = rtea.k
    p_cross = rtea.p
    n_c = rtea.n_c
    sigma = rtea.sigma
    #find pareto set of the first evaluations: (estimated pareto)
    y = surrRTEAND.y
    y = permutedims(reshape(hcat(y...), (length(y[1]), length(y)))) #2d matrix
    Fronts = _nonDominatedSorting(y) #this returns the indexes
    pareto_front_index = Fronts[1]
    pareto_set = []
    pareto_front = []
    for i in 1:length(pareto_front_index)
        push!(pareto_set, surrRTEAND.x[pareto_front_index[i]])
        push!(pareto_front, surrRTEAND.y[pareto_front_index[i]])
    end
    number_of_revaluations = zeros(Int, length(pareto_set))
    iter = 1
    d = length(lb)
    dim_out = length(surrRTEAND.y[1])
    while iter < maxiters
        if iter < (1 - Z) * maxiters

            #sample pareto_set
            if length(pareto_set) < 2
                throw("Starting pareto set is too small, increase number of sampling point of the surrogate")
            end
            u = pareto_set[rand(1:length(pareto_set))]
            v = pareto_set[rand(1:length(pareto_set))]

            #children
            if rand() < p_cross
                mu = rand()
                if mu <= 0.5
                    beta = (2 * mu)^(1 / n_c + 1)
                else
                    beta = (1 / (2 * (1 - mu)))^(1 / n_c + 1)
                end
                x = 0.5 * ((1 + beta) * v + (1 - beta) * u)
            else
                x = v
            end

            #mutation
            for i in 1:d
                x_new[i] = x[i] + rand(Normal(0, sigma))
            end
            y_new = obj(x_new)

            #update pareto
            new_to_pareto = false
            for i in 1:length(pareto_set)
                counter = zeros(Int, dim_out)
                #compare the y_new values to pareto, if there is at least one entry where it dominates all the others, then it can be in pareto
                for l in 1:dim_out
                    if y_new[l] < pareto_front[i][l]
                        counter[l]
                    end
                end
            end
            for j in 1:dim_out
                if counter[j] == dim_out
                    new_to_pareto = true
                end
            end
            if new_to_pareto == true
                push!(pareto_set, x_new)
                push!(pareto_front, y_new)
                push!(number_of_revaluations, 0)
            end
            add_point!(surrRTEAND, new_x, new_y)
        end
        for k in 1:K
            val, pos = findmin(number_of_revaluations)
            x_r = pareto_set[pos]
            y_r = obj(x_r)
            number_of_revaluations[pos] = number_of_revaluations + 1
            #check if it is again in the pareto set or not, if not eliminate it from pareto
            still_in_pareto = false
            for i in 1:length(pareto_set)
                counter = zeros(Int, dim_out)
                for l in 1:dim_out
                    if y_r[l] < pareto_front[i][l]
                        counter[l]
                    end
                end
            end
            for j in 1:dim_out
                if counter[j] == dim_out
                    still_in_pareto = true
                end
            end
            if still_in_pareto == false
                #remove from pareto
                deleteat!(pareto_set, pos)
                deleteat!(pareto_front, pos)
                deleteat!(number_of_revaluationsm, pos)
            end
        end
        iter = iter + 1
    end
    return pareto_set, pareto_front
end

function surrogate_optimize(obj::Function, ::EI, lb, ub, krig, sample_type::SectionSample;
                            maxiters = 100, num_new_samples = 100)
    dtol = 1e-3 * norm(ub - lb)
    eps = 0.01
    for i in 1:maxiters
        d = length(krig.x)
        # Sample lots of points from the design space -- we will evaluate the EI function at these points
        new_sample = sample(num_new_samples, lb, ub, sample_type)

        # Find the best point so far
        f_min = minimum(krig.y)

        # Allocate some arrays
        evaluations = zeros(eltype(krig.x[1]), num_new_samples)  # Holds EI function evaluations
        point_found = false                                     # Whether we have found a new point to test
        new_x_max = zero(eltype(krig.x[1]))                     # New x point
        new_EI_max = zero(eltype(krig.x[1]))                    # EI at new x point
        diff_x = zeros(eltype(krig.x[1]), d)

        # For each point in the sample set, evaluate the Expected Improvement function
        while point_found == false
            for j in 1:length(new_sample)
                std = std_error_at_point(krig, new_sample[j])
                u = krig(new_sample[j])
                if abs(std) > 1e-6
                    z = (f_min - u - eps) / std
                else
                    z = 0
                end
                # Evaluate EI at point new_sample[j]
                evaluations[j] = (f_min - u - eps) * cdf(Normal(), z) +
                                 std * pdf(Normal(), z)
            end
            # find the sample which maximizes the EI function
            index_max = argmax(evaluations)
            x_new = new_sample[index_max]   # x point which maximized EI
            EI_new = maximum(evaluations)   # EI at the new point
            for l in 1:d
                diff_x[l] = norm(krig.x[l] .- x_new)
            end
            bit_x = diff_x .> dtol
            #new_min_x has to have some distance from krig.x
            if false in bit_x
                #The new_point is not actually that new, discard it!
                deleteat!(evaluations, index_max)
                deleteat!(new_sample, index_max)
                if length(new_sample) == 0
                    println("Out of sampling points.")
                    return section_sampler_returner(sample_type, krig.x, krig.y, lb, ub,
                                                    krig)
                end
            else
                point_found = true
                new_x_max = x_new
                new_EI_max = EI_new
            end
        end
        # if the EI is less than some tolerance times the difference between the maximum and minimum points
        # in the surrogate, then we terminate the optimizer.
        if new_EI_max < 1e-6 * norm(maximum(krig.y) - minimum(krig.y))
            println("Termination tolerance reached.")
            return section_sampler_returner(sample_type, krig.x, krig.y, lb, ub, krig)
        end
        add_point!(krig, Tuple(new_x_max), obj(new_x_max))
    end
    println("Completed maximum number of iterations.")
end

function section_sampler_returner(sample_type::SectionSample, surrn_x, surrn_y,
                                  lb, ub, surrn)
    d_fixed = QuasiMonteCarlo.fixed_dimensions(sample_type)
    @assert length(surrn_y) == size(surrn_x)[1]
    surrn_xy = [(surrn_x[y], surrn_y[y]) for y in 1:length(surrn_y)]
    section_surr1_xy = filter(xyz -> xyz[1][d_fixed] == Tuple(sample_type.x0[d_fixed]),
                              surrn_xy)
    section_surr1_x = [xy[1] for xy in section_surr1_xy]
    section_surr1_y = [xy[2] for xy in section_surr1_xy]
    if length(section_surr1_xy) == 0
        @debug "No new point added - surrogate locally stable"
        N_NEW_POINTS = 100
        section_surr1_x = sample(N_NEW_POINTS, lb, ub, sample_type)
        section_surr1_y = zeros(N_NEW_POINTS)
        for i in 1:size(section_surr1_x, 1)
            xi = Tuple([section_surr1_x[i, :]...])[1]
            section_surr1_y[i] = surrn(xi)
        end
    end
    index = argmin(section_surr1_y)
    return (section_surr1_x[index, :][1], section_surr1_y[index])
end
