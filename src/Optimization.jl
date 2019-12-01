using LinearAlgebra
using Distributions

abstract type SurrogateOptimizationAlgorithm end

struct SRBF <: SurrogateOptimizationAlgorithm end
struct LCBS <: SurrogateOptimizationAlgorithm end
struct EI <: SurrogateOptimizationAlgorithm end
struct DYCORS <: SurrogateOptimizationAlgorithm end
struct SOP{P} <: SurrogateOptimizationAlgorithm
    p::P
end

function merit_function(point,w,surr::AbstractSurrogate,s_max,s_min,d_max,d_min,box_size)
    if length(point)==1
        D_x = box_size+1
        for i = 1:length(surr.x)
            distance = norm(surr.x[i]-point)
            if distance < D_x
                D_x = distance
            end
        end
        return w*(surr(point) - s_min)/(s_max-s_min) + (1-w)*((d_max - D_x)/(d_max - d_min))
    else
        D_x = norm(box_size)+1
        for i = 1:length(surr.x)
            distance = norm(surr.x[i] .- point)
            if distance < D_x
                D_x = distance
            end
        end
        return w*(surr(point) - s_min)/(s_max-s_min) + (1-w)*((d_max - D_x)/(d_max - d_min))
    end
end




"""
The main idea is to pick the new evaluations from a set of candidate points where each candidate point is generated as an N(0, sigma^2)
distributed perturbation from the current best solution.
The value of sigma is modified based on progress and follows the same logic as
in many trust region methods; we increase sigma if we make a lot of progress
(the surrogate is accurate) and decrease sigma when we aren’t able to make progress
(the surrogate model is inaccurate).
More details about how sigma is updated is given in the original papers.

After generating the candidate points we predict their objective function value
and compute the minimum distance to previously evaluated point.
Let the candidate points be denoted by C and let the function value predictions
be s(x\\_i) and the distance values be d(x\\_i), both rescaled through a
linear transformation to the interval [0,1]. This is done to put the values on
the same scale.
The next point selected for evaluation is the candidate point x that minimizes
the weighted-distance merit function:

``merit(x) = ws(x) + (1-w)(1-d(x))``

where `` 0 \\leq w \\leq 1 ``.
That is, we want a small function value prediction and a large minimum distance
from previously evalauted points.
The weight w is commonly cycled between
a few values to achieve both exploitation and exploration.
When w is close to zero we do pure exploration while w close to 1 corresponds to explotation.
"""
function surrogate_optimize(obj::Function,::SRBF,lb,ub,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
    scale = 0.2
    success = 0
    failure = 0
    w_range = [0.3,0.5,0.7,0.95]

    #Vector containing size in each direction
    box_size = lb-ub
    success = 0
    failures = 0
    dtol = 1e-3*norm(ub-lb)
    d = length(surr.x)
    num_of_iterations = 0
    for w in Iterators.cycle(w_range)
        num_of_iterations += 1
        if num_of_iterations == maxiters
            index = argmin(surr.y)
            return (surr.x[index],surr.y[index])
        end
        for k = 1:maxiters
            incumbent_value = minimum(surr.y)
            incumbent_x = surr.x[argmin(surr.y)]

            new_lb = incumbent_x .- 3*scale*norm(incumbent_x .-lb)
            new_ub = incumbent_x .+ 3*scale*norm(incumbent_x .-ub)

            @inbounds for i = 1:length(new_lb)
                if new_lb[i] < lb[i]
                    new_lb = collect(new_lb)
                    new_lb[i] = lb[i]
                end
                if new_ub[i] > ub[i]
                    new_ub = collect(new_ub)
                    new_ub[i] = ub[i]
                end
            end

            new_sample = sample(num_new_samples,new_lb,new_ub,sample_type)
            s = zeros(eltype(surr.x[1]),num_new_samples)
            for j = 1:num_new_samples
                s[j] = surr(new_sample[j])
            end
            s_max = maximum(s)
            s_min = minimum(s)

            d_min = norm(box_size .+ 1)
            d_max = 0.0
            for r = 1:length(surr.x)
                for c = 1:num_new_samples
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

            evaluation_of_merit_function = zeros(float(eltype(surr.x[1])),num_new_samples)
            @inbounds for r = 1:num_new_samples
                evaluation_of_merit_function[r] = merit_function(new_sample[r],w,surr,s_max,s_min,d_max,d_min,box_size)
            end
            new_addition = false
            adaptive_point_x = Tuple{}
            diff_x = zeros(eltype(surr.x[1]),d)
            while new_addition == false
                #find minimum
                new_min_y = minimum(evaluation_of_merit_function)
                min_index = argmin(evaluation_of_merit_function)
                new_min_x = new_sample[min_index]
                for l = 1:d
                    diff_x[l] = norm(surr.x[l] .- new_min_x)
                end
                bit_x = diff_x .> dtol
                #new_min_x has to have some distance from krig.x
                if false in bit_x
                    #The new_point is not actually that new, discard it!

                    deleteat!(evaluation_of_merit_function,min_index[1])
                    deleteat!(new_sample,min_index)

                    if length(new_sample) == 0
                        println("Out of sampling points")
                        index = argmin(surr.y)
                        return (surr.x[index],surr.y[index])
                    end
                else
                    new_addition = true
                    adaptive_point_x = Tuple(new_min_x)
                end
            end

            #4) Evaluate objective function at adaptive point
            adaptive_point_y = obj(adaptive_point_x)

            #5) Update surrogate with (adaptive_point,objective(adaptive_point)
            add_point!(surr,adaptive_point_x,adaptive_point_y)

            #6) How to go on?
            if surr(adaptive_point_x) < incumbent_value
                #success
                incumbent_x = adaptive_point_x
                incumbent_value = adaptive_point_y
                if failure == 0
                    success +=1
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
                scale = scale*2
                if scale > 0.8*norm(ub-lb)
                    println("Exiting, scale too big")
                    index = argmin(surr.y)
                    return (surr.x[index],surr.y[index])
                end
                success = 0
                failure = 0
            end

            if failure == 5
                scale = scale/2
                #check bounds and go on only if > 1e-5*interval
                if scale < 1e-5
                    println("Exiting, too narrow")
                    index = argmin(surr.y)
                    return (surr.x[index],surr.y[index])
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
function surrogate_optimize(obj::Function,::SRBF,lb::Number,ub::Number,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
#Suggested by:
#https://www.mathworks.com/help/gads/surrogate-optimization-algorithm.html
    scale = 0.2
    success = 0
    failure = 0
    w_range = [0.3,0.5,0.7,0.95]
    box_size = lb-ub
    success = 0
    failures = 0
    dtol = 1e-3*norm(ub-lb)
    num_of_iterations = 0
    for w in Iterators.cycle(w_range)
        num_of_iterations += 1
        if num_of_iterations == maxiters
            index = argmin(surr.y)
            return (surr.x[index],surr.y[index])
        end
        for k = 1:maxiters
            #1) Sample near incumbent (the 2 fraction is arbitrary here)
            incumbent_value = minimum(surr.y)
            incumbent_x = surr.x[argmin(surr.y)]

            new_lb = incumbent_x-scale*norm(incumbent_x-lb)
            new_ub = incumbent_x+scale*norm(incumbent_x-ub)
            if new_lb < lb
                new_lb = lb
            end
            if new_ub > ub
                new_ub = ub
            end
            new_sample = sample(num_new_samples,new_lb,new_ub,sample_type)

            #2) Create  merit function
            s = zeros(eltype(surr.x[1]),num_new_samples)
            for j = 1:num_new_samples
                s[j] = surr(new_sample[j])
            end
            s_max = maximum(s)
            s_min = minimum(s)

            d_min = box_size + 1
            d_max = 0.0
            for r = 1:length(surr.x)
                for c = 1:num_new_samples
                    distance_rc = norm(surr.x[r]-new_sample[c])
                    if distance_rc > d_max
                        d_max = distance_rc
                    end
                    if distance_rc < d_min
                        d_min = distance_rc
                    end
                end
            end
            #3) Evaluate merit function in the sampled points
            evaluation_of_merit_function = merit_function.(new_sample,w,surr,s_max,s_min,d_max,d_min,box_size)

            new_addition = false
            adaptive_point_x = zero(eltype(new_sample[1]));
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
                    deleteat!(evaluation_of_merit_function,min_index)
                    deleteat!(new_sample,min_index)
                    if length(new_sample) == 0
                        println("Out of sampling points")
                        index = argmin(surr.y)
                        return (surr.x[index],surr.y[index])
                    end
                else
                new_addition = true
                adaptive_point_x = new_min_x
                end
            end
            #4) Evaluate objective function at adaptive point
            adaptive_point_y = obj(adaptive_point_x)

            #5) Update surrogate with (adaptive_point,objective(adaptive_point)
            add_point!(surr,adaptive_point_x,adaptive_point_y)

            #6) How to go on?
            if surr(adaptive_point_x) < incumbent_value
                #success
                incumbent_x = adaptive_point_x
                incumbent_value = adaptive_point_y
                if failure == 0
                    success +=1
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
                scale = scale*2
                #check bounds cant go more than [a,b]
                if scale > 0.8*norm(ub-lb)
                    println("Exiting,scale too big")
                    index = argmin(surr.y)
                    return (surr.x[index],surr.y[index])
                end
                success = 0
                failure = 0
            end

            if failure == 5
                scale = scale/2
                #check bounds and go on only if > 1e-5*interval
                if scale < 1e-5
                    println("Exiting, too narrow")
                    index = argmin(surr.y)
                    return (surr.x[index],surr.y[index])
                end
                sucess = 0
                failure = 0
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
function surrogate_optimize(obj::Function,::LCBS,lb::Number,ub::Number,krig::Kriging,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
    #Default value
    k = 2.0
    dtol = 1e-3 * norm(ub-lb)
    for i = 1:maxiters
        new_sample = sample(num_new_samples,lb,ub,sample_type)
        evaluations = zeros(eltype(krig.x[1]), num_new_samples)
        for j = 1:num_new_samples
            evaluations[j] = krig(new_sample[j]) +
                             k*sqrt(std_error_at_point(krig,new_sample[j]))
        end

        new_addition = false
        min_add_x = zero(eltype(new_sample[1]));
        min_add_y = zero(eltype(krig.y[1]));
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
                deleteat!(evaluations,min_index)
                deleteat!(new_sample,min_index)

                if length(new_sample) == 0
                    println("Out of sampling points")
                    index = argmin(krig.y)
                    return (krig.x[index],krig.y[index])
                end
             else
                new_addition = true
                min_add_x = new_min_x
                min_add_y = new_min_y
            end
        end
        if min_add_y < 1e-6*(maximum(krig.y) - minimum(krig.y))
            return
        else
            min_add_y = obj(min_add_x) # I actually add the objc function at that point
            add_point!(krig,min_add_x,min_add_y)
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
function surrogate_optimize(obj::Function,::LCBS,lb,ub,krig::Kriging,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
    #Default value
    k = 2.0
    dtol = 1e-3 * norm(ub-lb)
    for i = 1:maxiters
        d = length(krig.x)
        new_sample = sample(num_new_samples,lb,ub,sample_type)
        evaluations = zeros(eltype(krig.x[1]),num_new_samples)
        for j = 1:num_new_samples
            evaluations[j] = krig(new_sample[j]) +
                             k*sqrt(std_error_at_point(krig,new_sample[j]))
        end

        new_addition = false
        min_add_x = Tuple{}
        min_add_y = zero(eltype(krig.y[1]));
        diff_x = zeros(eltype(krig.x[1]),d)
        while new_addition == false
            #find minimum
            new_min_y = minimum(evaluations)
            min_index = argmin(evaluations)
            new_min_x = new_sample[min_index]
            for l = 1:d
                diff_x[l] = norm(krig.x[l] .- new_min_x)
            end
            bit_x = diff_x .> dtol
            #new_min_x has to have some distance from krig.x
            if false in bit_x
                #The new_point is not actually that new, discard it!
                deleteat!(evaluations,min_index)
                deleteat!(new_sample,min_index)

                if length(new_sample) == 0
                    println("Out of sampling points")
                    index = argmin(krig.y)
                    return (krig.x[index],krig.y[index])
                end
             else
                new_addition = true
                min_add_x = new_min_x
                min_add_y = new_min_y
            end
        end
        if min_add_y < 1e-6*(maximum(krig.y) - minimum(krig.y))
            index = argmin(krig.y)
            return (krig.x[index],krig.y[index])
        else
            min_add_y = obj(min_add_x) # I actually add the objc function at that point
            add_point!(krig,Tuple(min_add_x),min_add_y)
        end
    end
end

"""
Expected improvement method 1D
"""
function surrogate_optimize(obj::Function,::EI,lb::Number,ub::Number,krig::Kriging,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
        dtol = 1e-3*norm(ub-lb)
        eps = 0.01
        for i = 1:maxiters
            new_sample = sample(num_new_samples,lb,ub,sample_type)
            f_max = maximum(krig.y)
            evaluations = zeros(eltype(krig.x[1]),num_new_samples)
            point_found = false
            new_x_max = zero(eltype(krig.x[1]))
            new_y_max = zero(eltype(krig.x[1]))
            while point_found == false
                for j = 1:length(new_sample)
                    std = std_error_at_point(krig,new_sample[j])
                    u = krig(new_sample[j])
                    if abs(std) > 1e-6
                        z = (u - f_max - eps)/std
                    else
                        z = 0
                    end
                    evaluations[j] = (u-f_max-eps)*cdf(Normal(),z) + std*pdf(Normal(),z)
                end
                index_max = argmax(evaluations)
                x_new = new_sample[index_max]
                y_new = maximum(evaluations)
                diff_x = abs.(krig.x .- x_new)
                bit_x = diff_x .> dtol
                #new_min_x has to have some distance from krig.x
                if false in bit_x
                    #The new_point is not actually that new, discard it!
                    deleteat!(evaluations,index_max)
                    deleteat!(new_sample,index_max)
                    if length(new_sample) == 0
                        println("Out of sampling points")
                        index = argmin(krig.y)
                        return (krig.x[index],krig.y[index])
                    end
                 else
                    point_found = true
                    new_x_max = x_new
                    new_y_max = y_new
                end
            end
            if new_y_max < 1e-6*norm(maximum(krig.y)-minimum(krig.y))
                index = argmin(krig.y)
                return (krig.x[index],krig.y[index])
            end
            add_point!(krig,new_x_max,obj(new_x_max))
        end
        println("Completed maximum number of iterations")
end

"""
This is an implementation of Expected Improvement (EI),
arguably the most popular acquisition function in Bayesian optimization.
Under a Gaussian process (GP) prior, the goal is to
maximize expected improvement:

``EI(x) := E[max(f_{best}-f(x),0)``


"""
function surrogate_optimize(obj::Function,::EI,lb,ub,krig::Kriging,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
        dtol = 1e-3*norm(ub-lb)
        eps = 0.01
        for i = 1:maxiters
            d = length(krig.x)
            new_sample = sample(num_new_samples,lb,ub,sample_type)
            f_max = maximum(krig.y)
            evaluations = zeros(eltype(krig.x[1]),num_new_samples)
            point_found = false
            new_x_max = zero(eltype(krig.x[1]))
            new_y_max = zero(eltype(krig.x[1]))
            diff_x = zeros(eltype(krig.x[1]),d)
            while point_found == false
                for j = 1:length(new_sample)
                    std = std_error_at_point(krig,new_sample[j])
                    u = krig(new_sample[j])
                    if abs(std) > 1e-6
                        z = (u - f_max - eps)/std
                    else
                        z = 0
                    end
                    evaluations[j] = (u-f_max-eps)*cdf(Normal(),z) + std*pdf(Normal(),z)
                end
                index_max = argmax(evaluations)
                x_new = new_sample[index_max]
                y_new = maximum(evaluations)
                for l = 1:d
                    diff_x[l] = norm(krig.x[l] .- x_new)
                end
                bit_x = diff_x .> dtol
                #new_min_x has to have some distance from krig.x
                if false in bit_x
                    #The new_point is not actually that new, discard it!
                    deleteat!(evaluations,index_max)
                    deleteat!(new_sample,index_max)
                    if length(new_sample) == 0
                        println("Out of sampling points")
                        index = argmin(krig.y)
                        return (krig.x[index],krig.y[index])
                    end
                 else
                    point_found = true
                    new_x_max = x_new
                    new_y_max = y_new
                end
            end
            if new_y_max < 1e-6*norm(maximum(krig.y)-minimum(krig.y))
                index = argmin(krig.y)
                return (krig.x[index],krig.y[index])
            end
            add_point!(krig,Tuple(new_x_max),obj(new_x_max))
        end
        println("Completed maximum number of iterations")
end



function adjust_step_size(sigma_n,sigma_min,C_success,t_success,C_fail,t_fail)
    if C_success >= t_success
        sigma_n = 2*sigma_n
        C_success = 0
    end
    if C_fail >= t_fail
        sigma_n = max(sigma_n/2,sigma_min)
        C_fail = 0
    end
    return sigma_n,C_success,C_fail
end

function select_evaluation_point_1D(new_points1,surr1::AbstractSurrogate,numb_iters,maxiters)
    v = [0.3,0.5,0.8,0.95]
    k = 4
    n = length(surr1.x)
    if mod(maxiters-1,4) != 0
        w_nR = v[mod(maxiters-1,4)]
    else
        w_nR = v[4]
    end
    w_nD = 1 - w_nR

    l = length(new_points1)
    evaluations1 = zeros(eltype(surr1.y[1]),l)

    for i = 1:l
        evaluations1[i] = surr1(new_points1[i])
    end
    s_max = maximum(evaluations1)
    s_min = minimum(evaluations1)
    V_nR = zeros(eltype(surr1.y[1]),l)
    for i = 1:l
        if abs(s_max-s_min) <= 10e-6
            V_nR[i] = 1.0
        else
            V_nR[i] = (evaluations1[i] - s_min)/(s_max-s_min)
        end
    end

    #Compute score V_nD
    V_nD = zeros(eltype(surr1.y[1]),l)
    delta_n_x = zeros(eltype(surr1.x[1]),l)
    delta = zeros(eltype(surr1.x[1]),n)
    for j = 1:l
        for i = 1:n
            delta[i] = norm(new_points1[j]-surr1.x[i])
        end
        delta_n_x[j] = minimum(delta)
    end
    delta_n_max = maximum(delta_n_x)
    delta_n_min = minimum(delta_n_x)
    for i = 1:l
        if abs(delta_n_max-delta_n_min) <= 10e-6
            V_nD[i] = 1.0
        else
            V_nD[i] = (delta_n_max - delta_n_x[i])/(delta_n_max-delta_n_min)
        end
    end

    #Compute weighted score
    W_n = w_nR*V_nR + w_nD*V_nD
    return new_points1[argmin(W_n)]
end
"""
surrogate_optimize(obj::Function,::DYCORS,lb::Number,ub::Number,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)

DYCORS optimization method in 1D, following closely: Combining radial basis function
surrogates and dynamic coordinate search in high-dimensional expensive black-box optimzation".
"""
function surrogate_optimize(obj::Function,::DYCORS,lb::Number,ub::Number,surr1::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
    x_best = argmin(surr1.y)
    y_best = minimum(surr1.y)
    sigma_n = 0.2*norm(ub-lb)
    d = length(lb)
    sigma_min = 0.2*(0.5)^6*norm(ub-lb)
    t_success = 3
    t_fail = max(d,5)
    C_success = 0
    C_fail = 0
    for k = 1:maxiters
        p_select = min(20/d,1)*(1-log(k))/log(maxiters-1)
        # In 1D I_perturb is always equal to one, no need to sample
        d = 1
        I_perturb = d
        new_points = zeros(eltype(surr1.x[1]),num_new_samples)
        for i = 1:num_new_samples
            new_points[i] = x_best + rand(Normal(0,sigma_n))
            while new_points[i] < lb || new_points[i] > ub
                if new_points[i] > ub
                    #reflection
                    new_points[i] = maximum(surr1.x) - norm(new_points[i] - maximum(surr1.x))
                end
                if new_points[i] < lb
                    #reflection
                    new_points[i] = minimum(surr1.x) + norm(new_points[i]-minimum(surr1.x))
                end
            end
        end

        x_new = select_evaluation_point_1D(new_points,surr1,k,maxiters)
        f_new = obj(x_new)

        if f_new < y_best
            C_success = C_success + 1
            C_fail = 0
        else
            C_fail = C_fail + 1
            C_success = 0
        end

        sigma_n,C_success,C_fail = adjust_step_size(sigma_n,sigma_min,C_success,t_success,C_fail,t_fail)

        if f_new < y_best
            x_best = x_new
            y_best = f_new
            add_point!(surr1,x_best,y_best)
        end
    end
    index = argmin(surr1.y)
    return (surr1.x[index],surr1.y[index])
end


function select_evaluation_point_ND(new_points,surrn::AbstractSurrogate,numb_iters,maxiters)
    v = [0.3,0.5,0.8,0.95]
    k = 4
    n = size(surrn.x,1)
    d = size(surrn.x,2)
    if mod(maxiters-1,4) != 0
        w_nR = v[mod(maxiters-1,4)]
    else
        w_nR = v[4]
    end
    w_nD = 1 - w_nR

    l = size(new_points,1)
    evaluations = zeros(eltype(surrn.y[1]),l)
    for i = 1:l
        evaluations[i] = surrn(Tuple(new_points[i,:]))
    end
    s_max = maximum(evaluations)
    s_min = minimum(evaluations)
    V_nR = zeros(eltype(surrn.y[1]),l)
    for i = 1:l
        if abs(s_max-s_min) <= 10e-6
            V_nR[i] = 1.0
        else
            V_nR[i] = (evaluations[i] - s_min)/(s_max-s_min)
        end
    end

    #Compute score V_nD
    V_nD = zeros(eltype(surrn.y[1]),l)
    delta_n_x = zeros(eltype(surrn.x[1]),l)
    delta = zeros(eltype(surrn.x[1]),n)
    for j = 1:l
        for i = 1:n
            delta[i] = norm(new_points[j,:]-collect(surrn.x[i]))
        end
        delta_n_x[j] = minimum(delta)
    end
    delta_n_max = maximum(delta_n_x)
    delta_n_min = minimum(delta_n_x)
    for i = 1:l
        if abs(delta_n_max-delta_n_min) <= 10e-6
            V_nD[i] = 1.0
        else
            V_nD[i] = (delta_n_max - delta_n_x[i])/(delta_n_max-delta_n_min)
        end
    end
    #Compute weighted score
    W_n = w_nR*V_nR + w_nD*V_nD
    return new_points[argmin(W_n),:]
end

"""
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
function surrogate_optimize(obj::Function,::DYCORS,lb,ub,surrn::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
    x_best = collect(surrn.x[argmin(surrn.y)])
    y_best = minimum(surrn.y)
    sigma_n = 0.2*norm(ub-lb)
    d = length(lb)
    sigma_min = 0.2*(0.5)^6*norm(ub-lb)
    t_success = 3
    t_fail = max(d,5)
    C_success = 0
    C_fail = 0
    for k = 1:maxiters
        p_select = min(20/d,1)*(1-log(k))/log(maxiters-1)
        new_points = zeros(eltype(surrn.x[1]),num_new_samples,d)
        for j = 1:num_new_samples
            w = sample(d,0,1,sample_type)
            I_perturb = w .< p_select
            if ~(true in I_perturb)
                val = rand(1:d)
                I_perturb = vcat(zeros(Int,val-1),1,zeros(Int,d-val))
            end
            I_perturb = Int.(I_perturb)
            for i = 1:d
                if I_perturb[i] == 1
                    new_points[j,i] = x_best[i] + rand(Normal(0,sigma_n))
                else
                    new_points[j,i] = x_best[i]
                end
            end
        end

        for i = 1:num_new_samples
            for j = 1:d
                while new_points[i,j] < lb[j] || new_points[i,j] > ub[j]
                    if new_points[i,j] > ub[j]
                        new_points[i,j] = maximum(surrn.x)[j] - norm(new_points[i,j] - maximum(surrn.x)[j])
                    end
                    if new_points[i,j] < lb[j]
                        new_points[i,j] = minimum(surrn.x)[j] + norm(new_points[i]-minimum(surrn.x)[j])
                    end
                end
            end
        end

        #ND version
        x_new = select_evaluation_point_ND(new_points,surrn,k,maxiters)
        f_new = obj(x_new)


        if f_new < y_best
            C_success = C_success + 1
            C_fail = 0
        else
            C_fail = C_fail + 1
            C_success = 0
        end

        sigma_n,C_success,C_fail = adjust_step_size(sigma_n,sigma_min,C_success,t_success,C_fail,t_fail)

        if f_new < y_best
            x_best = x_new
            y_best = f_new
            add_point!(surrn,Tuple(x_best),y_best)
        end
    end
    index = argmin(surrn.y)
    return (surrn.x[index],surrn.y[index])
end


function obj2_1D(value,points)
    min = +Inf
    my_p = filter(x->abs(x-value)>10^-6,points)
    for i = 1:length(my_p)
        new_val = norm(my_p[i]-value)
        if new_val < min
            min = new_val
        end
    end
    return min
end

function I_tier_ranking_1D(P,surrSOP::AbstractSurrogate)
    #obj1 = objective_function
    #obj2 = obj2_1D
    Fronts = Dict{Int,Array{eltype(surrSOP.x[1]),1}}()
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
                val2_p = obj2_1D(p,P)
                val1_q = surrSOP.y[q_index]
                val2_q = obj2_1D(q,P)
                p_dominates_q = (val1_p < val1_q || abs(val1_p-val1_q) <= 10^-5) &&
                                (val2_p < val2_q || abs(val2_p-val2_q) <= 10^-5) &&
                                ((val1_p < val1_q) || (val2_p < val2_q))

                q_dominates_p = (val1_p < val1_q || abs(val1_p-val1_q) < 10^-5) &&
                                (val2_p < val2_q || abs(val2_p-val2_q) < 10^-5) &&
                                ((val1_p < val1_q) || (val2_p < val2_q))
                if q_dominates_p
                    n_p += 1
                end
                k = k + 1
            end
        if n_p == 0
            # no individual dominates p
            push!(F,p)
        end
        j = j + 1
        end
        if length(F) > 0
            Fronts[i] = F
            P = setdiff(P,F)
            i = i + 1
        else
            return Fronts
        end
    end
    return F
end

function II_tier_ranking_1D(D::Dict,srg::AbstractSurrogate)
    for i = 1:length(D)
        pos = []
        yn = []
        for j = 1:length(D[i])
            push!(pos,findall(e->e==D[i][j],srg.x))
            push!(yn,srg.y[pos[j]])
        end
        D[i] = D[i][sortperm(D[i])]
    end
    return D
end

function Hypervolume_Pareto_improving(f1_new,f2_new,Pareto_set)
    if size(Pareto_set,1) == 1
        area_before = zero(eltype(f1_new))
    else
        my_p = Pareto_set
        #Area before
        v_ref = [maximum(Pareto_set[:,1]),maximum(Pareto_set[:,2])]
        my_p = vcat(my_p,v_ref)
        v = sortperm(my_p[:,2])
        my_p[:,1] = my_p[:,1][v]
        my_p[:,2] = my_p[:,2][v]
        area_before = zero(eltype(f1_new))
        for j = 1:length(v)-1
            area_before += (my_p[j+1,2]-my_p[j,2])*(v_ref[1]-my_p[j])
        end
    end
    #Area after
    Pareto_set = vcat(Pareto_set,[f1_new f2_new])
    v_ref = [maximum(Pareto_set[:,1]) maximum(Pareto_set[:,2])]
    Pareto_set = vcat(Pareto_set,v_ref)
    v = sortperm(Pareto_set[:,2])
    Pareto_set[:,1] = Pareto_set[:,1][v]
    Pareto_set[:,2] = Pareto_set[:,2][v]
    area_after = zero(eltype(f1_new))
    for j = 1:length(v)-1
        area_after += (Pareto_set[j+1,2]-Pareto_set[j,2])*(v_ref[1]-Pareto_set[j])
    end
    return area_after - area_before
end



"""
surrogate_optimize(obj::Function,::SOP,lb::Number,ub::Number,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)

SOP Surrogate optimization method, following closely the following papers:

    -SOP: parallel surrogate global optimization with Pareto center selection for computationally expensive single objective problems by Tipaluck Krityakierne
    - Multiobjective Optimization Using Evolutionary Algorithms by Kalyan Deb
#Suggested number of new_samples = min(500*d,5000)
"""
function surrogate_optimize(obj::Function,sop1::SOP,lb::Number,ub::Number,surrSOP::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
    d = length(lb)
    N_fail = 3
    N_tenure = 5
    tau = 10^-5
    num_P = sop1.p
    centers_global = surrSOP.x
    r_centers_global = 0.2*norm(ub-lb)*ones(length(surrSOP.x))
    N_failures_global = zeros(length(surrSOP.x))
    tabu = []
    N_tenures_tabu = []
    for k = 1:maxiters
        N_tenures_tabu .+= 1
        #deleting points that have been in tabu for too long
        del = N_tenures_tabu .> N_tenure

        if length(del) > 0
            for i = length(del)
                if del[i]
                    del[i] = i
                end
            end
            deleteat!(N_tenures_tabu,del)
            deleteat!(tabu,del)
        end


        ##### P CENTERS ######
        C = []

        #S(x) set of points already evaluated
        #Rank points in S with:
        #1) Non dominated sorting
        Fronts_I = I_tier_ranking_1D(centers_global,surrSOP)
        #2) Second tier ranking
        Fronts = II_tier_ranking_1D(Fronts_I,surrSOP)
        ranked_list = []
        for i = 1:length(Fronts)
            for j = 1:length(Fronts[i])
                push!(ranked_list,Fronts[i][j])
            end
        end
        ranked_list = eltype(surrSOP.x[1]).(ranked_list)

        centers_full = 0
        i = 1
        while i <= length(ranked_list) && centers_full == 0
            flag = 0
            for j = 1:length(ranked_list)
                for m = 1:length(tabu)
                    if abs(ranked_list[j]-tabu[m]) < tau
                        flag = 1
                    end
                end
                for l = 1:length(centers_global)
                    if abs(ranked_list[j]-centers_global[l]) < tau
                        flag = 1
                    end
                end
            end
            if flag == 1
                skip
            else
                push!(C,ranked_list[i])
                if length(C) == num_P
                    centers_full = 1
                end
            end
            i = i + 1
        end

        # I examined all the points in the ranked list but num_selected < num_p
        # I just iterate again using only radius rule
        if length(C) < num_P
            i = 1
            while i <= length(ranked_list) && centers_full == 0
                flag = 0
                for j = 1:length(ranked_list)
                    for m = 1:length(centers_global)
                        if abs(centers_global[j] - ranked_list[m]) < tau
                            flag = 1
                        end
                    end
                end
                if flag == 1
                    skip
                else
                    push!(C,ranked_list[i])
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
                push!(C,ranked_list[i])
                if length(C) == num_P
                    centers_full = 1
                end
                i = i + 1
            end
        end

        #Here I have selected C = [] containing the centers
        r_centers = 0.2*norm(ub-lb)*ones(num_P)
        N_failures = zeros(num_P)
        #2.3 Candidate search
        new_points = zeros(eltype(surrSOP.x[1]),num_P,2)
        for i = 1:num_P
            N_candidates = zeros(eltype(surrSOP.x[1]),num_new_samples)
            #Using phi(n) just like DYCORS, merit function = surrogate
            #Like in DYCORS, I_perturb = 1 always
            evaluations = zeros(eltype(surrSOP.y[1]),num_new_samples)
            for j = 1:num_new_samples
                a = lb - C[i]
                b = ub - C[i]
                N_candidates[j] = C[i] + rand(TruncatedNormal(0,r_centers[i],a,b))
                evaluations[j] = surrSOP(N_candidates[j])
            end
            x_best = N_candidates[argmin(evaluations)]
            y_best = minimum(evaluations)
            new_points[i,1] = x_best
            new_points[i,2] = y_best
        end

        #new_points[i] now contains:
        #[x_1,y_1; x_2,y_2,...,x_{num_new_samples},y_{num_new_samples}]


        #2.4 Adaptive learning and tabu archive
        for i=1:num_P
            if new_points[i,1] in centers_global
                r_centers[i] = r_centers_global[i]
                N_failures[i] = N_failures_global[i]
            end

            f_1 = new_points[i,1]
            f_2 = obj2_1D(f_1,surrSOP.x)

            l = length(Fronts[1])
            Pareto_set = zeros(eltype(surrSOP.x[1]),l,2)

            for j = 1:l
                val = obj2_1D(Fronts[1][j],surrSOP.x)
                Pareto_set[j,1] = Fronts[1][j]
                Pareto_set[j,2] = val
            end
            if (Hypervolume_Pareto_improving(f_1,f_2,Pareto_set)<tau)
                #failure
                r_centers[i] = r_centers[i]/2
                N_failures[i] += 1
                if N_failures[i] > N_fail
                    push!(tabu,C[i])
                    push!(N_tenures_tabu,0)
                end
            else
                #P_i is success
                #Adaptive_learning
                add_point!(surrSOP,new_points[i,1],new_points[i,2])
                push!(r_centers_global,r_centers[i])
                push!(N_failures_global,N_failures[i])
            end
        end
    end
    index = argmin(surrSOP.y)
    return (surrSOP.x[index],surrSOP.y[index])
end
