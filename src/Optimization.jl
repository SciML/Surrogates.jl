using LinearAlgebra
using Distributions

abstract type SurrogateOptimizationAlgorithm end
struct SRBF <: SurrogateOptimizationAlgorithm end
struct LCBS <: SurrogateOptimizationAlgorithm end
struct EI <: SurrogateOptimizationAlgorithm end
struct DYCORS <: SurrogateOptimizationAlgorithm end

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
SRBF ND:
surrogate_optimize(obj::Function,::SRBF,lb,ub,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
"""
function surrogate_optimize(obj::Function,::SRBF,lb,ub,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
#Suggested by:
#https://www.mathworks.com/help/gads/surrogate-optimization-algorithm.html
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
    for w in Iterators.cycle(w_range)
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

            #2) Create  merit function
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
                        return
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
                    return
                end
                success = 0
                failure = 0
            end

            if failure == 5
                scale = scale/2
                #check bounds and go on only if > 1e-5*interval
                if scale < 1e-5
                    println("Exiting, too narrow")
                    return
                end
                sucess = 0
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
    for w in Iterators.cycle(w_range)
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
                        return
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
                    return
                end
                success = 0
                failure = 0
            end

            if failure == 5
                scale = scale/2
                #check bounds and go on only if > 1e-5*interval
                if scale < 1e-5
                    println("Exiting, too narrow")
                    return
                end
                sucess = 0
                failure = 0
            end
        end
    end
end

"""
LCBS 1D
Implementation of Lower Confidence Bound (LCB), goal is to minimize:
LCB(x) := E[x] - k * sqrt(Var[x]), default value of k = 2
https://pysot.readthedocs.io/en/latest/options.html#strategy
surrogate_optimize(obj::Function,::LCBS,lb::Number,ub::Number,krig::Kriging,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
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
                    return
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
LCBS ND
Implementation of Lower Confidence Bound (LCB), goal is to minimize:
LCB(x) := E[x] - k * sqrt(Var[x]), default value of k = 2
https://pysot.readthedocs.io/en/latest/options.html#strategy
surrogate_optimize(obj::Function,::LCBS,lb,ub,krig::Kriging,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
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
                    return
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
            add_point!(krig,Tuple(min_add_x),min_add_y)
        end
    end
end

"""
Expected improvement method 1D
surrogate_optimize(obj::Function,::EI,lb::Number,ub::Number,krig::Kriging,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
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
                        return
                    end
                 else
                    point_found = true
                    new_x_max = x_new
                    new_y_max = y_new
                end
            end
            if new_y_max < 1e-6*norm(maximum(krig.y)-minimum(krig.y))
                return
            end
            add_point!(krig,new_x_max,obj(new_x_max))
        end
end

"""
Expected improvement method ND
surrogate_optimize(obj::Function,::EI,lb,ub,krig::Kriging,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
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
                        return
                    end
                 else
                    point_found = true
                    new_x_max = x_new
                    new_y_max = y_new
                end
            end
            if new_y_max < 1e-6*norm(maximum(krig.y)-minimum(krig.y))
                return
            end
            add_point!(krig,Tuple(new_x_max),obj(new_x_max))
        end
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

function select_evaluation_point_1D(new_points,surr::AbstractSurrogate,numb_iters,maxiters)
    v = [0.3,0.5,0.8,0.95]
    k = 4
    n = length(surr.x)
    if mod(maxiters-1,4) != 0
        w_nR = v[mod(maxiters-1,4)]
    else
        w_nR = v[4]
    end
    w_nD = 1 - w_nR

    l = length(new_points)
    evaluations = zeros(eltype(surr.y[1]),l)

    for i = 1:l
        evaluations[i] = surr(new_points[i])
    end
    s_max = maximum(evaluations)
    s_min = minimum(evaluations)
    V_nR = zeros(eltype(surr.y[1]),l)
    for i = 1:l
        if abs(s_max-s_min) <= 10e-6
            V_nR[i] = 1.0
        else
            V_nR[i] = (evaluations[i] - s_min)/(s_max-s_min)
        end
    end

    #Compute score V_nD
    V_nD = zeros(eltype(surr.y[1]),l)
    delta_n_x = zeros(eltype(surr.x[1]),l)
    delta = zeros(eltype(surr.x[1]),n)
    for j = 1:l
        for i = 1:n
            delta[i] = norm(new_points[j]-surr.x[i])
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
    return new_points[argmin(W_n)]
end
"""
surrogate_optimize(obj::Function,::DYCORS,lb::Number,ub::Number,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)

DYCORS optimization method in 1D, following closely: Combining radial basis function
surrogates and dynamic coordinate search in high-dimensional expensive black-box optimzation".
"""
function surrogate_optimize(obj::Function,::DYCORS,lb::Number,ub::Number,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
    x_best = argmin(surr.y)
    y_best = minimum(surr.y)
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
        new_points = zeros(eltype(surr.x[1]),num_new_samples)
        for i = 1:num_new_samples
            new_points[i] = x_best + rand(Normal(0,sigma_n))
            while new_points[i] < lb || new_points[i] > ub
                if new_points[i] > ub
                    #reflection
                    new_points[i] = maximum(surr.x) - norm(new_points[i] - maximum(surr.x))
                end
                if new_points[i] < lb
                    #reflection
                    new_points[i] = minimum(surr.x) + norm(new_points[i]-minimum(surr.x))
                end
            end
        end

        x_new = select_evaluation_point_1D(new_points,surr,k,maxiters)
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
            add_point!(surr,x_best,y_best)
        end
    end
end


function select_evaluation_point_ND(new_points,surr::AbstractSurrogate,numb_iters,maxiters)
    v = [0.3,0.5,0.8,0.95]
    k = 4
    n = size(surr.x,1)
    d = size(surr.x,2)
    if mod(maxiters-1,4) != 0
        w_nR = v[mod(maxiters-1,4)]
    else
        w_nR = v[4]
    end
    w_nD = 1 - w_nR

    l = size(new_points,1)
    evaluations = zeros(eltype(surr.y[1]),l)
    for i = 1:l
        evaluations[i] = surr(Tuple(new_points[i,:]))
    end
    s_max = maximum(evaluations)
    s_min = minimum(evaluations)
    V_nR = zeros(eltype(surr.y[1]),l)
    for i = 1:l
        if abs(s_max-s_min) <= 10e-6
            V_nR[i] = 1.0
        else
            V_nR[i] = (evaluations[i] - s_min)/(s_max-s_min)
        end
    end

    #Compute score V_nD
    V_nD = zeros(eltype(surr.y[1]),l)
    delta_n_x = zeros(eltype(surr.x[1]),l)
    delta = zeros(eltype(surr.x[1]),n)
    for j = 1:l
        for i = 1:n
            delta[i] = norm(new_points[j,:]-collect(surr.x[i]))
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
surrogate_optimize(obj::Function,::DYCORS,lb,ub,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)

DYCORS optimization method in ND, following closely: Combining radial basis function
surrogates and dynamic coordinate search in high-dimensional expensive black-box optimzation".
"""
function surrogate_optimize(obj::Function,::DYCORS,lb,ub,surr::AbstractSurrogate,sample_type::SamplingAlgorithm;maxiters=100,num_new_samples=100)
    x_best = collect(surr.x[argmin(surr.y)])
    y_best = minimum(surr.y)
    sigma_n = 0.2*norm(ub-lb)
    d = length(lb)
    sigma_min = 0.2*(0.5)^6*norm(ub-lb)
    t_success = 3
    t_fail = max(d,5)
    C_success = 0
    C_fail = 0
    for k = 1:maxiters
        p_select = min(20/d,1)*(1-log(k))/log(maxiters-1)
        new_points = zeros(eltype(surr.x[1]),num_new_samples,d)
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
                        new_points[i,j] = maximum(surr.x)[j] - norm(new_points[i,j] - maximum(surr.x)[j])
                    end
                    if new_points[i,j] < lb[j]
                        new_points[i,j] = minimum(surr.x)[j] + norm(new_points[i]-minimum(surr.x)[j])
                    end
                end
            end
        end

        #ND version
        x_new = select_evaluation_point_ND(new_points,surr,k,maxiters)
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
            add_point!(surr,Tuple(x_best),y_best)
        end
    end
end
