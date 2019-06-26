using LinearAlgebra

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
SRBF(lb,ub,surr::AbstractSurrogate,
             maxiters::Int,
             sample_type::SamplingAlgorithm,num_new_samples::Int)
Finds minimum of objective function while sampling the AbstractSurrogate at
the same time.
"""
function SRBF(lb,ub,surr::AbstractSurrogate,maxiters::Int,sample_type::SamplingAlgorithm,num_new_samples::Int,obj::Function)
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
    @inbounds for k = 1:maxiters
        for w in Iterators.cycle(w_range)

            #1) Sample near incumbent (the 2 fraction is arbitrary here)
            incumbent_value = minimum(surr.y)
            incumbent_x = surr.x[argmin(surr.y)]

            new_lb = incumbent_x .- scale*norm(incumbent_x .-lb)
            new_ub = incumbent_x .+ scale*norm(incumbent_x .-lb)

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
                #check bounds
                @inbounds for q = 1:length(lb)
                    if lb[q]*scale < lb[q] || ub[q]*scale > ub[q]
                        println("Exiting, searched the whole box")
                        return
                    end
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
SRBF(lb::Number,ub::Number,surr::AbstractSurrogate,
             maxiters::Int,
             sample_type::SamplingAlgorithm,num_new_samples::Int)
Finds minimum of objective function while sampling the AbstractSurrogate at
the same time.
"""
function SRBF(lb::Number,ub::Number,surr::AbstractSurrogate,maxiters::Int,
                      sample_type::SamplingAlgorithm,num_new_samples::Int,obj::Function)
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
    @inbounds for k = 1:maxiters
        for w in Iterators.cycle(w_range)

            #1) Sample near incumbent (the 2 fraction is arbitrary here)
            incumbent_value = minimum(surr.y)
            incumbent_x = surr.x[argmin(surr.y)]

            new_lb = incumbent_x-scale*norm(incumbent_x-lb)/2
            new_ub = incumbent_x+scale*norm(incumbent_x-ub)/2
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
                if lb*scale < lb || ub*scale > ub
                    println("Exiting, searched the whole box")
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
LCBS(lb::Number,ub::Number,surr::Kriging,maxiters::Int,
      sample_type::SamplingAlgorithm,num_new_samples::Int,obj::Function))

Implementation of Lower Confidence Bound (LCB), goal is to minimize:
LCB(x) := E[x] - k * sqrt(Var[x]), default value of k = 2
https://pysot.readthedocs.io/en/latest/options.html#strategy
"""
function LCBS(lb::Number,ub::Number,krig::Kriging,maxiters::Int,
             sample_type::SamplingAlgorithm,num_new_samples::Int,obj::Function)

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
LCBS(lb,ub,surr::Kriging,maxiters::Int,
      sample_type::SamplingAlgorithm,num_new_samples::Int,obj::Function))

Implementation of Lower Confidence Bound (LCB), goal is to minimize:
LCB(x) := E[x] - k * sqrt(Var[x]), default value of k = 2
https://pysot.readthedocs.io/en/latest/options.html#strategy
"""
function LCBS(lb,ub,krig::Kriging,maxiters::Int,
             sample_type::SamplingAlgorithm,num_new_samples::Int,obj::Function)

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
