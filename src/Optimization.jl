using LinearAlgebra

function merit_function(point,w,surr::AbstractSurrogate,s_max,s_min,d_max,d_min,box_size)
    D_x = box_size+1
    for i = 1:length(surr.x)
        distance = norm(surr.x[i]-point)
        if distance < D_x
            D_x = distance
        end
    end
    return w*(surr(point) - s_min)/(s_max-s_min) + (1-w)*((d_max - D_x)/(d_max - d_min))
end

#1D version due to operations on lb and ub
function optimization(lb::Number,ub::Number,surr::AbstractSurrogate,maxiters::Int,sample_type::SamplingAlgorithm,num_new_samples::Int)
#Suggested by:
#https://www.mathworks.com/help/gads/surrogate-optimization-algorithm.html
    scale = 0.2
    success = 0
    failure = 0
    w_range = [0.3,0.5,0.7,0.95]
    box_size = lb-ub
    success = 0
    failures = 0
    for k = 1:maxiters
        for w in Iterators.cycle(w_range)

            #1) Sample near incumbent (the 2 fraction is arbitrary here)
            incumbent_value = minimum(surr.y)
            incumbent_x = surr.x[argmin(surr.y)]

            new_sample = sample(num_new_samples,
                            incumbent_x-scale*norm(incumbent_x-lb)/2,
                            incumbent_x+scale*norm(incumbent_x-ub)/2,sample_type)

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

            #4) Find minimum of merit function = adaptive point
            adaptive_point_x = new_sample[argmin(evaluation_of_merit_function)]

            #4) Evaluate objective function at adaptive point
            adaptive_point_y = surr(adaptive_point_x)

            #5) Update surrogate with (adaptive_point,objective(adaptive_point)
            add_point!(surr,adaptive_point_x,adaptive_point_y)

            #6) How to go on?
            if surr(adaptive_point_x)[1] < incumbent_value
                incumbent_x = adaptive_point_x
                incumbent_value = adaptive_point_y
                success += 1
            else
                failure += 1
            end

            if (success == 3 & failure == 0) || (success - failure == 0)
                scale = scale*2
                #check bounds cant go more than [a,b]
                if lb*scale < lb || ub*scale > ub
                    println("Exiting, searched the whole box")
                    return
                    println("QUI")
                end
                success = 0
                failure = 0
            end

            if (failure == 5 & success == 0) || (failure - success == 0)
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
