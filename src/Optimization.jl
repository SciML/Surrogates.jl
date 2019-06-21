using Surrogates
using LinearAlgebra

function merit_function(point,w,krig::Kriging,s_max,s_min,d_max,d_min,box_size)
    D_x = box_size+1
    for i = 1:length(krig.x)
        distance = norm(krig.x[i]-point)
        if distance < D_x
            D_x = distance
        end
    end
    return w*(krig(x)[1] - s_min)/(s_max-s_min) + (1-w)*((d_max - D_x)/(d_max - d_min))
end

#1D version due to operations on lb and ub
function optimization(lb,ub,krig::Kriging,maxiters,sample::Function,num_new_samples)
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
            incumbent_value = minimum(krig.y)
            incumbent_x = x[argmin(krig.y)]

            new_sample = sample(num_new_samples,
                            incumbent_x-scale*norm(incumbent_x-a)/2,
                            incumbent_x+scale*norm(incumbent_x-b)/2)

            #2) Create  merit function
            s = zeros(eltype(krig.x[1]),num_new_samples)
            for j = 1:num_new_samples
                s[j] = krig(new_sample[j])[1]
            end
            s_max = maximum(s)
            s_min = minimum(s)

            d_min = box_size + 1
            d_max = 0.0
            for r = 1:length(krig.x)
                for c = 1:num_new_samples
                    distance_rc = norm(krig.x[r]-new_sample[c])
                    if distance_rc > d_max
                        d_max = distance_rc
                    end
                    if distance_rc < d_min
                        d_min = distance_rc
                    end
                end
            end

            #3) Evaluate merit function in the sampled points
            evaluation_of_merit_function = merit_function.(new_sample,w,krig,s_max,s_min,d_max,d_min,box_size)

            #4) Find minimum of merit function = adaptive point
            adaptive_point_x = new_sample[argmin(evaluation_of_merit_function)]

            #4) Evaluate objective function at adaptive point
            adaptive_point_y = krig(new_sample)[1]

            #5) Update surrogate with (adaptive_point,objective(adaptive_point)
            add_point!(krig,adaptive_point_x,adaptive_point_y)

            #6) How to go on?
            if krig(adaptive_point_x)[1] < incumbent_value
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

#Example
objective_function = x -> 2*x+1
x = [2.0,4.0,6.0]
y = [5.0,9.0,13.0]
p = 2
a = 2
b = 6
my_k = Kriging(x,y,p)
optimization(a,b,my_k,10,random_sample,10)
println(length(my_k.x))
