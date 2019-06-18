using Surrogates
using LinearAlgebra

#Building the surrogate with linear radial basis
objective_function = x -> 2*x+1
x = [1.0,2.0,3.0]
y = [3.0,5.0,7.0]
incumbent = argmin(y)
a = 0
b = 4
phi = z -> norm(z)
q = 1
my_rad = RadialBasis(x,y,a,b,phi,q)
function optimization(lb,ub,rad::RadialBasis,maxiters,sample::Function)
#Suggested by:
#https://www.mathworks.com/help/gads/surrogate-optimization-algorithm.html
scale = 0.2
w_range = [0.2,0.5,0.7,0.95]
i = 1
box_size = lb-ub
num_new_samples = 10
success = 0
failures = 0
for k = 1:maxiters
    if i == 4
        i = 1
    end
    w = w_range[i]
    i = i + 1

    #1) Sample near incumbent
    incumbent_value = minimum(rad.y)
    incumbent_x = x[argmin(rad.y)]
    new_sample = sample(num_new_samples,incumbent_x-scale,incumbent_x+scale)

    #2) Create  merit function STILL NEED TO FIND D_MAX, D_X AND D_MIN
    s = zeros(eltype(rad.x[1]),num_new_samples)
    for j = 1:num_new_samples
        s[j] = rad(new_sample[j])
    end
    s_max = max(s)
    s_min = min(s)

    for r = 1:num_new_samples
        for c = 1:num

    merit_function =
    x -> w*(rad(x) - s_min)/(s_max-s_min) + (1-w)*((d_max - d_x)/d_max - d_min))

    #3) Evaluate merit function in sampled_points
    evaluation_of_merit_function = merit_function.(new_sample)

    #4) Find minimum of merit function = adaptive point
    adaptive_point_x = new_sample[argmin(evaluation_of_merit_function)]

    #4) Evaluate objective function at adaptive point
    adaptive_point_y = rad(new_sample)

    #5) Update surrogate with (adaptive_point,objective(adaptive_point)
    add_point!(rad,adaptive_point_x,adaptive_point_y)

    #6) How to go on?
    if rad(adaptive_point_x) < incumbent_value
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
            exit()
        end
        success = 0
        failure = 0
    end
    if (failure == 5 & success == 0) || (failure - success == 0)
        scale = scale/2
        #check bounds and go on only if > 1e-5*interval
        if scale < 1e-5
            println("Exiting, too narrow")
            exit()
        end
        sucess = 0
        failure = 0
    end

end

end

optimization(a,b,my_rad,10,random_sample)