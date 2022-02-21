#using Distributions 
using Surrogates
using ForwardDiff
# include("../src/Sampling.jl") 
# include("../src/GEKPLS.jl")


# lb = [0.0,0.0]
# ub = [5.0,5.5]
# x = sample(5,lb,ub,SobolSample())
# sphere = x -> x[1]^2+x[2]^2
# y = sphere.(x)
# theta0 = [0.01, 0.01]
# extra_points=1
# n_comp=2
# gradients=2

X = [[ 1.  2.  3.]
     [ 4.  5.  6.]
     [ 7.  8.  9.]
     [10. 11. 12.]]
y = zeros(size(X)[1], 1)
grads = zeros(size(X));
f(x) = x[1]^2 + x[2]^2 + x[3]^2;
g = x -> ForwardDiff.gradient(f, x);

for i in 1:size(X)[1]   
    y[i]=f(X[i,:])
    grads[i,:]=g(X[i,:])
end

n_comp = 2
delta_x = .0001
xlimits = [ -10.0  10.0;
            -10.0  10.0;
            -10.0  10.0]
extra_points = 2
theta0 = .001

println("calling on my_gekpls")
my_gekpls = GEKPLS(X,y, grads, n_comp, delta_x, xlimits, extra_points, theta0) 
my_gekpls(3.0)




#todo: write test for add_point