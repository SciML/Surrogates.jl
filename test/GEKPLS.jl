# using Distributions #remove later
# include("../src/Sampling.jl") #remove later
# include("../src/GEKPLS.jl")#remove later
# include("../src/Surrogates.jl")

lb = [0.0,0.0]
ub = [5.0,5.5]
x = sample(5,lb,ub,SobolSample())
sphere = x -> x[1]^2+x[2]^2
y = sphere.(x)
theta0 = [0.01, 0.01]
extra_points=1
n_comp=2

my_gekpls = GEKPLS(x,y,lb,ub,theta0, extra_points, n_comp) #vik
my_gekpls(3.0)




#todo: write test for add_point