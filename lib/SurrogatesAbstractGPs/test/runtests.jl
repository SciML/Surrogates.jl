using SafeTestsets, Test
using Surrogates: sample, SobolSample

@safetestset "AbstractGPSurrogate" begin 

using Surrogates
using SurrogatesAbstractGPs
using AbstractGPs
using Zygote

@testset "1D -> 1D" begin 
    lb = 0.0
    ub = 3.0
    f = x -> log(x)*exp(x);
    x = sample(5,lb,ub,SobolSample())
    y = f.(x)
    agp1D = AbstractGPSurrogate(x,y, gp=GP(SqExponentialKernel()), Σy=0.05)
    x_new = 2.5
    y_actual = f.(x_new)
    y_predicted = agp1D([x_new])
    @test isapprox(y_predicted, y_actual, atol=0.1)
end 

@testset "add points 1D" begin
    lb = 0.0
    ub = 3.0
    f = x -> x^2;
    x_points = sample(5,lb,ub,SobolSample())
    y_points = f.(x_points)
    agp1D = AbstractGPSurrogate([x_points[1]],[y_points[1]], gp=GP(SqExponentialKernel()), Σy=0.05)
    x_new = 2.5
    y_actual = f.(x_new)
    for i in 2:length(x_points)
        add_point!(agp1D, x_points[i], y_points[i]);
    end
    y_predicted = agp1D([x_new]);
    @test isapprox(y_predicted, y_actual, atol=0.1)
end


@testset "2D -> 1D" begin
    lb = [0.0; 0.0]
    ub = [2.0; 2.0]
    log_exp_f = x -> log(x[1])*exp(x[2])
    x = sample(50,lb,ub,SobolSample())
    y = log_exp_f.(x)
    agp_2D = AbstractGPSurrogate(x, y)
    x_new_2D = (2.0, 1.0)
    y_actual = log_exp_f(x_new_2D)
    y_predicted = agp_2D(x_new_2D)
    @test isapprox(y_predicted, y_actual, atol=0.1)
end 

@testset "add points 2D" begin
    lb = [0.0;0.0]
    ub = [2.0;2.0]
    sphere = x -> x[1]^2 + x[2]^2
    x = sample(20, lb, ub, SobolSample())
    y = sphere.(x)
    agp_2D = AbstractGPSurrogate([x[1]],[y[1]])
    logpdf_vals = []
    push!(logpdf_vals, logpdf_surrogate(agp_2D))
    for i in 2:length(x)
        add_point!(agp_2D,x[i],y[i])
        push!(logpdf_vals, logpdf_surrogate(agp_2D))
    end
    @test first(logpdf_vals) < last(logpdf_vals) #as more points are added log marginal posterior predictive probability increases
end


@testset "check ND prediction" begin
    lb = [-1.0;-1.0;-1.0]
    ub = [1.0;1.0;1.0]
    f = x ->hypot(x...)
    x = sample(25,lb,ub,SobolSample())
    y = f.(x)
    agpND = AbstractGPSurrogate(x, y, gp = GP(SqExponentialKernel()), Σy = 0.05)
    x_new = (-0.8,0.8,0.8)
    @test agpND(x_new) ≈ f(x_new) atol=0.2
end

@testset "Optimization 1D" begin
    objective_function = x -> 2*x+1
    lb = 0.0
    ub = 6.0
    x = [2.0,4.0,6.0]
    y = [5.0,9.0,13.0]
    p = 2
    a = 2
    b = 6
    my_k_EI1 = AbstractGPSurrogate(x,y)
    surrogate_optimize(objective_function,EI(),a,b,my_k_EI1,UniformSample(),maxiters=200,num_new_samples=155)
end

@testset "Optimization ND" begin
    objective_function_ND = z -> 3*hypot(z...)+1
    x = [(1.2,3.0),(3.0,3.5),(5.2,5.7)]
    y = objective_function_ND.(x)
    theta = [2.0,2.0]
    lb = [1.0,1.0]
    ub = [6.0,6.0]
    my_k_E1N = AbstractGPSurrogate(x,y)
    surrogate_optimize(objective_function_ND,EI(),lb,ub,my_k_E1N,UniformSample())
end


@testset "check working of logpdf_surrogate 1D" begin
    lb = 0.0
    ub = 3.0
    f = x -> log(x)*exp(x);
    x = sample(5,lb,ub,SobolSample())
    y = f.(x)
    agp1D = AbstractGPSurrogate(x,y, gp=GP(SqExponentialKernel()), Σy=0.05)
    logpdf_surrogate(agp1D)
end

@testset "check working of logpdf_surrogate ND" begin
    lb = [0.0; 0.0]
    ub = [2.0; 2.0]
    f = x -> log(x[1])*exp(x[2]);
    x = sample(5,lb,ub,SobolSample())
    y = f.(x)
    agpND = AbstractGPSurrogate(x,y, gp=GP(SqExponentialKernel()), Σy=0.05)
    logpdf_surrogate(agpND)
end

lb = 0.0
ub = 3.0
n = 10
x = sample(n,lb,ub,SobolSample())
f = x -> x^2
y = f.(x)

#AbstractGP 1D
@testset "AbstractGP 1D" begin
    agp1D = AbstractGPSurrogate(x,y, gp=GP(SqExponentialKernel()), Σy=0.05)
    g = x -> agp1D'(x)
    g([2.0])
end

lb = [0.0,0.0]
ub = [10.0,10.0]
n = 5
x = sample(n,lb,ub,SobolSample())
f = x -> x[1]*x[2]
y = f.(x)

# AbstractGP ND
@testset "AbstractGPSurrogate ND" begin
    my_agp = AbstractGPSurrogate(x,y, gp=GP(SqExponentialKernel()), Σy=0.05)
    g = x ->Zygote.gradient(my_agp, x)
    #g([(2.0,5.0)])
    g((2.0,5.0))
 end

end
