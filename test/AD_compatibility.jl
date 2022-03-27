using Surrogates
using AbstractGPs
using LinearAlgebra
using Flux
using Flux: @epochs
using Zygote
using PolyChaos
using Test
#using Zygote: @nograd
#=
#FORWARD
###### 1D ######
lb = 0.0
ub = 10.0
n = 5
x = sample(n,lb,ub,SobolSample())
f = x -> x^2
y = f.(x)

#Radials
my_rad = RadialBasis(x,y,lb,ub,x->norm(x),2)
g = x -> ForwardDiff.derivative(my_rad,x)
g(5.0)

#Kriging
p = 1.5
my_krig = Kriging(x,y,p)
g = x -> ForwardDiff.derivative(my_krig,x)
g(5.0)

#Linear Surrogate
my_linear = LinearSurrogate(x,y,lb,ub)
g = x -> ForwardDiff.derivative(my_linear,x)
g(5.0)

#Inverse distance
p = 1.4
my_inverse = InverseDistanceSurrogate(x,y,p,lb,ub)
g = x -> ForwardDiff.derivative(my_inverse,x)
g(5.0)

#Lobachevsky
n = 4
α = 2.4
my_loba = LobachevskySurrogate(x,y,α,n,lb,ub)
g = x -> ForwardDiff.derivative(my_loba,x)
g(5.0)

#Second order polynomial
my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
g = x -> ForwardDiff.derivative(my_second,x)
g(5.0)

###### ND ######
lb = [0.0,0.0]
ub = [10.0,10.0]
n = 5
x = sample(n,lb,ub,SobolSample())
f = x -> x[1]*x[2]
y = f.(x)

#Radials
my_rad = RadialBasis(x,y,[lb,ub],z->norm(z),2)
g = x -> ForwardDiff.gradient(my_rad,x)
g([2.0,5.0])

#Kriging
theta = [2.0,2.0]
p = [1.9,1.9]
my_krig = Kriging(x,y,p,theta)
g = x -> ForwardDiff.gradient(my_krig,x)
g([2.0,5.0])

#Linear Surrogate
my_linear = LinearSurrogate(x,y,lb,ub)
g = x -> ForwardDiff.gradient(my_linear,x)
g([2.0,5.0])

#Inverse Distance
p = 1.4
my_inverse = InverseDistanceSurrogate(x,y,p,lb,ub)
g = x -> ForwardDiff.gradient(my_inverse,x)
g([2.0,5.0])

#Lobachevsky
alpha = [1.4,1.4]
n = 4
my_loba_ND = LobachevskySurrogate(x,y,alpha,n,lb,ub)
g = x -> ForwardDiff.gradient(my_loba_ND,x)
g([2.0,5.0])

#Second order polynomial
my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
g = x -> ForwardDiff.gradient(my_second,x)
g([2.0,5.0])
=#

##############
### ZYGOTE ###
##############


############
#### 1D ####
############
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

#Radials
@testset "Radials 1D" begin
    my_rad = RadialBasis(x,y,lb,ub,rad = linearRadial)
    g = x -> my_rad'(x)
    g(5.0)
end

# #Kriging
@testset "Kriging 1D" begin
    my_p = 1.5
    my_krig = Kriging(x,y,lb,ub,p=my_p)
    g = x -> my_krig'(x)
    g(5.0)
end

# #Linear Surrogate
@testset "Linear Surrogate" begin
    my_linear = LinearSurrogate(x,y,lb,ub)
    g = x -> my_linear'(x)
    g(5.0)
end

#Inverse distance
@testset "Inverse Distance" begin
    my_p = 1.4
    my_inverse = InverseDistanceSurrogate(x,y,lb,ub,p=my_p)
    g = x -> my_inverse'(x)
    g(5.0)
end

#Second order polynomial
@testset "Second Order Polynomial" begin
    my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
    g = x -> my_second'(x)
    g(5.0)
end 

#Lobachevsky
@testset "Lobachevsky" begin
    n = 4
    α = 2.4
    my_loba = LobachevskySurrogate(x,y,lb,ub, alpha = α, n = 4)
    g = x -> my_loba'(x)
    g(0.0)
end

#NN
@testset "NN" begin
    my_model = Chain(Dense(1,1), first)
    my_loss(x, y) = Flux.mse(my_model(x), y)
    my_opt = Descent(0.01)
    n_echos = 1
    my_neural = NeuralSurrogate(x,y,lb,ub,model=my_model,loss=my_loss,opt=my_opt,n_echos=1)
    g = x->my_neural'(x)
    g(3.4)
end

#Wendland
@testset "Wendland" begin
    my_wend = Wendland(x,y,lb,ub)
    g = x -> my_wend'(x)
    g(3.0)
end

# MOE and VariableFidelity for free because they are Linear combinations
# of differentiable surrogates

# #Polynomialchaos
@testset "Polynomial Chaos" begin
    n = 50
    x = sample(n,lb,ub,SobolSample())
    y = f.(x)
    my_poli = PolynomialChaosSurrogate(x,y,lb,ub)
    g = x -> my_poli'(x)
    g(3.0)
end

# #Gek
@testset "GEK" begin
    n = 10
    lb = 0.0
    ub = 5.0
    x = sample(n,lb,ub,SobolSample())
    f = x-> x^2
    y1 = f.(x)
    der = x->2*x
    y2 = der.(x)
    y = vcat(y1,y2)
    my_gek = GEK(x,y,lb,ub)
    g = x-> my_gek'(x)
    g(3.0)
end

################
###### ND ######
################

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

#Radials
@testset "Radials ND" begin
    my_rad = RadialBasis(x,y,lb,ub,rad = linearRadial, scale_factor = 2.1)
    g = x -> Zygote.gradient(my_rad,x)
    g((2.0,5.0))
end

# Kriging
@testset "Kriging ND" begin
    my_theta = [2.0,2.0]
    my_p = [1.9,1.9]
    my_krig = Kriging(x,y,lb,ub,p=my_p,theta=my_theta)
    g = x -> Zygote.gradient(my_krig,x)
    g((2.0,5.0))
end

# Linear Surrogate
@testset "Linear Surrogate ND" begin
    my_linear = LinearSurrogate(x,y,lb,ub)
    g = x -> Zygote.gradient(my_linear,x)
    g((2.0,5.0))
end

#Inverse Distance
@testset "Inverse Distance ND" begin
    my_p = 1.4
    my_inverse = InverseDistanceSurrogate(x,y,lb,ub,p=my_p)
    g = x -> Zygote.gradient(my_inverse,x)
    g((2.0,5.0))
end


#Lobachevsky not working yet weird issue with Zygote @nograd
#=
Zygote.refresh()
alpha = [1.4,1.4]
n = 4
my_loba_ND = LobachevskySurrogate(x,y,alpha,n,lb,ub)
g = x -> Zygote.gradient(my_loba_ND,x)
g((2.0,5.0))
=#

# Second order polynomial mutating arrays
@testset "SecondOrderPolynomialSurrogate ND" begin
    my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
    g = x -> Zygote.gradient(my_second,x)
    g((2.0,5.0))
end 

#NN
@testset "NN ND" begin
    my_model = Chain(Dense(2,1), first)
    my_loss(x, y) = Flux.mse(my_model(x), y)
    my_opt = Descent(0.01)
    n_echos = 1
    my_neural = NeuralSurrogate(x,y,lb,ub,model=my_model,loss=my_loss,opt=my_opt,n_echos=1)
    g = x -> Zygote.gradient(my_neural, x)
    g((2.0,5.0))
end

#wendland
@testset "Wendland ND" begin
    my_wend_ND = Wendland(x,y,lb,ub)
    g = x -> Zygote.gradient(my_wend_ND,x)
    g((2.0,5.0))
end
# #MOE and VariableFidelity for free because they are Linear combinations
# #of differentiable surrogates


# #PolynomialChaos
@testset "Polynomial Chaos ND" begin
    n = 50
    lb = [0.0,0.0]
    ub = [10.0,10.0]
    x = sample(n,lb,ub,SobolSample())
    f = x -> x[1]*x[2]
    y = f.(x)
    my_poli_ND = PolynomialChaosSurrogate(x,y,lb,ub)
    g = x -> Zygote.gradient(my_poli_ND,x)
    g((1.0,1.0))

    n = 10
    d = 2
    lb = [0.0,0.0]
    ub = [5.0,5.0]
    x = sample(n,lb,ub,SobolSample())
    f = x -> x[1]^2 + x[2]^2
    y1 = f.(x)
    grad1 = x -> 2*x[1]
    grad2 = x -> 2*x[2]
    function create_grads(n,d,grad1,grad2,y)
        c = 0
        y2 = zeros(eltype(y[1]),n*d)
        for i = 1:n
            y2[i+c] = grad1(x[i])
            y2[i+c+1] = grad2(x[i])
            c = c+1
        end
        return y2
    end
    y2 = create_grads(n,d,grad1,grad2,y)
    y = vcat(y1,y2)
    my_gek_ND = GEK(x,y,lb,ub)
    g = x -> Zygote.gradient(my_gek_ND,x)
    g((2.0,5.0))
end

# ###### ND -> ND ######

lb = [0.0, 0.0]
ub = [10.0, 2.0]
n = 5
x = sample(n,lb,ub,SobolSample())
f = x -> [x[1]^2, x[2]]
y = f.(x)

#NN
@testset "NN ND -> ND" begin
    my_model = Chain(Dense(2,2))
    my_loss(x, y) = Flux.mse(my_model(x), y)
    my_opt = Descent(0.01)
    n_echos = 1
    my_neural = NeuralSurrogate(x,y,lb,ub,model=my_model,loss=my_loss,opt=my_opt,n_echos=1)
    Zygote.gradient(x -> sum(my_neural(x)), (2.0, 5.0))

    my_rad = RadialBasis(x,y,lb,ub,rad = linearRadial)
    Zygote.gradient(x -> sum(my_rad(x)), (2.0, 5.0))

    my_p = 1.4
    my_inverse = InverseDistanceSurrogate(x,y,lb,ub,p=my_p)
    my_inverse((2.0, 5.0))
    Zygote.gradient(x -> sum(my_inverse(x)), (2.0, 5.0))

    my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
    Zygote.gradient(x -> sum(my_second(x)), (2.0, 5.0))
end