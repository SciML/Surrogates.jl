using Surrogates

#1D
obj = x -> sin(x) + sin(x)^2 + sin(x)^3
lb = 0.0
ub = 10.0
x = sample(5,lb,ub,LowDiscrepancySample(2))
y = obj.(x)
p = 3.5
InverseDistance = InverseDistanceSurrogate(x,y,p,lb,ub)
prediction = InverseDistance(5.0)
add_point!(InverseDistance,5.0,-0.91)
add_point!(InverseDistance,[5.1,5.2],[1.0,2.0])


#ND
lb = [0.0,0.0]
ub = [10.0,10.0]
n = 100
x = sample(n,lb,ub,SobolSample())
f = x -> x[1]*x[2]^2
y = f.(x)
p = 3.0
InverseDistance = InverseDistanceSurrogate(x,y,p,lb,ub)
prediction = InverseDistance((1.0,2.0))
add_point!(InverseDistance,(5.0,3.4),-0.91)
add_point!(InverseDistance,[(5.1,5.2),(5.3,6.7)],[1.0,2.0])

# Multi-output #98
f  = x -> [x^2, x]
lb = 1.0
ub = 10.0
x  = sample(5, lb, ub, SobolSample())
push!(x, 2.0)
y  = f.(x)
surrogate = InverseDistanceSurrogate(x, y, 1, lb, ub)
@test surrogate(2.0) ≈ [4, 2]
surrogate((0.0, 0.0))

f  = x -> [x[1], x[2]^2]
lb = [1.0, 2.0]
ub = [10.0, 8.5]
x  = sample(20, lb, ub, SobolSample())
push!(x, (1.0, 2.0))
y  = f.(x)
surrogate = InverseDistanceSurrogate(x, y, 1.4, lb, ub)
@test surrogate((1.0, 2.0)) ≈ [1, 4]
x_new = (2.0, 2.0)
y_new = f(x_new)
add_point!(surrogate, x_new, y_new)
@test surrogate(x_new) ≈ y_new
surrogate((0.0, 0.0))
