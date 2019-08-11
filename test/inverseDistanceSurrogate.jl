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
obj = x -> sin(x[1]) + sin(x[1])^2 + sin(x[2])^3
lb = [0.0,0.0]
ub = [10.0,10.0]
x = sample(5,lb,ub,SobolSample())
y = obj.(x)
p = 3.5
InverseDistance = InverseDistanceSurrogate(x,y,p,lb,ub)
prediction = InverseDistance((1.0,2.0))
add_point!(InverseDistance,(5.0,3.4),-0.91)
add_point!(InverseDistance,[(5.1,5.2),(5.3,6.7)],[1.0,2.0])
