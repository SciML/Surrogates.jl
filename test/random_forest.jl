using Surrogates, XGBoost

#1D
obj_1D = x -> 3*x+1
x = [1.0,2.0,3.0,4.0,5.0]
y = obj_1D.(x)
a = 0.0
b = 10.0
num_round = 2
my_forest_1D = RandomForestSurrogate(x,y,a,b,num_round)
val = my_forest_1D(3.5)
add_point!(my_forest_1D,6.0,19.0)
add_point!(my_forest_1D,[7.0,8.0],obj_1D.([7.0,8.0]))

#ND
lb = [0.0,0.0,0.0]
ub = [10.0,10.0,10.0]
x = sample(5,lb,ub,SobolSample())
obj_ND = x -> x[1] * x[2]^2 * x[3]
y = obj_ND.(x)
my_forest_ND = RandomForestSurrogate(x,y,lb,ub,num_round)
val = my_forest_ND((1.0,1.0,1.0))
add_point!(my_forest_ND,(1.0,1.0,1.0),1.0)
add_point!(my_forest_ND,[(1.2,1.2,1.0),(1.5,1.5,1.0)],[1.728,3.375])
