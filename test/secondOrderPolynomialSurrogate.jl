using Surrogates

#1D
lb = 0.0
ub = 5.0
obj_1D = x -> log(x)*exp(x)
x = sample(5,lb,ub,SobolSample())
y = obj_1D.(x)
my_second_order_poly = SecondOrderPolynomialSurrogate(x,y,lb,ub)
val = my_second_order_poly(5.0)
add_point!(my_second_order_poly,5.0,238.86)
add_point!(my_second_order_poly,[6.0,7.0],[722.84,2133.94])

#=
#ND
lb = [0.0,0.0]
ub = [10.0,10.0]
obj_ND = x -> log(x[1])*exp(x[2])
x = sample(5,lb,ub,UniformSample())
y = obj_ND.(x)
my_second_order_poly = SecondOrderPolynomialSurrogate(x,y,lb,ub)
val = my_second_order_poly((5.0,7.0))
add_point!(my_second_order_poly,(5.0,7.0),1764.96)
add_point!(my_second_order_poly,[(1.5,1.5),(3.4,5.4)],[1.817,270.95])
=#
