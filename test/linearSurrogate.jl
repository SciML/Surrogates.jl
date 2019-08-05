using Surrogates

#1D
x = [1.0,2.0,3.0]
y = [1.5,3.5,5.3]
lb = 0.0
ub = 7.0
my_linear_surr_1D = LinearSurrogate(x,y,lb,ub)
val = my_linear_surr_1D(5.0)
add_point!(my_linear_surr_1D,4.0,7.2)
add_point!(my_linear_surr_1D,[5.0,6.0],[8.3,9.7])

#ND
lb = [0.0,0.0]
ub = [10.0,10.0]
x = sample(5,lb,ub,SobolSample())
y = [4.0,5.0,6.0,7.0,8.0]
my_linear_ND = LinearSurrogate(x,y,lb,ub)
add_point!(my_linear_ND,(10.0,11.0),9.0)
add_point!(my_linear_ND,[(8.0,5.0),(9.0,9.5)],[4.0,5.0])
val = my_linear_ND((10.0,11.0))
