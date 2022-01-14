using Surrogates
 
#1D
 
obj_1D = x-> 2*x+1
a = 0.0
b = 10.0
x = sample(50,a,b,SobolSample())
y = obj_1D.(x)
my_svm_1D = SVMSurrogate(x,y,a,b)
val = my_svm_1D(5.0)
@test isapprox(val, [11.0], atol=.5)
add_point!(my_svm_1D,3.1,7.2)
add_point!(my_svm_1D,[3.2,3.5],[7.4,8.0])
val2 = my_svm_1D(5.0)
@test isapprox(val, [11.0], atol=.5)
 
#ND
 
obj_N = x -> x[1]^2*x[2]
lb = [0.0,0.0]
ub = [10.0,10.0]
x = sample(200,lb,ub,SobolSample())
y = obj_N.(x)
my_svm_ND = SVMSurrogate(x,y,lb,ub)
val = my_svm_ND((5.0,1.2))
actual = obj_N((5.0,1.2))
@test isapprox(val, actual, atol=3)
add_point!(my_svm_ND,(1.0,1.0),1.0)
add_point!(my_svm_ND,[(1.2,1.2),(1.5, 1.5)],[1.728,3.375])
val1 = my_svm_ND((5.0,1.2))
actual1 = obj_N((5.0,1.2))
@test isapprox(val1, actual1, atol=3)