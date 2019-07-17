using Surrogates

#1D
x = [1.0,2.0,3.0]
y = [1.5,3.5,5.3]
lb = 0.0
ub = 7.0
my_linear_surr = LinearSurrogate(x,y,lb,ub)
val = my_linear_surr(5.0)
add_point!(my_linear_surr,4.0,7.2)
add_point!(my_linear_surr,[5.0,6.0],[8.3,9.7])

#ND
