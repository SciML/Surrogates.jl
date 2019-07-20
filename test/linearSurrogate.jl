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
x = [1.0 2.0 3.0;4.0 5.0 6.0; 7.0 8.0 9.0; 3.5 4.7 5.8]
y = [4.0,5.0,6.0,7.0]
lb = [0.0,0.0,0.0]
ub = [7.5,10.0,30.0]
my_linear_ND = LinearSurrogate(x,y,lb,ub)
val = my_linear_ND([1.2 3.4 3.9])
add_point!(my_linear_ND,[4.0 8.0 12.0], 10.5)
add_point!(my_linear_ND,[4.0 8.0 12.0; 3.1 5.6 12.3], [10.5,9.4])
