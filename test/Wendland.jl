using Surrogates


#1D
x = [1.0,2.0,3.0]
y = [3.0,5.0,7.0]
lb = 0.0
ub = 5.0
my_wend = Wendland(x,y,lb,ub)

add_point!(my_wend,0.5,4.0)
val = my_wend(0.5)


#ND
lb = [0.0,0.0]
ub = [4.0,4.0]
x = sample(5,lb,ub,SobolSample())
f = x -> x[1]+x[2]
y = f.(x)
my_wend_ND = Wendland(x,y,lb,ub)
est = my_wend_ND((1.0,2.0))
add_point!(my_wend_ND,(3.0,3.5),4.0)
add_point!(my_wend_ND,[(9.0,10.0),(12.0,13.0)],[10.0,11.0])
#todo
