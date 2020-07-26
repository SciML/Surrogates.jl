using Surrogates

#1D
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
val = my_gek(2.0)
std_err = std_error_at_point(my_gek,1.0)
add_point!(my_gek,2.5,2.5^2)

#ND
