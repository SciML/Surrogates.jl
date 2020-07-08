using Surrogates


lb = 0.0
ub = 5.0
n = 20
x = sample(n,lb,ub,SobolSample())
f = x->2*x+x^2
y = f.(x)
my_ear1d = EarthSurrogate(x,y,lb,ub)
val = my_ear1d(3.0)
add_point!(my_ear1d,6.0,48.0)
