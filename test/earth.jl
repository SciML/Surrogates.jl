using Surrogates

#1D
lb = 0.0
ub = 5.0
n = 20
x = sample(n, lb, ub, SobolSample())
f = x -> 2 * x + x^2
y = f.(x)
my_ear1d = EarthSurrogate(x, y, lb, ub)
val = my_ear1d(3.0)
add_point!(my_ear1d, 6.0, 48.0)

#ND
lb = [0.0, 0.0]
ub = [10.0, 10.0]
n = 30
x = sample(n, lb, ub, SobolSample())
f = x -> x[1] * x[2] + x[1]
y = f.(x)
my_earnd = EarthSurrogate(x, y, lb, ub)
val = my_earnd((2.0, 2.0))
add_point!(my_earnd, (2.0, 2.0), 6.0)
