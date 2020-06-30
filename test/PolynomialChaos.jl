using Surrogates
using PolyChaos


#1D
n = 20
lb = 0.0
ub = 4.0
f = x-> 2*x
x = sample(n,lb,ub,SobolSample())
y = f.(x)

my_pce = PolynomialChaosSurrogate(x,y,lb,ub)
val = my_pce(2.0)
add_point!(my_pce,3.0,6.0)
my_pce_changed = PolynomialChaosSurrogate(x,y,lb,ub,op=Uniform01OrthoPoly(1))

#ND
n = 60
lb = [0.0,0.0]
ub = [5.0,5.0]
f = x-> x[1]*x[2]
x = sample(n,lb,ub,SobolSample())
y = f.(x)

my_pce = PolynomialChaosSurrogate(x,y,lb,ub)
val = my_pce((2.0,2.0))
add_point!(my_pce,(2.0,3.0),6.0)

op1 = Uniform01OrthoPoly(1)
op2 = Beta01OrthoPoly(2,2,1.2)
ops = [op1,op2]
multi_poly = MultiOrthoPoly(ops,min(1,2))
my_pce_changed = PolynomialChaosSurrogate(x,y,lb,ub,op = multi_poly)
