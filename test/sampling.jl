using Surrogates

lb = 0.1
ub = 1.0
n = 100
s = sample(n,lb,ub,GridSample(0.1))
sample(n,lb,ub,UniformSample())
sample(n,lb,ub,SobolSample())
sample(n,lb,ub,LatinHypercubeSample())

f(x) = x^2
f.(s)

lb = [0.1,-0.5]
ub = [1.0,20.0]
n = 100
s = sample(n,lb,ub,GridSample([0.1,0.5]))
sample(n,lb,ub,UniformSample())
sample(n,lb,ub,SobolSample())
sample(n,lb,ub,LatinHypercubeSample())

f(x) = x[1]+x[2]^2
f.(s)
