using Surrogates

#1D
lb = 0.0
ub = 5.0
n = 5
d = 1
sample(n,lb,ub,GridSample(0.1))
sample(n,lb,ub,UniformSample())
sample(n,lb,ub,SobolSample())
sample(n,lb,ub,LatinHypercubeSample())
sample(20,lb,ub,LowDiscrepancySample(10))
sample(5,d,NormalSample(0,1))
sample(5,d,CauchySample(0,1))

#ND
lb = [0.1,-0.5]
ub = [1.0,20.0]
n = 5
d = 2
s = sample(n,lb,ub,GridSample([0.1,0.5]))
sample(n,lb,ub,UniformSample())
sample(n,lb,ub,SobolSample())
sample(n,lb,ub,LatinHypercubeSample())
sample(n,lb,ub,LowDiscrepancySample([10,3]))
sample(n,d,NormalSample(0,1))
sample(n,d,CauchySample(0,1))
