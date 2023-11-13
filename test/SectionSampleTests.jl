"""
    Sobel-sample x+y in [0,10]x[0,10],
    then minimize it on Section([NaN,10.0]),
    and verify that the minimum is on x,y=(0,10)
    rather than in (0,0)
"""

using QuasiMonteCarlo
using Surrogates
using Test

lb = [0.0, 0.0, 0.0]
ub = [10.0, 10.0, 10.0]
x = Surrogates.sample(10, lb, ub, LatinHypercubeSample())
f = x -> x[1] + x[2] + x[3]
y = f.(x)
f([0, 0, 0]) == 0

f_hat = Kriging(x, y, lb, ub)

f_hat([0, 0, 0])

isapprox(f([0, 0, 0]), f_hat([0, 0, 0]))

""" The global minimum is at (0,0) """

(xy_min, f_hat_min) = surrogate_optimize(f,
                                         DYCORS(), lb, ub,
                                         f_hat,
                                         SobolSample())

isapprox(xy_min[1], 0.0, atol = 1e-1)

""" The minimum on the (0,10) section is around (0,10) """

section_sampler_z_is_10 = SectionSample([NaN64, NaN64, 10.0],
                                        Surrogates.RandomSample())

@test [3] == Surrogates.fixed_dimensions(section_sampler_z_is_10)
@test [1, 2] == Surrogates.free_dimensions(section_sampler_z_is_10)

Surrogates.sample(5, lb, ub, section_sampler_z_is_10)

(xy_min, f_hat_min) = surrogate_optimize(f,
                                         EI(), lb, ub,
                                         f_hat,
                                         section_sampler_z_is_10, maxiters = 1000)

isapprox(xy_min[1], 0.0, atol = 0.1)
isapprox(xy_min[2], 0.0, atol = 0.1)
isapprox(xy_min[3], 10.0, atol = 0.1)
