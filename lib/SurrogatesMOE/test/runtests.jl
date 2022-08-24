using Surrogates
using SurrogatesMOE
using SurrogatesFlux
using Flux
using SurrogatesPolyChaos
using Test

# function discont_1D(x)
#     if x < 0.0
#         return -5.0
#     elseif x >= 0.0
#         return 5.0
#     end
# end

# lb = -1.0
# ub = 1.0
# x = sample(50, lb, ub, SobolSample())
# y = discont_1D.(x)

# RAD_1D = RadialBasis(x, y, lb, ub, rad = linearRadial(), scale_factor = 1.0, sparse = false)
# expert_types = [
#     RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0,
#                         sparse = false),
#     RadialBasisStructure(radial_function = cubicRadial(), scale_factor = 1.0,
#                         sparse = false),
# ]

# MOE_1D_RAD_RAD = MOE(x,y, expert_types)
# MOE_at0 = MOE_1D_RAD_RAD(0.0)
# RAD_at0 = RAD_1D(0.0)
# true_val = 5.0
# @test (abs(RAD_at0 - true_val) > abs(MOE_at0 - true_val))

# KRIG_1D = Kriging(x, y, lb, ub, p = 1.0, theta = 1.0)
# expert_types = [InverseDistanceStructure(p = 1.0),
#     KrigingStructure(p = 1.0, theta = 1.0)
#     ]
# MOE_1D_INV_KRIG = MOE(x,y, expert_types)
# MOE_at0 = MOE_1D_INV_KRIG(0.0)
# KRIG_at0 = KRIG_1D(0.0)
# true_val = 5.0
# @test (abs(KRIG_at0 - true_val) > abs(MOE_at0 - true_val))

function rmse(a,b)
    a = vec(a)
    b = vec(b)
    if(size(a)!=size(b))
        println("error in inputs")
        return
    end
    n = size(a,1)
    return sqrt(sum((a - b).^2)/n)
end

function discont_NDIM(x)
    if(x[1] >= 0.0 && x[2] >= 0.0)
        return sum(x.^2) + 5
    else
        return sum(x.^2) - 5
    end
end
lb = [-1.0, -1.0]
ub = [1.0, 1.0]
x = sample(200, lb, ub, UniformSample())
y = discont_NDIM.(x)
x_test = sample(10, lb, ub, GoldenSample())


expert_types = [InverseDistanceStructure(p = 1.0),
    RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0,
                        sparse = false)
    ]
moe_nd_inv_rad = MOE(x,y, expert_types, ndim=2)
moe_pred_vals = moe_nd_inv_rad.(x_test)
true_vals = discont_NDIM.(x_test)
moe_rmse = rmse(true_vals, moe_pred_vals)
rbf = RadialBasis(x, y, lb, ub)
rbf_pred_vals = rbf.(x_test)
rbf_rmse = rmse(true_vals, rbf_pred_vals)
@test (rbf_rmse > moe_rmse)


expert_types = [KrigingStructure(p = [1.0, 1.0], theta = [1.0, 1.0]), LinearStructure()]
MOE_ND_KRIG_LIN = MOE(x, y, expert_types, ndim=2)
MOE_pred_vals = MOE_ND_KRIG_LIN.(x_test)
KRIG = Kriging(x, y, lb, ub, p = [1.0, 1.0], theta = [1.0, 1.0])
krig_pred_vals = KRIG.(x_test)
krig_rmse = rmse(true_vals, krig_pred_vals)
moe_rmse = rmse(true_vals, MOE_pred_vals)
@test krig_rmse > moe_rmse