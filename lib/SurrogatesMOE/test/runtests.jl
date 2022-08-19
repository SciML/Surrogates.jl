using Surrogates
using SurrogatesMOE
using SurrogatesFlux
using Flux
using SurrogatesPolyChaos
using Test

function discont_1D(x)
    if x < 0.0
        return -5.0
    elseif x >= 0.0
        return 5.0
    end
end

lb = -1.0
ub = 1.0
x = sample(50, lb, ub, SobolSample())
y = discont_1D.(x)

RAD_1D = RadialBasis(x, y, lb, ub, rad = linearRadial(), scale_factor = 1.0, sparse = false)
expert_types = [
    RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0,
                        sparse = false),
    RadialBasisStructure(radial_function = cubicRadial(), scale_factor = 1.0,
                        sparse = false),
]

MOE_1D_RAD_RAD = MOE(x,y, expert_types)
MOE_at0 = MOE_1D_RAD_RAD(0.0)
RAD_at0 = RAD_1D(0.0)
true_val = 5.0
@test (abs(RAD_at0 - true_val) > abs(MOE_at0 - true_val))

KRIG_1D = Kriging(x, y, lb, ub, p = 1.0, theta = 1.0)
expert_types = [InverseDistanceStructure(p = 1.0),
    KrigingStructure(p = 1.0, theta = 1.0)
    ]
MOE_1D_INV_KRIG = MOE(x,y, expert_types)
MOE_at0 = MOE_1D_INV_KRIG(0.0)
KRIG_at0 = KRIG_1D(0.0)
true_val = 5.0
@test (abs(KRIG_at0 - true_val) > abs(MOE_at0 - true_val))

function discont_NDIM(x)
    if(x[1] >= 0.0 && x[2] >= 0.0)
        return 5.0
    else
        return -5.0
    end
end
lb = [-1.0, -1.0]
ub = [1.0, 1.0]
x = sample(200, lb, ub, SobolSample())
y = discont_NDIM.(x)

expert_types = [InverseDistanceStructure(p = 1.0),
    RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0,
                        sparse = false)
    ]
MOE_ND_INV_RAD = MOE(x,y, expert_types, ndim=2)
at00 = MOE_ND_INV_RAD([0.0, 0.0])
