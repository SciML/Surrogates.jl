using Surrogates, BenchmarkTools
using StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

lb, ub = 0.0, 10.0
f(x) = log(x) * x + sin(x)
xs = lb .+ (ub - lb) .* rand(rng, 50)
ys = f.(xs)

# 2D surrogate
lb2, ub2 = [0.0, 0.0], [10.0, 10.0]
f2(x) = sin(x[1]) * cos(x[2])
xs2 = collect(
    zip(
        lb2[1] .+ (ub2[1] - lb2[1]) .* rand(rng, 40),
        lb2[2] .+ (ub2[2] - lb2[2]) .* rand(rng, 40)
    )
)
xs2 = [collect(t) for t in xs2]
ys2 = f2.(xs2)

# =============================================================================
# Surrogate construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()

SUITE["construct"]["radial_basis"] = @benchmarkable RadialBasis(
    $xs, $ys, $lb, $ub
)
SUITE["construct"]["kriging"] = @benchmarkable Kriging($xs, $ys, $lb, $ub)
SUITE["construct"]["linear"] = @benchmarkable LinearSurrogate($xs, $ys, $lb, $ub)
SUITE["construct"]["inverse_distance"] = @benchmarkable InverseDistanceSurrogate(
    $xs, $ys, $lb, $ub
)
SUITE["construct"]["wendland"] = @benchmarkable Wendland($xs, $ys, $lb, $ub)
SUITE["construct"]["lobachevsky"] = @benchmarkable LobachevskySurrogate(
    $xs, $ys, $lb, $ub
)
SUITE["construct"]["radial_2d"] = @benchmarkable RadialBasis(
    $xs2, $ys2, $lb2, $ub2
)

# =============================================================================
# Prediction
# =============================================================================

SUITE["predict"] = BenchmarkGroup()

rad = RadialBasis(xs, ys, lb, ub)
krig = Kriging(xs, ys, lb, ub)

SUITE["predict"]["radial"] = @benchmarkable $rad(3.5)
SUITE["predict"]["kriging"] = @benchmarkable $krig(3.5)

# =============================================================================
# update! and surrogate optimization
# =============================================================================

SUITE["opt"] = BenchmarkGroup()

SUITE["opt"]["update"] = @benchmarkable update!(s, 5.5, $(f(5.5))) setup = (
    s = RadialBasis(copy($xs), copy($ys), $lb, $ub)
)
SUITE["opt"]["surrogate_optimize"] = @benchmarkable surrogate_optimize(
    $f, SRBF(), $lb, $ub, s, SobolSample(); maxiters = 20
) setup = (s = RadialBasis(copy($xs), copy($ys), $lb, $ub))
