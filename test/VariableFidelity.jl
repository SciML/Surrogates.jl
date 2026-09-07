using Surrogates
using Test

# The defining property of the model: it is `low_fid_surr + eps_surr`, where
# `eps_surr` interpolates `y_high - low_fid_surr(x_high)`. The sum must therefore
# reproduce the high-fidelity samples, whatever the low-fidelity surrogate does.
function high_fidelity_error(v)
    return maximum(abs(v.y[i] - v(v.x[i])) for i in 1:(v.num_high_fidel))
end

@testset "1D" begin
    lb, ub = 0.0, 10.0
    f = x -> x^2 + 3sin(x)
    x = sample(20, lb, ub, SobolSample())
    y = f.(x)

    v = VariableFidelitySurrogate(x, y, lb, ub)
    @test v.num_high_fidel == 10
    @test high_fidelity_error(v) < 1.0e-10
    @test isapprox(v(3.0), f(3.0), rtol = 0.15)
end

@testset "ND" begin
    lb, ub = [0.0, 0.0], [5.0, 5.0]
    f = x -> x[1] * x[2]
    x = sample(20, lb, ub, SobolSample())
    y = f.(x)

    v = VariableFidelitySurrogate(x, y, lb, ub)
    @test high_fidelity_error(v) < 1.0e-10
    @test isapprox(v((2.0, 2.0)), f((2.0, 2.0)), rtol = 0.25)
end

@testset "update! refits the residual surrogate" begin
    # `eps_surr` is fitted against `low_fid_surr`. Extending the low-fidelity
    # surrogate without refitting it leaves a correction for a surrogate that no
    # longer exists, and the sum stops reproducing the high-fidelity data — by a
    # factor of 1e13 on this problem before the refit was added.
    lb, ub = 0.0, 10.0
    f = x -> x^2 + 3sin(x)
    x = sample(20, lb, ub, SobolSample())
    y = f.(x)

    v = VariableFidelitySurrogate(x, y, lb, ub)
    @test high_fidelity_error(v) < 1.0e-10

    update!(v, 4.321, f(4.321))
    @test length(v.x) == 21
    @test high_fidelity_error(v) < 1.0e-10

    # A batch of new low-fidelity samples, and the ND path.
    lbn, ubn = [0.0, 0.0], [5.0, 5.0]
    g = p -> p[1] * p[2]
    xn = sample(20, lbn, ubn, SobolSample())
    yn = g.(xn)
    vn = VariableFidelitySurrogate(xn, yn, lbn, ubn)
    update!(vn, (3.0, 3.0), g((3.0, 3.0)))
    @test length(vn.x) == 21
    @test high_fidelity_error(vn) < 1.0e-10
end

@testset "update! leaves the caller's containers alone" begin
    lb, ub = 0.0, 10.0
    f = x -> 2x
    x = sample(10, lb, ub, SobolSample())
    y = f.(x)

    v = VariableFidelitySurrogate(x, y, lb, ub)
    update!(v, 3.21, f(3.21))
    @test length(x) == 10
    @test length(y) == 10
    @test length(v.x) == 11
end

@testset "every supported structure builds on both fidelity levels" begin
    lb, ub = 0.0, 10.0
    f = x -> x^2 + 3sin(x)
    x = sample(20, lb, ub, SobolSample())
    y = f.(x)

    structures = [
        RadialBasisStructure(
            radial_function = linearRadial(), scale_factor = 1.0, sparse = false
        ),
        KrigingStructure(p = 1.9, theta = 0.1),
        LinearStructure(),
        InverseDistanceStructure(p = 1.0),
        LobachevskyStructure(alpha = 2.0, n = 6, sparse = false),
        SecondOrderPolynomialStructure(),
        WendlandStructure(eps = 1.0, maxiters = 300, tol = 1.0e-6),
    ]

    for s in structures
        low = VariableFidelitySurrogate(x, y, lb, ub; low_fid_structure = s)
        @test isfinite(low(3.0))
        high = VariableFidelitySurrogate(x, y, lb, ub; high_fid_structure = s)
        @test isfinite(high(3.0))
    end
end

@testset "input validation" begin
    lb, ub = 0.0, 10.0
    f = x -> 2x
    x = sample(20, lb, ub, SobolSample())
    y = f.(x)

    # An unsupported name used to `throw` a bare `String`, which is not an
    # `Exception` and so could not be caught by type.
    @test_throws ArgumentError VariableFidelitySurrogate(
        x, y, lb, ub; low_fid_structure = (name = "NotASurrogate",)
    )
    @test_throws ArgumentError VariableFidelitySurrogate(
        x, y, lb, ub; high_fid_structure = (name = "NotASurrogate",)
    )

    # `GEK` needs `n(1 + d)` observations, values then gradients, and a
    # variable-fidelity design carries only function values.
    gek = GEKStructure(p = 2.0, theta = 0.5)
    @test_throws ArgumentError VariableFidelitySurrogate(
        x, y, lb, ub; low_fid_structure = gek
    )
    @test_throws ArgumentError VariableFidelitySurrogate(
        x, y, lb, ub; high_fid_structure = gek
    )

    # An empty side of the split reached the inner constructor as a `BoundsError`
    # naming neither the keyword nor which side fell short.
    @test_throws ArgumentError VariableFidelitySurrogate(
        x, y, lb, ub; num_high_fidel = length(x)
    )
    @test_throws ArgumentError VariableFidelitySurrogate(x, y, lb, ub; num_high_fidel = 0)
    @test_throws ArgumentError VariableFidelitySurrogate(x, y, lb, ub; num_high_fidel = -3)
end

@testset "mixed structures" begin
    lb, ub = 0.0, 10.0
    f = x -> x^2 + 3sin(x)
    x = sample(20, lb, ub, SobolSample())
    y = f.(x)

    v = VariableFidelitySurrogate(
        x, y, lb, ub; num_high_fidel = 8,
        low_fid_structure = InverseDistanceStructure(p = 1.0),
        high_fid_structure = RadialBasisStructure(
            radial_function = linearRadial(), scale_factor = 1.0, sparse = false
        )
    )
    @test v.num_high_fidel == 8
    @test high_fidelity_error(v) < 1.0e-10
end
