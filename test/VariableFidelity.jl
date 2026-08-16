using Surrogates
using Test

@testset "VariableFidelitySurrogate" begin
    @testset "structure named tuples expose name" begin
        @test LinearStructure().name == "LinearSurrogate"
        @test SecondOrderPolynomialStructure().name == "SecondOrderPolynomialSurrogate"
        @test KrigingStructure(p = 1.5, theta = 1.0).name == "Kriging"
        @test InverseDistanceStructure(p = 1.0).name == "InverseDistanceSurrogate"
        @test WendlandStructure(eps = 0.1, maxiters = 300, tol = 1.0e-6).name == "Wendland"
    end

    @testset "unknown structure name throws" begin
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 2.0, 3.0, 4.0]
        @test_throws ArgumentError VariableFidelitySurrogate(
            x, y, 0.0, 5.0, low_fid_structure = (name = "NoSuchSurrogate",)
        )
        @test_throws ArgumentError VariableFidelitySurrogate(
            x, y, 0.0, 5.0, high_fid_structure = (name = "NoSuchSurrogate",)
        )
    end

    @testset "1D with default structures" begin
        n = 10
        lb = 0.0
        ub = 10.0
        x = sample(n, lb, ub, SobolSample())
        f = x -> 2 * x
        y = f.(x)
        my_varfid = VariableFidelitySurrogate(x, y, lb, ub)
        # Both fidelities see the same noiseless linear function, so the
        # combined prediction must track it closely.
        @test isapprox(my_varfid(3.0), f(3.0), atol = 1.0e-1)
        update!(my_varfid, 3.0, f(3.0))
        @test isapprox(my_varfid(3.0), f(3.0), atol = 1.0e-1)
    end

    @testset "every supported structure builds (1D, both slots)" begin
        n = 16
        lb = 0.0
        ub = 10.0
        x = sample(n, lb, ub, SobolSample())
        f = x -> 3.0 + x^2 / 10
        y = f.(x)
        structures = [
            RadialBasisStructure(
                radial_function = linearRadial(), scale_factor = 1.0, sparse = false
            ),
            KrigingStructure(p = 1.5, theta = 0.1),
            LinearStructure(),
            InverseDistanceStructure(p = 1.4),
            LobachevskyStructure(alpha = 2.0, n = 4, sparse = false),
            SecondOrderPolynomialStructure(),
            WendlandStructure(eps = 0.1, maxiters = 300, tol = 1.0e-6),
        ]
        for low in structures, high in structures
            surr = VariableFidelitySurrogate(
                x, y, lb, ub, num_high_fidel = 8,
                low_fid_structure = low, high_fid_structure = high
            )
            @test isfinite(surr(3.3))
        end
    end

    @testset "reproduces the high-fidelity data" begin
        # The defining property of the additive bridge: with an interpolating
        # residual surrogate, low_fid + eps_surr must recover y at every
        # high-fidelity sample, whatever the low-fidelity model does.
        lb = 0.0
        ub = 10.0
        x = sample(16, lb, ub, SobolSample())
        f = x -> 3.0 + x^2 / 10
        y = f.(x)
        rb = RadialBasisStructure(
            radial_function = cubicRadial(), scale_factor = 1.0, sparse = false
        )
        for low in (rb, LinearStructure(), SecondOrderPolynomialStructure())
            surr = VariableFidelitySurrogate(
                x, y, lb, ub, num_high_fidel = 8,
                low_fid_structure = low, high_fid_structure = rb
            )
            @test all(isapprox(surr(x[i]), y[i], rtol = 1.0e-6) for i in 1:8)
        end
    end

    @testset "num_high_fidel is validated" begin
        lb = 0.0
        ub = 10.0
        x = sample(8, lb, ub, SobolSample())
        y = (v -> v^2).(x)
        # Out of range: no data would be left for one of the two fidelities.
        for bad in (0, -1, 8, 12)
            @test_throws ArgumentError VariableFidelitySurrogate(
                x, y, lb, ub, num_high_fidel = bad
            )
        end
        # In-range values pass the guard. Boundary values such as 1 or
        # length(x) - 1 clear it but may still be rejected downstream by the
        # inner surrogate's own minimum sample count.
        @test VariableFidelitySurrogate(x, y, lb, ub, num_high_fidel = 4) isa
            VariableFidelitySurrogate
    end

    @testset "update! keeps the high-fidelity agreement" begin
        # eps_surr is fitted to y_high - low_fid_surr(x_high). Refitting the
        # low-fidelity model without rebuilding eps_surr used to leave it stale
        # and silently break this invariant.
        lb = 0.0
        ub = 10.0
        x = sample(16, lb, ub, SobolSample())
        f = v -> 3.0 + v^2 / 10
        y = f.(x)
        rb = RadialBasisStructure(
            radial_function = cubicRadial(), scale_factor = 1.0, sparse = false
        )
        v = VariableFidelitySurrogate(
            x, y, lb, ub, num_high_fidel = 8,
            low_fid_structure = rb, high_fid_structure = rb
        )
        @test all(isapprox(v(x[i]), y[i], rtol = 1.0e-6) for i in 1:8)
        update!(v, 9.5, f(9.5))
        @test all(isapprox(v(x[i]), y[i], rtol = 1.0e-6) for i in 1:8)
        update!(v, [9.7, 9.9], f.([9.7, 9.9]))
        @test all(isapprox(v(x[i]), y[i], rtol = 1.0e-6) for i in 1:8)
        @test length(v.x) == 19
    end

    @testset "vector-valued responses" begin
        lb = 0.0
        ub = 10.0
        x = sample(16, lb, ub, SobolSample())
        f = v -> [3.0 + v^2 / 10, v]
        y = f.(x)
        rb = RadialBasisStructure(
            radial_function = cubicRadial(), scale_factor = 1.0, sparse = false
        )
        v = VariableFidelitySurrogate(
            x, y, lb, ub, num_high_fidel = 8,
            low_fid_structure = rb, high_fid_structure = rb
        )
        @test all(isapprox(v(x[i]), y[i], rtol = 1.0e-6) for i in 1:8)
    end

    @testset "ND with default structures" begin
        n = 10
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        x = sample(n, lb, ub, SobolSample())
        f = x -> x[1] * x[2]
        y = f.(x)
        my_varfidND = VariableFidelitySurrogate(x, y, lb, ub)
        @test isfinite(my_varfidND((2.0, 2.0)))
        update!(my_varfidND, (3.0, 3.0), f((3.0, 3.0)))
        @test isfinite(my_varfidND((2.0, 2.0)))
    end

    @testset "every supported structure builds (ND, both slots)" begin
        n = 20
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        x = sample(n, lb, ub, SobolSample())
        f = x -> x[1] * x[2] / 10 + 1.0
        y = f.(x)
        structures = [
            RadialBasisStructure(
                radial_function = linearRadial(), scale_factor = 1.0, sparse = false
            ),
            KrigingStructure(p = [1.5, 1.5], theta = [0.1, 0.1]),
            LinearStructure(),
            InverseDistanceStructure(p = 1.4),
            LobachevskyStructure(alpha = [1.0, 1.0], n = 4, sparse = false),
            SecondOrderPolynomialStructure(),
            WendlandStructure(eps = 0.1, maxiters = 300, tol = 1.0e-6),
        ]
        for low in structures, high in structures
            surr = VariableFidelitySurrogate(
                x, y, lb, ub, num_high_fidel = 10,
                low_fid_structure = low, high_fid_structure = high
            )
            @test isfinite(surr((3.3, 4.4)))
        end
    end
end
