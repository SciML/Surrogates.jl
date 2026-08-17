using Base
using Test
using LinearAlgebra
using Surrogates
@testset "RadialBasis" begin
    @testset "1D" begin
        lb = 0.0
        ub = 4.0
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]

        my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial())
        @test my_rad(3.0) ≈ 6.0
        update!(my_rad, 4.0, 10.0)
        @test my_rad(3.0) ≈ 6.0
        update!(my_rad, [3.2, 3.3, 3.4], [8.0, 9.0, 10.0])
        @test my_rad(3.0) ≈ 6.0

        @test RadialBasis(x, y, lb, ub, rad = cubicRadial()) isa RadialBasis
        my_rad = RadialBasis(x, y, lb, ub, rad = multiquadricRadial())
        @test my_rad isa RadialBasis

        # Test that input dimension is properly checked for 1D radial surrogates
        @test_throws ArgumentError my_rad(Float64[])
        @test_throws ArgumentError my_rad((2.0, 3.0, 4.0))
    end

    @testset "ND" begin
        x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
        y = [4.0, 5.0, 6.0]
        lb = [0.0, 3.0, 6.0]
        ub = [4.0, 7.0, 10.0]

        @testset "construction" begin
            my_rad = RadialBasis(x, y, lb, ub)
            @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0
        end

        @testset "update! with a single point" begin
            my_rad = RadialBasis(
                copy(x), copy(y), lb, ub, rad = linearRadial(), scale_factor = 1.0
            )
            update!(my_rad, (9.0, 10.0, 11.0), 10.0)
            @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0
        end

        @testset "update! with several points" begin
            my_rad = RadialBasis(copy(x), copy(y), lb, ub)
            update!(my_rad, [(9.0, 10.0, 11.0), (12.0, 13.0, 14.0)], [10.0, 11.0])
            @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0
        end

        @testset "repeated update! and other kernels" begin
            lb2 = [0.0, 0.0, 0.0]
            ub2 = [10.0, 10.0, 10.0]
            my_rad_ND = RadialBasis(copy(x), copy(y), lb2, ub2)
            update!(my_rad_ND, (3.5, 4.5, 1.2), 18.9)
            update!(my_rad_ND, [(3.2, 1.2, 6.7), (3.4, 9.5, 7.4)], [25.72, 239.0])
            @test my_rad_ND isa RadialBasis

            # Degree-1 kernels need at least 4 unisolvent samples in 3D, so they
            # get their own well-spread sample rather than the three collinear
            # points above.
            xs = sample(12, lb2, ub2, SobolSample())
            ys = (v -> v[1] + v[2] * v[3]).(xs)
            @test RadialBasis(xs, ys, lb2, ub2, rad = cubicRadial()) isa RadialBasis
            my_rad_ND = RadialBasis(xs, ys, lb2, ub2, rad = multiquadricRadial())
            @test isfinite(my_rad_ND((1.0, 1.0, 1.0)))
        end

        @testset "samples must be unisolvent for the polynomial tail" begin
            lb2 = [0.0, 0.0, 0.0]
            ub2 = [10.0, 10.0, 10.0]
            # Three collinear points: the [1 x y z] block has rank 2, so a
            # degree-1 tail (4 columns) cannot be determined. Degree 0 needs one
            # column and is fine.
            @test RadialBasis(x, y, lb2, ub2, rad = linearRadial()) isa RadialBasis
            for kern in (cubicRadial(), multiquadricRadial(), thinplateRadial())
                @test_throws ArgumentError RadialBasis(x, y, lb2, ub2, rad = kern)
            end
            # Enough samples but still degenerate: 12 collinear points in 3D.
            collinear = [(t, 2t, 3t) for t in 1.0:12.0]
            @test_throws ArgumentError RadialBasis(
                collinear, first.(collinear), lb2, ub2, rad = cubicRadial()
            )
        end

        @testset "exact at a sample, many and few points" begin
            f = v -> v[1] * v[2]
            lb2 = [1.0, 2.0]
            ub2 = [10.0, 8.5]
            for n in (500, 5)
                xs = sample(n, lb2, ub2, SobolSample())
                push!(xs, (1.0, 2.0))
                ys = f.(xs)
                my_radial_basis = RadialBasis(xs, ys, lb2, ub2, rad = linearRadial())
                @test my_radial_basis((1.0, 2.0)) ≈ 2
            end
        end

        @testset "dimension checks" begin
            f = v -> v[1] * v[2]
            lb2 = [1.0, 2.0]
            ub2 = [10.0, 8.5]
            xs = sample(5, lb2, ub2, SobolSample())
            push!(xs, (1.0, 2.0))
            my_radial_basis = RadialBasis(xs, f.(xs), lb2, ub2, rad = linearRadial())
            @test_throws ArgumentError my_radial_basis((1.0,))
            @test_throws ArgumentError my_radial_basis((2.0, 3.0, 4.0))
        end
    end

    @testset "multi-output" begin
        @testset "1D" begin
            f = t -> [t^2, t]
            lb = 1.0
            ub = 10.0
            x = sample(5, lb, ub, SobolSample())
            push!(x, 2.0)
            y = f.(x)
            my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial())
            @test my_radial_basis(2.0) ≈ [4, 2]
        end

        @testset "ND" begin
            f = v -> [v[1] * v[2], v[1] + v[2]^2]
            lb = [1.0, 2.0]
            ub = [10.0, 8.5]
            x = sample(5, lb, ub, SobolSample())
            push!(x, (1.0, 2.0))
            y = f.(x)
            my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial())
            @test my_radial_basis((1.0, 2.0)) ≈ [2, 5]

            x_new = (2.0, 2.0)
            y_new = f(x_new)
            update!(my_radial_basis, x_new, y_new)
            @test my_radial_basis(x_new) ≈ y_new
        end
    end

    @testset "sparse construction" begin
        @testset "1D" begin
            lb = 0.0
            ub = 4.0
            x = [1.0, 2.0, 3.0]
            y = [4.0, 5.0, 6.0]
            my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), sparse = true)
            @test my_rad(3.0) ≈ 6.0
        end

        @testset "ND" begin
            x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
            y = [4.0, 5.0, 6.0]
            lb = [0.0, 3.0, 6.0]
            ub = [4.0, 7.0, 10.0]
            my_rad = RadialBasis(x, y, lb, ub, sparse = true)
            @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0
        end
    end

    @testset "multiquadricRadial with default scale_factor" begin
        lb = [0.0, 0.0, 0.0]
        ub = [3.0, 3.0, 3.0]
        g = v -> sqrt(v[1]^2 + v[2]^2 + v[3]^2)
        x = sample(100, lb, ub, SobolSample())
        y = g.(x)

        mq_rad = RadialBasis(x, y, lb, ub, rad = multiquadricRadial())
        # Interpolation is exact at the samples, and the fit is accurate away from
        # them. Judged by error over a held-out set rather than at one point: a
        # single query point rewards a lucky fit, and this target (a cone,
        # non-smooth at the origin) has an uneven error distribution.
        @test all(isapprox(mq_rad(x[i]), y[i], atol = 1.0e-10) for i in eachindex(x))
        q = sample(2000, lb, ub, HaltonSample())
        @test sqrt(sum((mq_rad(p) - g(p))^2 for p in q) / length(q)) < 0.05

        # The shape parameter changes the fit. It is deliberately not asserted to be
        # *worse*: with the polynomial tail in place c = 0.9 is in fact marginally
        # more accurate than the default on this target.
        mq_rad_09 = RadialBasis(x, y, lb, ub, rad = multiquadricRadial(0.9))
        @test !isapprox(mq_rad_09([2.0, 2.0, 1.0]), mq_rad([2.0, 2.0, 1.0]), atol = 1.0e-8)
        @test all(isapprox(mq_rad_09(x[i]), y[i], atol = 1.0e-10) for i in eachindex(x))
    end

    @testset "issue 316" begin
        x = sample(1024, [-0.45, -0.4, -0.9], [0.4, 0.55, 0.35], SobolSample())
        # Bounds written as row matrices, so the query point below is a 1x3 Matrix.
        lb = [-0.45 -0.4 -0.9]
        ub = [0.4 0.55 0.35]

        function mockvalues(inp)
            a, b, c = inp
            p1 = reverse(
                vec(
                    [1.09903695e+1 -1.015005e+1 -4.0662974e+1 -1.41834931e+1 1.00604784e+1 4.34951623e+0 -1.06519689e-1 -1.93335202e-3]
                )
            )
            p2 = vec([2.12791877 2.12791877 4.23881665 -1.05464575])
            f = evalpoly(c, p1)
            f += p2[1] * a^2 + p2[2] * b^2 + p2[3] * a^2 * b + p2[4] * a * b^2
            return f
        end

        y = mockvalues.(x)
        rbf = RadialBasis(x, y, lb, ub, rad = multiquadricRadial(1.788))
        test_point = (lb .+ ub) ./ 2
        @test isapprox(rbf(test_point), mockvalues(test_point), atol = 0.001)
    end

    @testset "regularization" begin
        @testset "1D" begin
            lb = 0.0
            ub = 4.0
            x = [1.0, 2.0, 3.0]
            y = [4.0, 5.0, 6.0]
            my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), regularization = 1.0e-12)
            @test my_rad(3.0) ≈ 6.0
            update!(my_rad, 4.0, 10.0)
            @test my_rad(3.0) ≈ 6.0
            update!(my_rad, [3.2, 3.3, 3.4], [8.0, 9.0, 10.0])
            @test my_rad(3.0) ≈ 6.0
        end

        @testset "ND" begin
            x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
            y = [4.0, 5.0, 6.0]
            lb = [0.0, 3.0, 6.0]
            ub = [4.0, 7.0, 10.0]
            my_rad = RadialBasis(x, y, lb, ub)
            @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0

            my_rad = RadialBasis(
                copy(x), copy(y), lb, ub, rad = linearRadial(),
                scale_factor = 1.0, regularization = 1.0e-12
            )
            update!(my_rad, (9.0, 10.0, 11.0), 10.0)
            @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0
        end

        # A repeated sample makes the interpolation matrix singular; regularization
        # is what makes the solve succeed.
        @testset "regularization fixes the SingularException, 1D" begin
            lb = 0.0
            ub = 4.0
            x = [1.0, 1.0, 2.0, 3.0]
            y = [4.0, 4.0, 5.0, 6.0]
            @test_throws LinearAlgebra.SingularException RadialBasis(
                x, y, lb, ub, rad = linearRadial(), regularization = 0
            )
            my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), regularization = 1.0e-12)
            @test my_rad(3.0) ≈ 6.0
            @test my_rad(1.0) ≈ 4.0
        end

        @testset "regularization fixes the SingularException, ND" begin
            lb = [0.0, 3.0, 6.0]
            ub = [4.0, 7.0, 10.0]
            x = [(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
            y = [4.0, 4.0, 5.0, 6.0]
            @test_throws LinearAlgebra.SingularException RadialBasis(
                x, y, lb, ub, rad = linearRadial(), regularization = 0
            )
            my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), regularization = 1.0e-12)
            @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0
            @test my_rad((4.0, 5.0, 6.0)) ≈ 5.0
        end
    end

    @testset "polynomial tail" begin
        # The kernels offered here are only conditionally positive definite, so the
        # radial block alone is indefinite (all four have phi(0) = 0, hence a zero
        # diagonal and eigenvalues summing to zero). The augmented system
        # [Φ P; Pᵀ 0] is what makes the fit well posed, and it is what lets the
        # surrogate reproduce polynomials of degree <= rad.q exactly.
        kernels = (
            ("linearRadial", linearRadial(), 0),
            ("cubicRadial", cubicRadial(), 1),
            ("multiquadricRadial", multiquadricRadial(), 1),
            ("thinplateRadial", thinplateRadial(), 2),
        )

        @testset "1D: $name" for (name, kern, q) in kernels
            x = [0.0, 0.4, 1.1, 1.7, 2.3, 3.0]
            @testset "exact at nodes" begin
                f = t -> sin(t) + t^2
                s = RadialBasis(x, f.(x), 0.0, 3.0, rad = kern)
                @test all(isapprox(s(x[i]), f(x[i]), atol = 1.0e-10) for i in eachindex(x))
            end
            @testset "reproduces polynomials of degree <= $q" begin
                for deg in 0:q
                    f = t -> sum(0.7 * t^k for k in 0:deg)
                    s = RadialBasis(x, f.(x), 0.0, 3.0, rad = kern)
                    for t in (0.7, 1.5, 2.6)
                        @test isapprox(s(t), f(t), atol = 1.0e-9)
                    end
                end
            end
            @testset "sparse path agrees with dense" begin
                f = t -> 2t + 5
                dense = RadialBasis(x, f.(x), 0.0, 3.0, rad = kern)
                sp = RadialBasis(x, f.(x), 0.0, 3.0, rad = kern, sparse = true)
                @test isapprox(sp(1.5), dense(1.5), atol = 1.0e-8)
            end
        end

        @testset "ND: $name" for (name, kern, q) in kernels
            lb = [0.0, 0.0]
            ub = [3.0, 3.0]
            x = sample(20, lb, ub, SobolSample())
            @testset "exact at nodes" begin
                f = v -> sin(v[1]) + v[2]^2
                s = RadialBasis(x, f.(x), lb, ub, rad = kern)
                @test all(isapprox(s(x[i]), f(x[i]), atol = 1.0e-9) for i in eachindex(x))
            end
            @testset "reproduces a constant" begin
                # Degree 0 is within reach of every kernel here; without the
                # polynomial tail a sum of radial bumps cannot be constant between
                # the nodes.
                s = RadialBasis(x, fill(5.0, length(x)), lb, ub, rad = kern)
                for p in ((0.7, 2.2), (1.3, 2.1), (2.6, 0.4))
                    @test isapprox(s(p), 5.0, atol = 1.0e-9)
                end
            end
            if q >= 1
                @testset "reproduces an affine function" begin
                    f = v -> 3.0 + 2 * v[1] - v[2]
                    s = RadialBasis(x, f.(x), lb, ub, rad = kern)
                    for p in ((0.7, 2.2), (1.3, 2.1), (2.6, 0.4))
                        @test isapprox(s(p), f(p), atol = 1.0e-8)
                    end
                end
            end
        end

        @testset "thinplateRadial works in ND" begin
            # The kernel must branch on the radius, not on `z`: `iszero(::Tuple)`
            # needs `zero(::Tuple)`, which does not exist.
            lb = [0.0, 0.0]
            ub = [3.0, 3.0]
            x = sample(12, lb, ub, SobolSample())
            f = v -> v[1] + v[2]
            s = RadialBasis(x, f.(x), lb, ub, rad = thinplateRadial())
            @test isfinite(s((1.3, 2.1)))
            # The origin of the kernel must be finite, not NaN from 0 * log(0)
            @test thinplateRadial().phi((0.0, 0.0)) == 0.0
            @test thinplateRadial().phi(0.0) == 0.0
        end
    end

    @testset "off-node predictions" begin
        # The tests above pin behaviour *at* the samples, where any interpolant is
        # exact by construction. These pin behaviour *between* them, which is what
        # the surrogate is actually used for.
        kernels = (
            ("linearRadial", linearRadial()),
            ("cubicRadial", cubicRadial()),
            ("multiquadricRadial", multiquadricRadial()),
            ("thinplateRadial", thinplateRadial()),
        )

        @testset "1D accuracy between the nodes" begin
            f = t -> sin(t) + t / 3
            # Query strictly between nodes, and never on one.
            q = range(0.02, 2.98, length = 401)
            # Tolerance calibrated from measured behaviour at N = 40, with margin;
            # the worst kernel (linearRadial) sits at ~7e-4.
            @testset "$name" for (name, kern) in kernels
                x = collect(range(0.0, 3.0, length = 40))
                s = RadialBasis(x, f.(x), 0.0, 3.0, rad = kern)
                @test maximum(abs(s(t) - f(t)) for t in q) < 3.0e-3
            end
        end

        @testset "1D error shrinks under node refinement" begin
            # A surrogate that only happened to match at the samples would not
            # improve off-node as the samples are refined.
            f = t -> sin(t) + t / 3
            q = range(0.02, 2.98, length = 401)
            @testset "$name" for (name, kern) in kernels
                errs = map((10, 20, 40)) do n
                    x = collect(range(0.0, 3.0, length = n))
                    s = RadialBasis(x, f.(x), 0.0, 3.0, rad = kern)
                    maximum(abs(s(t) - f(t)) for t in q)
                end
                @test errs[2] < errs[1]
                @test errs[3] < errs[2]
            end
        end

        @testset "ND accuracy between the nodes" begin
            g = v -> sin(v[1]) + v[2]^2 / 4
            lb = [0.0, 0.0]
            ub = [2.0, 2.0]
            x = sample(64, lb, ub, SobolSample())
            q = sample(400, lb, ub, HaltonSample())
            tols = Dict(
                "linearRadial" => (0.3, 0.05), "cubicRadial" => (0.05, 0.01),
                "multiquadricRadial" => (0.08, 0.01), "thinplateRadial" => (0.05, 0.01),
            )
            @testset "$name" for (name, kern) in kernels
                s = RadialBasis(x, g.(x), lb, ub, rad = kern)
                e = [abs(s(p) - g(p)) for p in q]
                maxtol, rmstol = tols[name]
                @test maximum(e) < maxtol
                @test sqrt(sum(abs2, e) / length(e)) < rmstol
            end
        end

        @testset "ND error shrinks under node refinement" begin
            g = v -> sin(v[1]) + v[2]^2 / 4
            lb = [0.0, 0.0]
            ub = [2.0, 2.0]
            q = sample(400, lb, ub, HaltonSample())
            errs = map((36, 100, 225)) do n
                x = sample(n, lb, ub, SobolSample())
                s = RadialBasis(x, g.(x), lb, ub, rad = cubicRadial())
                maximum(abs(s(p) - g(p)) for p in q)
            end
            @test errs[2] < errs[1]
            @test errs[3] < errs[2]
        end

        @testset "prediction is independent of how the point is written" begin
            # `Matrix(1×d) .- Tuple(d)` broadcasts to a d×d outer difference, so
            # a matrix-shaped query point must be flattened first. Nodes hide
            # this; only off-node queries expose it.
            g = v -> sin(v[1]) + v[2]^2 / 4
            lb = [0.0, 0.0]
            ub = [2.0, 2.0]
            x = sample(40, lb, ub, SobolSample())
            @testset "$name" for (name, kern) in kernels
                s = RadialBasis(x, g.(x), lb, ub, rad = kern)
                for p in ((0.31, 1.27), (1.44, 0.62), (1.93, 1.85))
                    ref = s(p)
                    @test s([p[1], p[2]]) ≈ ref
                    @test s([p[1] p[2]]) ≈ ref
                end
            end
        end

        @testset "interpolation is linear in the responses" begin
            # s[a*y1 + b*y2] == a*s[y1] + b*s[y2] away from the nodes: the
            # coefficients solve a linear system, so this must hold to round-off.
            x = collect(range(0.0, 3.0, length = 15))
            q = (0.31, 1.27, 2.44, 2.93)
            y1 = sin.(x)
            y2 = x .^ 2 .- 1
            @testset "$name" for (name, kern) in kernels
                s1 = RadialBasis(x, y1, 0.0, 3.0, rad = kern)
                s2 = RadialBasis(x, y2, 0.0, 3.0, rad = kern)
                sc = RadialBasis(x, 2.5 .* y1 .- 1.5 .* y2, 0.0, 3.0, rad = kern)
                for t in q
                    @test sc(t) ≈ 2.5 * s1(t) - 1.5 * s2(t) atol = 1.0e-10
                end
            end
        end

        @testset "shifting the responses shifts the prediction" begin
            # s[y + c] == s[y] + c off-node. This holds only because the polynomial
            # tail can represent a constant; without it the shift is absorbed into
            # the radial weights and leaks into the shape of the fit.
            x = collect(range(0.0, 3.0, length = 15))
            y = sin.(x)
            @testset "$name" for (name, kern) in kernels
                s = RadialBasis(x, y, 0.0, 3.0, rad = kern)
                shifted = RadialBasis(x, y .+ 7.0, 0.0, 3.0, rad = kern)
                for t in (0.31, 1.27, 2.44, 2.93)
                    @test shifted(t) ≈ s(t) + 7.0 atol = 1.0e-10
                end
            end
        end

        @testset "update! matches a fresh fit off-node" begin
            f = t -> sin(t) + t / 3
            x = collect(range(0.0, 3.0, length = 12))
            extra = [0.37, 1.62, 2.71]
            # The union is formed up front and each surrogate gets its own copy,
            # so both fits see exactly the same samples.
            allx = vcat(x, extra)
            @testset "$name" for (name, kern) in kernels
                s = RadialBasis(copy(x), f.(x), 0.0, 3.0, rad = kern)
                update!(s, copy(extra), f.(extra))
                @test length(s.x) == length(allx)
                fresh = RadialBasis(copy(allx), f.(allx), 0.0, 3.0, rad = kern)
                for t in (0.2, 0.9, 1.8, 2.5)
                    @test s(t) ≈ fresh(t) atol = 1.0e-8
                end
            end
        end

        @testset "sparse and dense agree off-node" begin
            f = t -> sin(t) + t / 3
            x = collect(range(0.0, 3.0, length = 25))
            q = range(0.05, 2.95, length = 60)
            @testset "$name" for (name, kern) in kernels
                dense = RadialBasis(x, f.(x), 0.0, 3.0, rad = kern)
                sp = RadialBasis(x, f.(x), 0.0, 3.0, rad = kern, sparse = true)
                @test maximum(abs(sp(t) - dense(t)) for t in q) < 1.0e-7
            end
        end

        @testset "multi-output equals independent scalar fits off-node" begin
            x = collect(range(0.0, 3.0, length = 15))
            y1 = sin.(x)
            y2 = x .^ 2 .- 1
            @testset "$name" for (name, kern) in kernels
                s1 = RadialBasis(x, y1, 0.0, 3.0, rad = kern)
                s2 = RadialBasis(x, y2, 0.0, 3.0, rad = kern)
                mo = RadialBasis(x, [[a, b] for (a, b) in zip(y1, y2)], 0.0, 3.0, rad = kern)
                for t in (0.31, 1.27, 2.44, 2.93)
                    @test mo(t)[1] ≈ s1(t) atol = 1.0e-10
                    @test mo(t)[2] ≈ s2(t) atol = 1.0e-10
                end
            end
        end
    end
end
