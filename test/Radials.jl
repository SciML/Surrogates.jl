using Base
using Test
using LinearAlgebra
using Surrogates
using ForwardDiff
using Zygote

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

        my_rad = RadialBasis(x, y, lb, ub, rad = cubicRadial())
        my_rad = RadialBasis(x, y, lb, ub, rad = multiquadricRadial())

        @test_throws ArgumentError my_rad(Float64[])
        @test_throws ArgumentError my_rad((2.0, 3.0, 4.0))
    end

    @testset "ND" begin
        x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
        y = [4.0, 5.0, 6.0]
        lb = [0.0, 3.0, 6.0]
        ub = [4.0, 7.0, 10.0]

        my_rad = RadialBasis(x, y, lb, ub)
        @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0

        my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), scale_factor = 1.0)
        update!(my_rad, (9.0, 10.0, 11.0), 10.0)
        @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0

        my_rad = RadialBasis(x, y, lb, ub)
        update!(my_rad, [(9.0, 10.0, 11.0), (12.0, 13.0, 14.0)], [10.0, 11.0])
        @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0

        lb = [0.0, 0.0, 0.0]
        ub = [10.0, 10.0, 10.0]
        my_rad_ND = RadialBasis(x, y, lb, ub)
        update!(my_rad_ND, (3.5, 4.5, 1.2), 18.9)
        update!(my_rad_ND, [(3.2, 1.2, 6.7), (3.4, 9.5, 7.4)], [25.72, 239.0])
        my_rad_ND = RadialBasis(x, y, lb, ub, rad = cubicRadial())
        my_rad_ND = RadialBasis(x, y, lb, ub, rad = multiquadricRadial())
        prediction = my_rad_ND((1.0, 1.0, 1.0))

        f = x -> x[1] * x[2]
        lb = [1.0, 2.0]
        ub = [10.0, 8.5]
        x = sample(500, lb, ub, SobolSample())
        push!(x, (1.0, 2.0))
        y = f.(x)
        my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial())
        @test my_radial_basis((1.0, 2.0)) ≈ 2
        my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial())
        @test my_radial_basis((1.0, 2.0)) ≈ 2

        x = sample(5, lb, ub, SobolSample())
        push!(x, (1.0, 2.0))
        y = f.(x)
        my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial())
        @test my_radial_basis((1.0, 2.0)) ≈ 2

        @test_throws ArgumentError my_radial_basis((1.0,))
        @test_throws ArgumentError my_radial_basis((2.0, 3.0, 4.0))
    end

    @testset "multi-output" begin
        f = x -> [x^2, x]
        lb = 1.0
        ub = 10.0
        x = sample(5, lb, ub, SobolSample())
        push!(x, 2.0)
        y = f.(x)
        my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial())
        @test my_radial_basis(2.0) ≈ [4, 2]

        f = x -> [x[1] * x[2], x[1] + x[2]^2]
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

        # Responses may be tuples rather than vectors, and there is nothing
        # special about two outputs.
        pts = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0), (3.0, 3.0), (1.0, 3.0), (4.0, 0.5)]
        lb2 = [0.0, 0.0]
        ub2 = [4.0, 4.0]
        as_tuples = RadialBasis(pts, [(p[1], p[2]) for p in pts], lb2, ub2)
        as_vectors = RadialBasis(pts, [[p[1], p[2]] for p in pts], lb2, ub2)
        @test as_tuples((1.0, 2.0)) ≈ as_vectors((1.0, 2.0))
        three = RadialBasis(pts, [[p[1], p[2], p[1] + p[2]] for p in pts], lb2, ub2)
        @test length(three((1.0, 2.0))) == 3
        @test three((1.0, 2.0)) ≈ [1.0, 2.0, 3.0]

        # Integer responses promote the way the solve does.
        ints = RadialBasis(pts, [0, 1, 2, 3, 2, 1], lb2, ub2)
        @test ints((1.0, 2.0)) ≈ 1.0
    end

    @testset "sparse construction" begin
        lb = 0.0
        ub = 4.0
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), sparse = true)

        x_nd = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
        y_nd = [4.0, 5.0, 6.0]
        my_rad = RadialBasis(x_nd, y_nd, [0.0, 3.0, 6.0], [4.0, 7.0, 10.0], sparse = true)

        # A sparse assembly of a dense kernel is the same interpolant.
        dense = RadialBasis(x, y, lb, ub, rad = linearRadial())
        spars = RadialBasis(x, y, lb, ub, rad = linearRadial(), sparse = true)
        for p in (1.0, 1.7, 2.5, 3.0)
            @test dense(p) ≈ spars(p)
        end

        # The sparse assembly has to agree for the other kernels and for vector
        # responses too, not only for a scalar linear fit.
        pts = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0), (3.0, 3.0), (1.0, 3.0), (4.0, 0.5)]
        vals = [0.0, 1.0, 2.0, 3.0, 2.5, 1.2]
        lb2 = [0.0, 0.0]
        ub2 = [4.0, 4.0]
        for rad in (linearRadial(), cubicRadial(), thinplateRadial())
            d = RadialBasis(pts, vals, lb2, ub2; rad = rad, sparse = false)
            sp = RadialBasis(pts, vals, lb2, ub2; rad = rad, sparse = true)
            @test d((1.5, 2.5)) ≈ sp((1.5, 2.5))
        end
        multi = [[p[1], p[2]] for p in pts]
        d = RadialBasis(pts, multi, lb2, ub2; sparse = false)
        sp = RadialBasis(pts, multi, lb2, ub2; sparse = true)
        @test d((1.5, 2.5)) ≈ sp((1.5, 2.5))
    end

    @testset "multiquadric shape parameter is redundant with scale_factor" begin
        # `sqrt((c*r)^2 + 1)` is Hardy's `sqrt(r^2 + c0^2)` with `c0 = 1/c`,
        # times a constant the interpolant is invariant to. Both `c` and
        # `scale_factor` scale the radius, so they are one parameter.
        x = collect(range(0.0, 1.0, length = 9))
        y = sin.(3 .* x)
        a = RadialBasis(x, y, 0.0, 1.0; rad = multiquadricRadial(2.0), scale_factor = 1.0)
        b = RadialBasis(x, y, 0.0, 1.0; rad = multiquadricRadial(1.0), scale_factor = 0.5)
        for p in range(0.0, 1.0, length = 11)
            @test a(p) ≈ b(p)
        end
    end

    @testset "multiquadric shape parameter" begin
        lb = [0.0, 0.0, 0.0]
        ub = [3.0, 3.0, 3.0]
        g(x) = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
        x = sample(100, lb, ub, SobolSample())
        y = g.(x)
        mq_rad = RadialBasis(x, y, lb, ub, rad = multiquadricRadial())
        @test isapprox(mq_rad([2.0, 2.0, 1.0]), g([2.0, 2.0, 1.0]), atol = 0.0001)
        # A different shape parameter should not be as accurate.
        mq_rad = RadialBasis(x, y, lb, ub, rad = multiquadricRadial(0.9))
        @test !isapprox(mq_rad([2.0, 2.0, 1.0]), g([2.0, 2.0, 1.0]), atol = 0.0001)
    end

    @testset "issue 316: bounds given as row matrices" begin
        x = sample(1024, [-0.45, -0.4, -0.9], [0.4, 0.55, 0.35], SobolSample())
        lb = [-0.45 -0.4 -0.9]
        ub = [0.4 0.55 0.35]

        # Distinct names: inside a testset these would otherwise be assignments
        # to the captured `x` and `y` of the enclosing scope, not new locals.
        function mockvalues(in)
            xi, yi, zi = in
            p1 = reverse(
                vec(
                    [1.09903695e+1 -1.015005e+1 -4.0662974e+1 -1.41834931e+1 1.00604784e+1 4.34951623e+0 -1.06519689e-1 -1.93335202e-3]
                )
            )
            p2 = vec([2.12791877 2.12791877 4.23881665 -1.05464575])
            f = evalpoly(zi, p1)
            f += p2[1] * xi^2 + p2[2] * yi^2 + p2[3] * xi^2 * yi + p2[4] * xi * yi^2
            return f
        end

        y = mockvalues.(x)
        rbf = RadialBasis(x, y, lb, ub, rad = multiquadricRadial(1.788))
        test = (lb .+ ub) ./ 2
        @test isapprox(rbf(test), mockvalues(test), atol = 0.001)
    end

    @testset "regularization" begin
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

        x_nd = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
        y_nd = [4.0, 5.0, 6.0]
        lb_nd = [0.0, 3.0, 6.0]
        ub_nd = [4.0, 7.0, 10.0]
        my_rad = RadialBasis(x_nd, y_nd, lb_nd, ub_nd)
        @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0

        my_rad = RadialBasis(
            x_nd, y_nd, lb_nd, ub_nd, rad = linearRadial(), scale_factor = 1.0,
            regularization = 1.0e-12
        )
        update!(my_rad, (9.0, 10.0, 11.0), 10.0)
        @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0

        # A repeated sample makes the kernel matrix singular; the diagonal
        # regularization is what lets the solve go through.
        for reg in [0, 1.0e-12]
            local lb = 0.0
            local ub = 4.0
            local x = [1.0, 1.0, 2.0, 3.0]
            local y = [4.0, 4.0, 5.0, 6.0]
            if reg == 0
                @test_throws LinearAlgebra.SingularException RadialBasis(
                    x, y, lb, ub, rad = linearRadial(), regularization = reg
                )
            else
                local my_rad = RadialBasis(
                    x, y, lb, ub, rad = linearRadial(), regularization = reg
                )
                @test my_rad(3.0) ≈ 6.0
                @test my_rad(1.0) ≈ 4.0
            end
        end

        for reg in [0, 1.0e-12]
            local lb = [0.0, 3.0, 6.0]
            local ub = [4.0, 7.0, 10.0]
            local x = [
                (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0),
            ]
            local y = [4.0, 4.0, 5.0, 6.0]
            if reg == 0
                @test_throws LinearAlgebra.SingularException RadialBasis(
                    x, y, lb, ub, rad = linearRadial(), regularization = reg
                )
            else
                local my_rad = RadialBasis(
                    x, y, lb, ub, rad = linearRadial(), regularization = reg
                )
                @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0
                @test my_rad((4.0, 5.0, 6.0)) ≈ 5.0
            end
        end
    end

    @testset "every kernel interpolates its samples" begin
        # The kernels differ in their polynomial degree `q`, but all four are
        # solved as a plain interpolation system, so each must reproduce the
        # responses at the nodes it was built from.
        kernels = (
            "linear" => linearRadial(), "cubic" => cubicRadial(),
            "multiquadric" => multiquadricRadial(), "thinplate" => thinplateRadial(),
        )
        x = collect(0.0:1.0:5.0)
        y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
        for (name, rad) in kernels
            surr = RadialBasis(x, y, 0.0, 5.0; rad = rad)
            @test maximum(abs, [surr(p) - v for (p, v) in zip(x, y)]) < 1.0e-8
        end

        lb = [0.0, 0.0]
        ub = [4.0, 4.0]
        x_nd = sample(20, lb, ub, SobolSample())
        y_nd = [p[1] + p[2] for p in x_nd]
        for (name, rad) in kernels
            # thinplateRadial tested `iszero` on the sample difference itself,
            # which is a `MethodError` for a tuple: this kernel was unusable in
            # more than one dimension.
            surr = RadialBasis(x_nd, y_nd, lb, ub; rad = rad)
            @test surr((2.0, 2.0)) isa Number
            @test maximum(abs, [surr(p) - v for (p, v) in zip(x_nd, y_nd)]) < 1.0e-6
        end
    end

    @testset "thinplate at a node" begin
        # r^2 log(r) is 0 at r = 0 by continuation, and the kernel has to return
        # it rather than the NaN that log(0) produces.
        tp = thinplateRadial().phi
        @test tp(0.0) == 0.0
        @test isfinite(tp(0.0))
        @test tp((0.0, 0.0)) == 0.0
        @test tp((0.0, 0.0, 0.0)) == 0.0
        @test tp(2.0) ≈ 4 * log(2)
    end

    @testset "element types" begin
        # The interpolation matrix is built in `float(eltype)` of the samples,
        # and both `scale_factor` and `regularization` are taken to that type:
        # either left at its Float64 default would carry the whole solve into
        # Float64.
        for T in (Float64, Float32, BigFloat)
            x = T[0, 1, 2, 3]
            y = T[0, 1, 4, 9]
            surr = RadialBasis(x, y, T(0), T(3); rad = linearRadial())
            @test eltype(surr.coeff) == T
            @test surr(T(1.5)) isa T
            @test surr(T(2)) ≈ T(4)
        end
        # An integer design promotes the way `\` would, and no longer throws
        # `InexactError` for a scale the samples do not divide.
        surr = RadialBasis(
            [0, 1, 2, 3], [0, 1, 4, 9], 0, 3; rad = linearRadial(), scale_factor = 0.3
        )
        @test eltype(surr.coeff) == Float64
        @test surr(2) ≈ 4.0
        surr = RadialBasis(
            [0 // 1, 1 // 1, 2 // 1, 3 // 1], [0 // 1, 1 // 1, 4 // 1, 9 // 1],
            0 // 1, 3 // 1; rad = linearRadial()
        )
        @test surr(2 // 1) ≈ 4.0

        # `linearRadial` is evaluated through `_linear_distance` and every other
        # kernel through `rad.phi`, so a precision carried by one branch says
        # nothing about the other.
        for rad in (cubicRadial(), multiquadricRadial(), thinplateRadial())
            for T in (Float32, BigFloat)
                xs = T[0, 1, 2, 3, 4]
                ys = T[0, 1, 4, 9, 16]
                surr = RadialBasis(xs, ys, T(0), T(4); rad = rad)
                @test eltype(surr.coeff) == T
                @test surr(T(2.5)) isa T
                @test surr(T(3)) ≈ T(9) atol = sqrt(eps(T))
            end
        end
    end

    @testset "element types in more than one dimension" begin
        pts = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0), (3.0, 3.0), (1.0, 3.0), (4.0, 0.5)]
        vals = [0.0, 1.0, 2.0, 3.0, 2.5, 1.2]
        for T in (Float32, BigFloat)
            xs = [T.(p) for p in pts]
            ys = T.(vals)
            for rad in (linearRadial(), cubicRadial(), thinplateRadial())
                surr = RadialBasis(xs, ys, T.([0.0, 0.0]), T.([4.0, 4.0]); rad = rad)
                @test eltype(surr.coeff) == T
                @test surr(T.((1.0, 2.0))) isa T
                @test surr(T.((1.0, 2.0))) ≈ T(1) atol = sqrt(eps(T))
            end
        end
        # An integer design in more than one dimension promotes the same way.
        surr = RadialBasis(
            [(0, 0), (1, 2), (2, 1), (3, 3), (1, 3), (4, 1)], vals, [0, 0], [4, 4]
        )
        @test eltype(surr.coeff) == Float64
        @test surr((1, 2)) ≈ 1.0
    end

    @testset "input containers" begin
        pts = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0), (3.0, 3.0), (1.0, 3.0), (4.0, 0.5)]
        vals = [0.0, 1.0, 2.0, 3.0, 2.5, 1.2]
        lb = [0.0, 0.0]
        ub = [4.0, 4.0]
        # Samples as tuples or as vectors give the same surrogate.
        tuples = RadialBasis(pts, vals, lb, ub; rad = linearRadial())
        vectors = RadialBasis(collect.(pts), vals, lb, ub; rad = linearRadial())
        @test tuples((1.5, 2.5)) ≈ vectors((1.5, 2.5))

        # A point is a point however it is wrapped.
        @test tuples((1.5, 2.5)) ≈ tuples([1.5, 2.5])
        @test tuples((1.5, 2.5)) ≈ tuples([1.5 2.5])

        x1 = [0.0, 1.0, 2.0, 3.0]
        y1 = [0.0, 1.0, 4.0, 9.0]
        surr = RadialBasis(x1, y1, 0.0, 3.0; rad = linearRadial())
        @test surr(2.0) ≈ surr([2.0])
        @test surr(2.0) ≈ surr((2.0,))
        @test surr(2.0) ≈ surr(fill(2.0, 1, 1))
    end

    @testset "update! leaves the caller's containers alone" begin
        # `push!`/`append!` onto the surrogate's fields grew the very vectors
        # the caller passed in, so building a surrogate mutated its own inputs.
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        surr = RadialBasis(x, y, 0.0, 5.0; rad = linearRadial())
        update!(surr, 4.0, 10.0)
        @test x == [1.0, 2.0, 3.0]
        @test y == [4.0, 5.0, 6.0]
        @test length(surr.x) == 4
        @test surr(4.0) ≈ 10.0

        x_nd = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        y_nd = [1.0, 2.0, 3.0]
        surr = RadialBasis(x_nd, y_nd, [0.0, 0.0], [6.0, 7.0]; rad = linearRadial())
        update!(surr, [(2.0, 3.0), (4.0, 5.0)], [4.0, 5.0])
        @test length(x_nd) == 3
        @test length(surr.x) == 5
        @test surr((2.0, 3.0)) ≈ 4.0
    end
    @testset "queries of a different element type than the samples" begin
        # The accumulator holds coefficient times kernel value, so its type is
        # the promotion of the two. Taking it from the query alone made an
        # integer query throw InexactError on the first write.
        x = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0), (3.0, 3.0), (1.0, 0.0)]
        y = [0.0, 1.0, 2.0, 3.0, 1.5]
        surr = RadialBasis(x, y, [0.0, 0.0], [3.0, 3.0]; rad = linearRadial())
        @test surr((1, 2)) ≈ surr((1.0, 2.0))
        @test surr([1, 2]) ≈ surr((1.0, 2.0))

        x1 = [0.0, 1.0, 2.0, 3.0]
        y1 = [0.0, 1.0, 4.0, 9.0]
        surr1 = RadialBasis(x1, y1, 0.0, 3.0; rad = linearRadial())
        @test surr1(2) ≈ surr1(2.0)
        @test surr1(2) ≈ 4.0

        # An integer *design* promotes the same way.
        surri = RadialBasis([0, 1, 2, 3], [0, 1, 4, 9], 0, 3; rad = linearRadial())
        @test surri(2) ≈ 4.0
        @test surri(2.5) isa Float64
    end

    @testset "samples added without refitting are caught" begin
        # `update!` keeps the two containers in step; reaching past it does not.
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        surr = RadialBasis(x, y, 0.0, 5.0; rad = linearRadial())
        push!(surr.x, 4.0)
        err = try
            surr(2.0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("4 samples but only 3 coefficients", err.msg)
    end

    @testset "automatic differentiation" begin
        # Evaluation accumulates into a local for a scalar response and into a
        # Zygote buffer for a vector one, so both modes have to be exercised on
        # both paths.
        x1 = collect(0.0:0.5:10.0)
        y1 = x1 .^ 2
        # Off the nodes: every kernel is a function of `norm(z)`, whose
        # derivative at a zero difference is 0/0.
        query = 5.25
        for rad in (linearRadial(), cubicRadial(), multiquadricRadial(), thinplateRadial())
            surr = RadialBasis(x1, y1, 0.0, 10.0; rad = rad)
            fd = ForwardDiff.derivative(surr, query)
            @test fd isa Number
            @test Zygote.gradient(surr, query)[1] ≈ fd
        end

        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        xn = sample(40, lb, ub, SobolSample())
        surr = RadialBasis(xn, [p[1]^2 + p[2]^2 for p in xn], lb, ub; rad = cubicRadial())
        g = ForwardDiff.gradient(surr, [2.3, 3.1])
        @test g isa AbstractVector
        @test Zygote.gradient(surr, [2.3, 3.1])[1] ≈ g

        # Vector responses go through the buffered path.
        surr_multi = RadialBasis(
            xn, [[p[1]^2, p[2]] for p in xn], lb, ub; rad = linearRadial()
        )
        J = ForwardDiff.jacobian(p -> surr_multi(p), [2.3, 3.1])
        @test size(J) == (2, 2)
        @test Zygote.jacobian(p -> surr_multi(p), [2.3, 3.1])[1] ≈ J
    end
end
