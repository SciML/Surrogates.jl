using PrecompileTools

@setup_workload begin
    # Setup minimal test data for 1D surrogates
    x_1d = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_1d = [2.1, 4.0, 5.9, 8.0, 10.2]
    lb_1d = 0.0
    ub_1d = 6.0

    # Setup minimal test data for nD surrogates (2D)
    x_nd = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    y_nd = [1.0, 2.0, 3.0, 4.0]
    lb_nd = (0.0, 0.0)
    ub_nd = (10.0, 10.0)

    # GEK test data (1D) - includes gradients
    y_gek_1d = vcat(y_1d, 2.0 .* x_1d)

    @compile_workload begin
        # RadialBasis - most used surrogate (1D)
        rad_1d = RadialBasis(x_1d, y_1d, lb_1d, ub_1d)
        rad_1d(2.5)

        # RadialBasis - nD
        rad_nd = RadialBasis(x_nd, y_nd, lb_nd, ub_nd)
        rad_nd((2.5, 3.5))

        # Kriging - commonly used (1D)
        krig_1d = Kriging(x_1d, y_1d, lb_1d, ub_1d)
        krig_1d(2.5)
        std_error_at_point(krig_1d, 2.5)

        # Kriging - nD
        krig_nd = Kriging(x_nd, y_nd, lb_nd, ub_nd)
        krig_nd((2.5, 3.5))
        std_error_at_point(krig_nd, (2.5, 3.5))

        # LinearSurrogate (1D)
        lin_1d = LinearSurrogate(x_1d, y_1d, lb_1d, ub_1d)
        lin_1d(2.5)

        # LinearSurrogate (nD)
        lin_nd = LinearSurrogate(x_nd, y_nd, lb_nd, ub_nd)
        lin_nd((2.5, 3.5))

        # LobachevskySurrogate (1D)
        lob_1d = LobachevskySurrogate(x_1d, y_1d, lb_1d, ub_1d)
        lob_1d(2.5)

        # SecondOrderPolynomialSurrogate (1D)
        sop_1d = SecondOrderPolynomialSurrogate(x_1d, y_1d, lb_1d, ub_1d)
        sop_1d(2.5)

        # InverseDistanceSurrogate (1D)
        ids_1d = InverseDistanceSurrogate(x_1d, y_1d, lb_1d, ub_1d)
        ids_1d(2.5)

        # Wendland (1D)
        wend_1d = Wendland(x_1d, y_1d, lb_1d, ub_1d)
        wend_1d(2.5)

        # GEK - Gradient Enhanced Kriging (1D)
        gek_1d = GEK(x_1d, y_gek_1d, lb_1d, ub_1d)
        gek_1d(2.5)
        std_error_at_point(gek_1d, 2.5)

        # EarthSurrogate (1D)
        earth_1d = EarthSurrogate(x_1d, y_1d, lb_1d, ub_1d)
        earth_1d(2.5)

        # EarthSurrogate (nD)
        earth_nd = EarthSurrogate(x_nd, y_nd, lb_nd, ub_nd)
        earth_nd((2.5, 3.5))

        # Sampling - Sobol
        sample(5, lb_1d, ub_1d, SobolSample())
        sample(5, lb_nd, ub_nd, SobolSample())

        # Sampling - LatinHypercube
        sample(5, lb_1d, ub_1d, LatinHypercubeSample())
        sample(5, lb_nd, ub_nd, LatinHypercubeSample())

        # Sampling - Random
        sample(5, lb_1d, ub_1d, RandomSample())
        sample(5, lb_nd, ub_nd, RandomSample())
    end
end
