using SafeTestsets

@safetestset "RandomForestSurrogate" begin
    using Surrogates
    using SurrogatesRandomForest
    using Test
    using XGBoost: xgboost, predict
    @testset "1D" begin
        obj_1D = x -> 3 * x + 1
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = obj_1D.(x)
        a = 0.0
        b = 10.0
        num_round = 2
        my_forest_1D = RandomForestSurrogate(x, y, a, b; num_round = 2)
        xgboost1 = xgboost((reshape(x, length(x), 1), y); num_round = 2)
        val = my_forest_1D(3.5)
        @test predict(xgboost1, [3.5;;])[1] == val
        update!(my_forest_1D, [6.0], [19.0])
        update!(my_forest_1D, [7.0, 8.0], obj_1D.([7.0, 8.0]))
    end
    @testset "ND" begin
        lb = [0.0, 0.0, 0.0]
        ub = [10.0, 10.0, 10.0]
        x = collect.(sample(5, lb, ub, SobolSample()))
        obj_ND = x -> x[1] * x[2]^2 * x[3]
        y = obj_ND.(x)
        my_forest_ND = RandomForestSurrogate(x, y, lb, ub; num_round = 2)
        xgboostND = xgboost((reduce(hcat, x)', y); num_round = 2)
        val = my_forest_ND([1.0, 1.0, 1.0])
        @test predict(xgboostND, reshape([1.0, 1.0, 1.0], 3, 1))[1] == val
        update!(my_forest_ND, [[1.0, 1.0, 1.0]], [1.0])
        update!(my_forest_ND, [[1.2, 1.2, 1.0], [1.5, 1.5, 1.0]], [1.728, 3.375])
    end
end
