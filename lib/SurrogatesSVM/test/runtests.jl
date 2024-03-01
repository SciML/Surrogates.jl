using SafeTestsets

@safetestset "SVMSurrogate" begin
    using SurrogatesSVM
    using Surrogates
    using LIBSVM
    using Test
    @testset "1D" begin
        obj_1D = x -> 2 * x + 1
        a = 0.0
        b = 10.0
        x = sample(5, a, b, SobolSample())
        y = obj_1D.(x)
        svm = LIBSVM.fit!(SVC(), reshape(x, length(x), 1), y)
        my_svm_1D = SVMSurrogate(x, y, a, b)
        val = my_svm_1D([5.0])
        @test LIBSVM.predict(svm, [5.0;;])[1] == val
        update!(my_svm_1D, [3.1], [7.2])
        update!(my_svm_1D, [3.2, 3.5], [7.4, 8.0])
        svm = LIBSVM.fit!(SVC(), reshape(my_svm_1D.x, length(my_svm_1D.x), 1), my_svm_1D.y)
        val = my_svm_1D([3.1])
        @test LIBSVM.predict(svm, [3.1;;])[1] == val
    end
    @testset "ND" begin
        obj_N = x -> x[1]^2 * x[2]
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        x = collect.(sample(100, lb, ub, RandomSample()))
        y = obj_N.(x)
        svm = LIBSVM.fit!(SVC(), transpose(reduce(hcat, x)), y)
        my_svm_ND = SVMSurrogate(x, y, lb, ub)
        x_test = [5.0, 1.2]
        val = my_svm_ND(x_test)
        @test LIBSVM.predict(svm, reshape(x_test, 1, 2))[1] == val
        update!(my_svm_ND, [[1.0, 1.0]], [1.0])
        update!(my_svm_ND, [[1.2, 1.2], [1.5, 1.5]], [1.728, 3.375])
        svm = LIBSVM.fit!(SVC(), transpose(reduce(hcat, my_svm_ND.x)), my_svm_ND.y)
        x_test = [1.0, 1.0]
        val = my_svm_ND(x_test)
        @test LIBSVM.predict(svm, reshape(x_test, 1, 2))[1] == val
    end
end
