using Base
using Test
using LinearAlgebra
using Surrogates

@testset "Krigin_1D" begin
    @testset "Functionality" begin
        x = [1.0,2.0,3.0]
        y = [1.0,1.0,1.0]
        p = 1.6
        x_fake_new_value = 3.0
        mu,b,sigma,inverse_of_R = Kriging_1D(x,y,p)
        prediction_fake, std_error_fake = evaluate_Kriging_1D(x_fake_new_value,x,p,mu,b,sigma,inverse_of_R)
        @test std_error_fake < 10^-6
        @test prediction_fake ≈ 1.0

        x_new_value = 4.0
        prediction_true, std_error_true = evaluate_Kriging_1D(x_new_value,x,p,mu,b,sigma,inverse_of_R)
        @test prediction_true ≈ 1.0

    end
end
