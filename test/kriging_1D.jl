using Base
using Test
using LinearAlgebra
using Surrogates

@testset "Kriging" begin
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

        x = [1 2 3; 4 5 6; 7 8 9]
        y = [1,2,3]
        p = [1 1 1]
        theta = [2 2 2]
        mu, b, sigma,inverse_of_R = Kriging_ND(x,y,p,theta)
        new_point_fake = [4 5 6]
        prediction,std_error = evaluate_Kriging_ND(new_point_fake,x,p,theta,mu,b,sigma,inverse_of_R)
        @test std_error < 10^-6

    end
end
