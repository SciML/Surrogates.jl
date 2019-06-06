using Base
using Test
using LinearAlgebra
using Surrogates

@testset "Kriging" begin
    @testset "ND" begin
        #=
        x = [1 2 3; 4 5 6; 7 8 9]
        y = [1,2,3]
        p = [1 1 1]
        theta = [2 2 2]
        my_k = Kriging(x,y,p,theta)
        my_k2 = add_point!(my_k,[10 11 12],[4])
        est,err = current_estimate(my_k2,[1 2 3])
        @test err < 10^(-6)
        =#
    end
    @testset "1D" begin
        #=
        x = [1 2 3]
        y = [2 5 9]
        p = 1.7
        my_rad = Kriging(x,y,p)
        estim, err = current_estimate(my_rad,4)
        print(err)
        =#
    end

end
