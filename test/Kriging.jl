using Base
using Test
using LinearAlgebra
using Surrogates

@testset "Kriging" begin
    @testset "ND" begin

        #WITHOUT ADD POINT
        x = [1 2 3; 4 5 6; 7 8 9]
        y = [1,2,3]
        p = [1 1 1]
        theta = [2 2 2]
        my_k = Kriging(x,y,p,theta)
        est,std_err = my_k([1 2 3])
        @test std_err < 10^(-6)

        #WITH ADD POINT adding singleton
        x = [1 2 3; 4 5 6; 7 8 9]
        y = [1,2,3]
        p = [1 1 1]
        theta = [2 2 2]
        my_k = Kriging(x,y,p,theta)
        my_k2 = add_point!(my_k,[10 11 12],[4])
        est,std_err = my_k2([10 11 12])
        @test std_err < 10^(-6)

        #WITH ADD POINT ADDING MORE
        x = [1 2 3; 4 5 6; 7 8 9]
        y = [1,2,3]
        p = [1 1 1]
        theta = [2 2 2]
        my_k = Kriging(x,y,p,theta)
        my_k2 = add_point!(my_k,[10 11 12; 13 14 15],[4,5])
        est,std_err = my_k2([10 11 12])
        @test std_err < 10^(-6)

    end
    @testset "1D" begin
        #WITHOUT ADD POINT
        x = [1 2 3]
        y = [4,5,6]
        p = 1.3
        my_k = Kriging(x,y,p)
        est,std_err = my_k(1)
        @test std_err < 10^(-6)

        #WITH ADD POINT adding singleton
        x = [1 2 3]
        y = [4,5,6]
        p = 1.3
        my_k = Kriging(x,y,p)
        my_k2 = add_point!(my_k,[4],[9])
        est,std_err = my_k2([4])
        @test std_err < 10^(-6)

        #WITH ADD POINT adding more
        x = [1 2 3]
        y = [4,5,6]
        p = 1.3
        my_k = Kriging(x,y,p)
        my_k2 = add_point!(my_k,[4 5 6],[9,13,15])
        est,std_err = my_k2(4)
        @test std_err < 10^(-6)
    end

end
