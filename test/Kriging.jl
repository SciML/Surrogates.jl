using Base
using Test
using LinearAlgebra
using Surrogates

@testset "Kriging" begin
    @testset "ND" begin

        #WITHOUT ADD POINT
        x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
        y = [1.0,2.0,3.0]
        p = [1.0 1.0 1.0]
        theta = [2.0 2.0 2.0]
        my_k = Kriging(x,y,p,theta)
        est = my_k((1.0,2.0,3.0))
        std_err = std_error_at_point(my_k,(1.0,2.0,3.0))
        @test std_err < 10^(-6)


        #WITH ADD POINT adding singleton
        x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
        y = [1.0,2.0,3.0]
        p = [1.0 1.0 1.0]
        theta = [2.0 2.0 2.0]
        my_k = Kriging(x,y,p,theta)
        add_point!(my_k,(10.0,11.0,12.0),4.0)
        est = my_k((10.0,11.0,12.0))
        std_err = std_error_at_point(my_k,(10.0,11.0,12.0))
        @test std_err < 10^(-6)


        #WITH ADD POINT ADDING MORE
        x = [(1.0, 2.0, 3.0),(4.0,5.0,6.0),(7.0, 8.0, 9.0)]
        y = [1.0,2.0,3.0]
        p = [1.0 1.0 1.0]
        theta = [2.0 2.0 2.0]
        my_k = Kriging(x,y,p,theta)
        add_point!(my_k,[(10.0, 11.0, 12.0),(13.0,14.0,15.0)],[4.0,5.0])
        est = my_k((10.0,11.0,12.0))
        std_err = std_error_at_point(my_k,(10.0,11.0,12.0))
        @test std_err < 10^(-6)


    end
    @testset "1D" begin
        #WITHOUT ADD POINT
        x = [1.0,2.0,3.0]
        y = [4.0,5.0,6.0]
        p = 1.3
        my_k = Kriging(x,y,p)
        est = my_k(1.0)
        std_err = std_error_at_point(my_k,1.0)
        @test std_err < 10^(-6)

        #WITH ADD POINT adding singleton
        x = [1.0,2.0,3.0]
        y = [4.0,5.0,6.0]
        p = 1.3
        my_k = Kriging(x,y,p)
        add_point!(my_k,4.0,9.0)
        est = my_k(4.0)
        std_err = std_error_at_point(my_k,4.0)
        @test std_err < 10^(-6)


        #WITH ADD POINT adding more
        x = [1.0,2.0,3.0]
        y = [4.0,5.0,6.0]
        p = 1.3
        my_k = Kriging(x,y,p)
        add_point!(my_k,[4.0,5.0,6.0],[9.0,13.0,15.0])
        est = my_k(4.0)
        std_err = std_error_at_point(my_k,4.0)
        @test std_err < 10^(-6)
    end

end
