using Base
using Test
using LinearAlgebra
using Surrogates

@testset "Radial" begin

    @testset "1D" begin

        #WITHOUT ADD_POINT


        x = [1.0,2.0,3.0]
        y = [4.0,5.0,6.0]
        a = 0
        b = 4
        my_rad = RadialBasis(x,y,a,b,z->norm(z),1)
        est = my_rad(3.0)
        @test est ≈ 7.875


        #WITH ADD_POINT, adding singleton

        x = [1.0,2.0,3.0]
        y = [4.0,5.0,6.0]
        a = 0
        b = 4
        my_rad = RadialBasis(x,y,a,b,z->norm(z),1)
        my_rad2= add_point!(my_rad,4.0,10.0)
        est = my_rad2(3.0)
        @test est ≈ 6.499999999999991


        #WITH ADD_POINT, adding more

        x = [1.0,2.0,3.0]
        y = [4.0,5.0,6.0]
        a = 0
        b = 4
        my_rad = RadialBasis(x,y,a,b,z->norm(z),1)
        est_rad = my_rad(3.0)
        my_rad2= add_point!(my_rad,[3.2,3.3,3.4],[8.0,9.0,10.0])
        est_rad2 = my_rad2(3.0)
        @test est_rad2 ≈ 6.49416593818781




    end
    @testset "ND" begin
        #WITHOUT ADD_POINT


        x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
        y = [4.0,5.0,6.0]
        bounds = [[0.0,4.0],[3.0,7.0],[6.0,10.0]]
        my_rad = RadialBasis(x,y,bounds,z->norm(z),1)
        est = my_rad((1.0,2.0,3.0))
        @test est ≈ 4.0


        #WITH ADD_POINT, adding singleton
        x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
        y = [4.0,5.0,6.0]
        bounds = [[0.0,4.0],[3.0,7.0],[6.0,10.0]]
        my_rad = RadialBasis(x,y,bounds,z->norm(z),1)
        my_rad2 = add_point!(my_rad,(9.0,10.0,11.0),10.0)
        est = my_rad2((1.0,2.0,3.0))
        @test est ≈ 4.0
        


        #WITH ADD_POINT, adding more

        x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
        y = [4.0,5.0,6.0]
        bounds = [[0.0,4.0],[3.0,7.0],[6.0,10.0]]
        my_rad = RadialBasis(x,y,bounds,z->norm(z),1)
        my_rad2 = add_point!(my_rad,[(9.0,10.0,11.0),(12.0,13.0,14.0)],[10.0,11.0])
        est = my_rad2((1.0,2.0,3.0))
        @test est ≈ 4.0

    end
end
