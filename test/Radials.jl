using Base
using Test
using LinearAlgebra
using Surrogates

@testset "Radial" begin

    @testset "1D" begin

        #WITHOUT ADD_POINT
        x = [1 2 3]
        y = [4,5,6]
        a = 0
        b = 4
        my_rad = RadialBasis(x,y,a,b,z->norm(z),1)
        est = my_rad(3)
        @test est ≈ 7.875

        #WITH ADD_POINT, adding singleton
        x = [1 2 3]
        y = [4,5,6]
        a = 0
        b = 4
        my_rad = RadialBasis(x,y,a,b,z->norm(z),1)
        my_rad2= add_point!(my_rad,[4],[10])
        est = my_rad2(3)
        @test est ≈ 6.499999999999991

        #WITH ADD_POINT, adding more
        x = [1 2 3]
        y = [4,5,6]
        a = 0
        b = 4
        my_rad = RadialBasis(x,y,a,b,z->norm(z),1)
        my_rad2= add_point!(my_rad,[3.2 3.3 3.4],[8,9,10])
        est = my_rad(3)
        @test est ≈ 6.49416593818781

    end

    @testset "ND" begin
        #WITHOUT ADD_POINT
        x = [1 2 3; 4 5 6; 7 8 9]
        y = [4,5,6]
        bounds = [[0,4],[3,7],[6,10]]
        my_rad = RadialBasis(x,y,bounds,z->norm(z),1)
        est = my_rad([1 2 3])
        @test est ≈ 4.0

        #WITH ADD_POINT, adding singleton
        x = [1 2 3; 4 5 6; 7 8 9]
        y = [4,5,6]
        bounds = [[0,4],[3,7],[6,10]]
        my_rad = RadialBasis(x,y,bounds,z->norm(z),1)
        my_rad2 = add_point!(my_rad,[9 10 11],[10])
        est = my_rad2([1 2 3])
        @test est ≈ 4.0

        #WITH ADD_POINT, adding more
        x = [1 2 3; 4 5 6; 7 8 9]
        y = [4,5,6]
        bounds = [[0,4],[3,7],[6,10]]
        my_rad = RadialBasis(x,y,bounds,z->norm(z),1)
        my_rad2 = add_point!(my_rad,[9 10 11; 12 13 14],[10,11])
        est = my_rad2([1 2 3])
        @test est ≈ 4.0


    end
end
