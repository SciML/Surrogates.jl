using Base
using Test
using LinearAlgebra
using Surrogates

@testset "Radials_1D" begin
    @testset "Basis functions" begin
        #Linear basis function
        @test linear_basis_function.phi(2) == 2
        @test linear_basis_function.phi(-2) == 2
        @test linear_basis_function.phi([2.0,2.0,-1.0]) == 3

        #Cubic basis function
        @test cubic_basis_function.phi(3) == 27
        @test cubic_basis_function.phi(-2) == 8
        @test cubic_basis_function.phi([2.0,2.0,-1.0]) == 27

        #Thinplate basis function
        @test Float32(thinplate_basis_function.phi(3)) ≈ Float32(9.8875)
        @test Float32(thinplate_basis_function.phi(-4)) ≈ Float32(22.180)
        @test Float32(thinplate_basis_function.phi([2.0,3.0,1.0])) ≈ Float32(18.473)

        #Multiquadric basis function
        my_function = multiquadric_basis_function(3.0)
        @test Float32(my_function.phi(2)) ≈ Float32(3.6055)
        @test Float32(my_function.phi(-2)) ≈ Float32(3.6055)
        @test Float32(my_function.phi([2.0,3.0,1.0])) ≈ Float32(4.7958)
    end

    @testset "Functionality" begin
        x = [1.0,2.0,3.0]
        y = [1.0,1.0,1.0]
        a = 0
        b = 4
        x_star = 3.5
        coeff = Radial_1D(x,y,a,b,linear_basis_function)
        @test coeff ≈ [0.375,0.375,0.375,0.25]
        new_value = evaluate_Radial_1D(x_star,coeff,x,a,b,linear_basis_function)
        @test Float32(new_value) ≈ Float32(1.6875)
    end


end
