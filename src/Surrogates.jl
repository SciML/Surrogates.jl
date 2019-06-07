#=
module Surrogates

greet() = print("Hello World!")

end # module
=#
module Surrogates

abstract type AbstractSurrogate <: Function end

include("Radials.jl")
include("Kriging.jl")
end
