#=
module Surrogates

greet() = print("Hello World!")

end # module
=#
module Surrogates
include("Radials_1D.jl")
include("Kriging_1D.jl")
end
