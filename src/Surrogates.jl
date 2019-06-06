#=
module Surrogates

greet() = print("Hello World!")

end # module
=#
module Surrogates
include("Radials.jl")
include("Kriging.jl")
end
