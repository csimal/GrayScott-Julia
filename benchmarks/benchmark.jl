include("../src/GrayScott.jl")

using .GrayScott
using .GrayScott: SimpleGrayScott, AdvancedGrayScott
using .GrayScott: update!, output!
using BenchmarkTools
using GLMakie,

backends = [
    :SimpleGrayScott,
    :AdvancedGrayScott
]
