include("../src/GrayScott.jl")

using .GrayScott
using .GrayScott: SimpleGrayScott
using .GrayScott: update!, output!
using BenchmarkTools

opts = GrayScottOptions(nrow=100, ncol=100, num_output_steps=50)

params = GrayScottParams()

backend = SimpleGrayScott()

@time simulate(opts, params, backend)

init_cond = initial_state(opts)
out = similar(init_cond)
x = initial_state(init_cond, backend)
dx = similar(x)

@benchmark update!(dx, x, params, backend)

@benchmark output!(out, x, backend)
