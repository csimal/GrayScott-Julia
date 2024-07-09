using Revise

using Distributed
addprocs(8)

@everywhere include("../src/GrayScott.jl")
push!(LOAD_PATH, pwd() * "/src")


@everywhere using .GrayScott

using .GrayScott: SimpleGrayScott, AdvancedGrayScott, TurboGrayScott
using .GrayScott: ThreadedGrayScott, ParallelGrayScott
using .GrayScott: update!, output!, laplacian!
using BenchmarkTools
using GLMakie



opts = GrayScottOptions(nrow=256, ncol=256, num_output_steps=50, Δt=1.0)

params = GrayScottParams()

backend = SimpleGrayScott()

@time simulate(opts, params, SimpleGrayScott())

@time simulate(opts, params, AdvancedGrayScott())

@time simulate(opts, params, TurboGrayScott())

@time simulate(opts, params, ParallelGrayScott())

@time simulate(opts, params, backend)

init_cond = initial_condition(opts)
out = similar(init_cond)
x, dx = initial_state(init_cond, backend)

heatmap(init_cond[1,:,:])

@benchmark update!(dx, x, params, backend)

@benchmark output!(out, x, backend)


@benchmark update!(dx, x, params, AdvancedGrayScott())

@benchmark update!(dx, x, params, TurboGrayScott())



@benchmark laplacian!()
