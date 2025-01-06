
using Distributed
addprocs(8)

@everywhere using GrayScott
using GrayScott
using GrayScott: SimpleGrayScott, AdvancedGrayScott, TurboGrayScott
using GrayScott: ThreadedGrayScott, ParallelGrayScott, SIMDGrayScott
using GrayScott: update!, output!, reaction!, laplacian!
using GrayScott: laplacian_!
using BenchmarkTools
using GLMakie



opts = GrayScottOptions(nrow=256, ncol=256, num_output_steps=50, Δt=1.0)

params = GrayScottParams()

backend = SimpleGrayScott()

@time simulate(opts, params, SimpleGrayScott())

@time simulate(opts, params, AdvancedGrayScott())

@time simulate(opts, params, TurboGrayScott())

@time simulate(opts, params, ParallelGrayScott())

@time simulate(opts, params, SIMDGrayScott{8}())

@time simulate(opts, params, backend)

init_cond = initial_condition(opts)
out = similar(init_cond)
x, dx = initial_state(init_cond, backend)

heatmap(init_cond[1,:,:])

@benchmark update!(dx, x, params, backend)

@benchmark output!(out, x, backend)

@benchmark update!(dx, x, params, SimpleGrayScott())

@benchmark update!(dx, x, params, AdvancedGrayScott())

@benchmark update!(dx, x, params, TurboGrayScott())


u = x[:,:,1]
du = dx[:,:,2]
v = x[:,:,2]
dv = dx[:,:,2]

b_simple = @benchmark laplacian!(du, u, 0.3, SimpleGrayScott())

@code_warntype laplacian!(du, u, 0.3, SimpleGrayScott())

b_advanced = @benchmark laplacian!(du, u, 0.3, AdvancedGrayScott())

b_turbo = @benchmark laplacian!(du, u, 0.3, TurboGrayScott())

size(du)
using StaticKernels
k = GrayScott.k
k_ = StaticKernels.Kernel{(-1:1,-1:1)}(
    @inline w -> 0.25*w[-1,-1] + 0.5*w[0,-1] + 0.25*w[1,-1] + 0.5*w[-1,0] - 3.0*w[0,0] + 0.5*w[1,0] + 0.25*w[-1,1] + 0.5*w[0,1] + 0.25*w[1,1]
)
b_static = @benchmark laplacian_!(view(du, 2:257, 2:257), u, k, 0.3, SimpleGrayScott())

@code_warntype laplacian_!(du, u, k, 0.3, SimpleGrayScott())


struct GenericGrayScott <: GrayScott.AbstractGrayScott end

b_imfilter = @benchmark laplacian!(du, u, 0.3, GenericGrayScott())

using StatsPlots

begin
    boxplot(b_simple.times, label="Simple", yscale=:log10, ylabel="Time [ns]", title="Time to compute diffusion term")
    boxplot!(b_advanced.times, label="Advanced")
    boxplot!(b_static.times, label="StaticKernels.jl")
    #boxplot!(b_imfilter.times, label="ImageFilters.jl")
    boxplot!(b_turbo.times, label="LoopVectorization.jl")
end

br_simple = @benchmark reaction!(du, dv, u, v, params, SimpleGrayScott())
br_advanced = @benchmark reaction!(du, dv, u, v, params, AdvancedGrayScott())
br_turbo = @benchmark reaction!(du, dv, u, v, params, TurboGrayScott())

begin
    boxplot(br_simple.times, label="Simple", title="Time to compute reaction term", ylabel="Time [ns]")
    boxplot!(br_advanced.times, label="Advanced")
    boxplot!(br_turbo.times, label="Turbo")
end
