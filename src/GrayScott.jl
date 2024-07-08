module GrayScott

# Base scientific packages
using LinearAlgebra
using HDF5
using OrdinaryDiffEq
# HPC packages
using Distributed # Distributed computing
using SIMD # Explicit SIMD
using LoopVectorization # Autovectorization
using KernelAbstractions # Vendor agnostic GPU kernels
# utilities
using ArrayPadding
using ProgressMeter

export GrayScottOptions
export GrayScottParams
export initial_state
export simulate

Base.@kwdef struct GrayScottOptions
    nrow::Int = 1080
    ncol::Int = 1920
    output::String = "data/output.h5"
    num_output_steps::Int = 1000
    num_extra_steps::Int = 34
    Δt::Float64 = 1.0
end

function initial_state(opts::GrayScottOptions)
    x = zeros(2, opts.nrow, opts.ncol)
    x[1,:,:] .= 1.0
    imin = min(7 * opts.nrow ÷ 16 - 3, 1)
    imax = min(8 * opts.nrow ÷ 16 - 3, 1)
    jmin = 7 * opts.ncol ÷ 16
    jmax = 8 * opts.ncol ÷ 16

    x[1,imin:imax,jmin:jmax] .= 0.0
    x[2,imin:imax,jmin:jmax] .= 1.0
    return x
end

Base.@kwdef struct GrayScottParams{T<:Real}
    Dᵤ::T = 0.1 # Diffusion rate for u
    Dᵥ::T = 0.05 # Diffusion rate for v
    f::T = 0.054 # Birth rate
    k::T = 0.014 # Death rate
end

abstract type AbstractGrayScott end

initial_state(init_cond, ::AbstractGrayScott) = init_cond

include("implementations/grayscott_simple.jl")
include("implementations/grayscott_advanced.jl")
include("implementations/grayscott_parallel.jl")
include("implementations/grayscott_simd.jl")
include("implementations/grayscott_turbo.jl")
include("implementations/grayscott_gpu.jl")

function simulate(
    opts::GrayScottOptions,
    params::GrayScottParams,
    backend::AbstractGrayScott,
    init_cond = initial_state(opts)
)
    @assert ndims(init_cond) == 3 "Initial condition must be a 3-dimensional array"
    @assert size(init_cond, 1) == 2 "The first dimension of the initial condition must be of length 2"

    x = initial_state(init_cond, backend)
    dx = similar(x)

    out = zeros(opts.num_output_steps, size(init_cond)...)

    @showprogress for i in 1:opts.num_output_steps
        for _ in 1:opts.num_extra_steps
            update!(dx, x, params, backend)
            @. x += dx * opts.Δt

        end
        output!(view(out, i,:,:,:), x, backend)
    end
    return out
end


end
