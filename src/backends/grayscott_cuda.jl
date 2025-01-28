
"""
    CUDAGrayScott <: AbstractGrayScott

A GPU backend for the Gray-Scott ODE function using CUDA.jl

This backend only supports Nvidia GPUs.
"""
struct CUDAGrayScott <: AbstractGrayScott end

function initial_state(init_cond, ::CUDAGrayScott)
    nrow, ncol = size(init_cond)[1:2]
    u = CuArray{Float32,2}(undef, nrow+2, ncol+2)
    v = CuArray{Float32,2}(undef, nrow+2, ncol+2)
    du = CuArray{Float32,2}(undef, nrow+2, ncol+2)
    dv = CuArray{Float32,2}(undef, nrow+2, ncol+2)
    copyto!(u, pad(init_cond[:,:,1], 0.0, (1,1)))
    copyto!(v, pad(init_cond[:,:,2], 0.0, (1,1)))
    return (u,v), (du,dv)
end

function output!(out, state, ::CUDAGrayScott)
    u, v = state
    tmp = Array(u)
    copyto!(view(out, :, :, 1), tmp[2:end-1, 2:end-1])
    copyto!(tmp, v)
    copyto!(view(out, :, :, 2), tmp[2:end-1, 2:end-1])
end

function ode_step!(dx, x, params::GrayScottParams, ::CUDAGrayScott)
    nothing # the ode step is fused into the main kernel
end

function update!(dx, x, params::GrayScottParams, ::CUDAGrayScott)
    du, dv = dx
    u, v = x
    Dᵤ = Float32(params.Dᵤ)
    Dᵥ = Float32(params.Dᵥ)
    f = Float32(params.f)
    k = Float32(params.k)
    dt = Float32(params.dt)
    threads = (16, 16)
    blocks = ceil.(Int, size(du) ./ threads)
    @cuda blocks=blocks threads=threads update_kernel!(du, dv, u, v, Dᵤ, Dᵥ, f, k, dt)
end

# Fused kernel
function update_kernel!(du, dv, u, v, Dᵤ, Dᵥ, f, k, dt)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if 2 <= i < size(du, 1) && 2 <= j < size(du, 2)
        # reaction part
        uv = u[i,j] * v[i,j]^2
        @inbounds du[i,j] += -uv + f * (1.0f0 - u[i,j])
        @inbounds dv[i,j] += uv - (f + k) * v[i,j]
        # diffusion part
        tmp = 0.0f0
        @inbounds begin
            tmp += 0.05f0*u[i-1,j-1] + 0.2f0*u[i,j-1] + 0.05f0*u[i+1,j-1]
            tmp += 0.2f0*u[i-1,j] - 1.0f0*u[i,j] + 0.2f0*u[i+1,j]
            tmp += 0.05f0*u[i-1,j+1] + 0.2f0*u[i,j+1] + 0.05f0*u[i+1,j+1]
            du[i,j] += Dᵤ * tmp
        end
        tmp = 0.0f0
        @inbounds begin
            tmp += 0.05f0*v[i-1,j-1] + 0.2f0*v[i,j-1] + 0.05f0*v[i+1,j-1]
            tmp += 0.2f0*v[i-1,j] - 1.0f0*v[i,j] + 0.2f0*v[i+1,j]
            tmp += 0.05f0*v[i-1,j+1] + 0.2f0*v[i,j+1] + 0.05f0*v[i+1,j+1]
            dv[i,j] += Dᵥ * tmp
        end
        # Euler step
        u[i,j] += du[i,j] * dt
        v[i,j] += dv[i,j] * dt
        # clamp
        @inbounds u[i,j] = clamp(u[i,j], 0.0f0, 1.0f0)
        @inbounds v[i,j] = clamp(v[i,j], 0.0f0, 1.0f0)
    end
    return nothing
end

function grayscott_reaction_kernel!(du, dv, u, v, f, k)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i <= size(du, 1) && j <= size(du, 2)
        uv = u[i,j] * v[i,j]^2
        @inbounds du[i,j] += -uv + f * (1.0 - u[i,j])
        @inbounds dv[i,j] += uv - (f + k) * v[i,j]
    end
    return nothing
end

function grayscott_reaction!(du, dv, u, v, f, k)
    threads = (16, 16)
    blocks = ceil.(Int, size(du) ./ threads)
    @cuda blocks=blocks threads=threads grayscott_reaction_kernel!(du, dv, u, v, f, k)
end

function diffuse_kernel!(du, u, D)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if 2 <= i < size(du, 1) && 2 <= j < size(du, 2)
        tmp = 0.0f0
        @inbounds begin
            tmp += 0.05f0*u[i-1,j-1] + 0.2f0*u[i,j-1] + 0.05f0*u[i+1,j-1]
            tmp += 0.2f0*u[i-1,j] - 1.0f0*u[i,j] + 0.2f0*u[i+1,j]
            tmp += 0.05f0*u[i-1,j+1] + 0.2f0*u[i,j+1] + 0.05f0*u[i+1,j+1]
            du[i,j] += D * tmp
        end
    end
    return nothing
end

function diffuse!(du, u, D)
    threads = (16, 16)
    blocks = ceil.(Int, size(du) ./ threads)
    @cuda blocks=blocks threads=threads diffuse_kernel!(du, u, D)
end

function clamp01_kernel!(x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    if i <= size(x, 1) && j <= size(x, 2)
        @inbounds x[i,j] = clamp(x[i,j], 0.0f0, 1.0f0)
    end
    return nothing
end

function clamp01!(x)
    threads = (16, 16)
    blocks = ceil.(Int, size(x) ./ threads)
    @cuda blocks=blocks threads=threads clamp01_kernel!(x)
end