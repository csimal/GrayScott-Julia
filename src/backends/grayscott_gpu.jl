# NOTE: The following is based on code generated by Claude 3.5 Sonnet
using KernelAbstractions
using KernelAbstractions: @kernel, @index

"""
    GPUGrayScott <: AbstractGrayScott

A vendor agnostic GPU backend for the Gray-Scott ODE function. It relies on the `KernelAbstractions` package to generically define kernels.
"""
struct GPUGrayScott{T} <: AbstractGrayScott
    device::T
end

function GPUGrayScott()
    if KernelAbstractions.DEFAULT_DEVICE[] isa KernelAbstractions.GPU
        GPUGrayScott(KernelAbstractions.DEFAULT_DEVICE[])
    else
        GPUGrayScott(CPU())
    end
end

function initial_state(init_cond, gs::GPUGrayScott)
    nrow, ncol = size(init_cond)[2:3]
    u = KernelAbstractions.allocate(gs.device, eltype(init_cond), (nrow+2, ncol+2))
    v = KernelAbstractions.allocate(gs.device, eltype(init_cond), (nrow+2, ncol+2))
    copyto!(u, pad(init_cond[1,:,:], 0.0, (1,1)))
    copyto!(v, pad(init_cond[2,:,:], 0.0, (1,1)))
    du = similar(u)
    dv = similar(v)
    (u, v), (du, dv)
end

function output!(out, state, ::GPUGrayScott)
    u, v = state
    copyto!(view(out, 1, :, :), u[2:end-1, 2:end-1])
    copyto!(view(out, 2, :, :), v[2:end-1, 2:end-1])
end

function update!((du,dv), (u,v), params::GrayScottParams, gs::GPUGrayScott)
    dx .= 0
    update_kernel! = update_kernel!(gs.device)
    update_kernel!(du, dv, u, v, )
end

@kernel function update_kernel!(du, dv, u, v, Dᵤ, Dᵥ, f, k, dt)
    i, j = @index(Global, NTuple)
    if i > 1 && i < size(u, 1) && j > 1 && j < size(u, 2)
        @inbounds begin
            # Laplacian term
            tmp = 0.25f0 * (u[i-1,j-1] + u[i+1,j-1] + u[i-1,j+1] + u[i+1,j+1]) +
                  0.5f0  * (u[i,j-1] + u[i-1,j] + u[i+1,j] + u[i,j+1]) -
                  3.0f0  * u[i,j]
            du[i,j] = Dᵤ * tmp
            tmp = 0.25f0 * (v[i-1,j-1] + v[i+1,j-1] + v[i-1,j+1] + v[i+1,j+1]) +
                  0.5f0  * (v[i,j-1] + v[i-1,j] + v[i+1,j] + v[i,j+1]) -
                  3.0f0  * v[i,j]
            du[i,j] = Dᵥ * tmp

            # reaction term
            uv = u[i,j] * v[i,j]^2
            du[i,j] += -uv + f * (1.0 - u[i,j])
            dv[i,j] += uv - (f+k)*v[i,j]
            # Euler step
            u[i,j] += du[i,j] * dt
            v[i,j] += dv[i,j] * dt
        end
    end
end
