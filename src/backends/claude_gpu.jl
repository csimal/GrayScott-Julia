using KernelAbstractions

"""
    GPUGrayScott <: AbstractGrayScott

A backend-agnostic GPU implementation for computing the Gray-Scott ODE function using KernelAbstractions.
"""
struct GPUGrayScott{T} <: AbstractGrayScott
    device::T
end

# Constructor that defaults to CUDADevice if available, otherwise falls back to CPU
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

function output!(out, state, gs::GPUGrayScott)
    u, v = state
    copyto!(view(out, 1, :, :), u[2:end-1, 2:end-1])
    copyto!(view(out, 2, :, :), v[2:end-1, 2:end-1])
end

@kernel function laplacian_kernel!(du, u, D)
    i, j = @index(Global, NTuple)
    if i > 1 && i < size(u, 1) && j > 1 && j < size(u, 2)
        @inbounds begin
            tmp = 0.25 * (u[i-1,j-1] + u[i+1,j-1] + u[i-1,j+1] + u[i+1,j+1]) +
                  0.5  * (u[i,j-1] + u[i-1,j] + u[i+1,j] + u[i,j+1]) -
                  3.0  * u[i,j]
            du[i,j] = D * tmp
        end
    end
end

@kernel function update_kernel!(du, dv, u, v, f, k)
    i, j = @index(Global, NTuple)
    if i > 1 && i < size(u, 1) && j > 1 && j < size(u, 2)
        @inbounds begin
            uv = u[i,j] * v[i,j]^2
            du[i,j] += -uv + f * (1.0 - u[i,j])
            dv[i,j] += uv - (f+k)*v[i,j]
        end
    end
end

@kernel function update_state_kernel!(u, v, du, dv, dt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        u[i,j] += du[i,j] * dt
        v[i,j] += dv[i,j] * dt
    end
end

function update!((du, dv), (u, v), params::GrayScottParams, gs::GPUGrayScott)
    fill!(du, 0.0)
    fill!(dv, 0.0)

    ndrange = size(u)

    laplacian_kernel! = laplacian_kernel!(gs.device)
    update_kernel! = update_kernel!(gs.device)
    update_state_kernel! = update_state_kernel!(gs.device)

    event = laplacian_kernel!(du, u, params.Dᵤ; ndrange=ndrange)
    wait(event)
    event = laplacian_kernel!(dv, v, params.Dᵥ; ndrange=ndrange)
    wait(event)

    event = update_kernel!(du, dv, u, v, params.f, params.k; ndrange=ndrange)
    wait(event)

    event = update_state_kernel!(u, v, du, dv, params.Δt; ndrange=ndrange)
    wait(event)
end
