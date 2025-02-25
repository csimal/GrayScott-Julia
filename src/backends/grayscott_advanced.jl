
"""
    AdvancedGrayScott

A slightly sophisticated backend for the Gray-Scott ODE. It is essentially the same as `SimpleGrayScott` but with a few macros slapped on to disable array bounds checking and encourage the compiler to use SIMD instructions.
"""
struct AdvancedGrayScott <: AbstractGrayScott end

function output!(out, state, ::AdvancedGrayScott)
    @assert size(state, 1) == size(out, 1) + 2
    @assert size(state, 2) == size(out, 2) + 2

    for j in axes(out,2), i in axes(out,1)
        @inbounds out[i,j,1] = state[i+1,j+1,1]
        @inbounds out[i,j,2] = state[i+1,j+1,2]
    end
end

function reaction!(du, dv, u, v, params::GrayScottParams, ::AdvancedGrayScott)
    m, n = size(du)
    f, k = params.f, params.k
    @inbounds for j in 2:(n-1)
        @simd for i in 2:(m-1)
            uv = u[i,j] * v[i,j]^2
            du[i,j] += -uv + f * (1 - u[i,j])
            dv[i,j] += uv - (f+k)*v[i,j]
        end
    end
end

function laplacian!(du, u, D, ::AdvancedGrayScott)
    m, n = size(du)
    @inbounds for j in 2:(n-1)
        @simd for i in 2:(m-1)
            tmp = 0.0
            tmp += 0.25*u[i-1,j-1] + 0.5*u[i,j-1] + 0.25*u[i+1,j-1]
            tmp += 0.5*u[i-1,j] - 3.0*u[i,j] + 0.5*u[i+1,j]
            tmp += 0.25*u[i-1,j+1] + 0.5*u[i,j+1] + 0.25*u[i+1,j+1]
            du[i,j] = D * tmp
        end
    end
end
