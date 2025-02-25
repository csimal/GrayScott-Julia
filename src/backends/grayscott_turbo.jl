
"""
    TurboGrayScott <: AbstractGrayScott

A backend for the Gray-Scott equations using the LoopVectorization package for autovectorization.
"""
struct TurboGrayScott <: AbstractGrayScott end

function output!(out, state, ::TurboGrayScott)
    @assert size(state, 1) == size(out, 1) + 2
    @assert size(state, 2) == size(out, 2) + 2

    @turbo for j in axes(out,2), i in axes(out,1)
        @inbounds out[i,j,1] = state[i+1,j+1,1]
        @inbounds out[i,j,2] = state[i+1,j+1,2]
    end
end

function reaction!(du, dv, u, v, params::GrayScottParams, ::TurboGrayScott)
    m, n = size(du)
    f, k = params.f, params.k
    @turbo for j in 2:(n-1), i in 2:(m-1)
        @inbounds uv = u[i,j] * v[i,j]^2
        @inbounds du[i,j] += -uv + f * (1.0 - u[i,j])
        @inbounds dv[i,j] += uv - (f+k)*v[i,j]
    end
end

function laplacian!(du, u, D, ::TurboGrayScott)
    m, n = size(du)
    @turbo for j in 2:(n-1), i in 2:(m-1)
        tmp = 0.0
        tmp += 0.25*u[i-1,j-1] + 0.5*u[i,j-1] + 0.25*u[i+1,j-1]
        tmp += 0.5*u[i-1,j] - 3.0*u[i,j] + 0.5*u[i+1,j]
        tmp += 0.25*u[i-1,j+1] + 0.5*u[i,j+1] + 0.25*u[i+1,j+1]
        du[i,j] = D * tmp
    end
end
