
"""
    SimpleGrayScott <: AbstractGrayScott

A simple backend for computing the Gray-Scott ODE function with nothing special in terms of optimization.
"""
struct SimpleGrayScott <: AbstractGrayScott end

function output!(out, state, ::SimpleGrayScott)
    @assert size(state, 1) == size(out, 1) + 2
    @assert size(state, 2) == size(out, 2) + 2

    for j in axes(out,2), i in axes(out,1)
        out[i,j,1] = state[i+1,j+1,1]
        out[i,j,2] = state[i+1,j+1,2]
    end
end

function reaction!(du, dv, u, v, params::GrayScottParams, ::SimpleGrayScott)
    f, k = params.f, params.k
    m, n = size(du)
    for j in 2:(n-1), i in 2:(m-1)
        uv = u[i,j] * v[i,j]^2
        du[i,j] += -uv + f * (1.0 - u[i,j])
        dv[i,j] += uv - (f+k)*v[i,j]
    end
end

function laplacian!(du, u, D, ::SimpleGrayScott)
    m, n = size(du)
    for j in 2:(n-1), i in 2:(m-1)
        tmp = 0.0
        tmp += 0.25*u[i-1,j-1] + 0.5*u[i,j-1] + 0.25*u[i+1,j-1]
        tmp += 0.5*u[i-1,j] - 3.0*u[i,j] + 0.5*u[i+1,j]
        tmp += 0.25*u[i-1,j+1] + 0.5*u[i,j+1] + 0.25*u[i+1,j+1]
        du[i,j] = D * tmp
    end
end
