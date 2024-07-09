
"""
    SimpleGrayScott <: AbstractGrayScott

A simple backend for computing the Gray-Scott ODE function with nothing special in terms of optimization.
"""
struct SimpleGrayScott <: AbstractGrayScott end

function initial_state(init_cond, ::SimpleGrayScott)
    x = pad(init_cond, 0.0, (0,1,1))
    x, similar(x)
end

function output!(out, state, ::SimpleGrayScott)
    @assert size(state, 2) == size(out, 2) + 2
    @assert size(state, 3) == size(out, 3) + 2

    for j in axes(out,2), i in axes(out,1)
        out[1,i,j] = state[1,i+1,j+1]
        out[2,i,j] = state[2,i+1,j+1]
    end
end

function update!(dx, x, params::GrayScottParams, gs::SimpleGrayScott)
    u = view(x, 1,:,:)
    v = view(x, 2,:,:)
    du = view(dx, 1,:,:)
    dv = view(dx, 2,:,:)

    dx .= 0.0 # zero it out for sanity

    laplacian!(du, u, params.Dᵤ, gs)
    laplacian!(dv, v, params.Dᵥ, gs)

    f, k = params.f, params.k
    for i in eachindex(u)
        uv = u[i] * v[i]^2
        du[i] += -uv + f * (1.0 - u[i])
        dv[i] += uv - (f+k)*v[i]
    end

end

function laplacian!(du, u, D, ::SimpleGrayScott)
    m, n = size(du)
    for j in 2:(n-1), i in 2:(m-1)
        tmp = 0.0
        tmp += 0.25*u[i-1,j-1] + 0.5*u[i,j-1] + 0.25*u[i+1,j-1]
        tmp += 0.5*(u[i-1,j] + u[i+1,j]) - 3.0*u[i,j]
        tmp += 0.5*u[i-1,j+1] + 0.5*u[i,j+1] + 0.25*u[i+1,j+1]
        du[i,j] = D * tmp
    end
end
