
"""
    TurboGrayScott <: AbstractGrayScott

A backend for the Gray-Scott equations using the LoopVectorization package for autovectorization.

NOTE: Unfortunately, LoopVectorization is likely to be deprecated in the next version of Julia.
"""
struct TurboGrayScott <: AbstractGrayScott end

function initial_state(init_cond, ::TurboGrayScott)
    x = pad(init_cond, 0.0, (0,1,1))
    x, similar(x)
end

function output!(out, state, ::TurboGrayScott)
    @assert size(state, 2) == size(out, 2) + 2
    @assert size(state, 3) == size(out, 3) + 2

    @turbo for j in axes(out,2), i in axes(out,1)
        @inbounds out[1,i,j] = state[1,i+1,j+1]
        @inbounds out[2,i,j] = state[2,i+1,j+1]
    end
end

function update!(dx, x, params::GrayScottParams, gs::TurboGrayScott)
    u = view(x, 1,:,:)
    v = view(x, 2,:,:)
    du = view(dx, 1,:,:)
    dv = view(dx, 2,:,:)

    dx .= 0.0 # zero it out for sanity

    laplacian!(du, u, params.Dᵤ, gs)
    laplacian!(dv, v, params.Dᵥ, gs)

    f, k = params.f, params.k
    @turbo for i in eachindex(u)
        @inbounds uv = u[i] * v[i]^2
        @inbounds du[i] += -uv + f * (1.0 - u[i])
        @inbounds dv[i] += uv - (f+k)*v[i]
    end
end

# Note: this overrides the default method
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
