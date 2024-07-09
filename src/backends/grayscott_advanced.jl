
"""
    AdvancedGrayScott

A slightly sophisticated backend for the Gray-Scott ODE. It is essentially the same as `SimpleGrayScott` but with a few macros slapped on to disable array bounds checking and encouraging the compiler to use SIMD.
"""
struct AdvancedGrayScott <: AbstractGrayScott end

function initial_state(init_cond, ::AdvancedGrayScott)
    x = pad(init_cond, 0.0, (0,1,1))
    x, similar(x)
end

function output!(out, state, ::AdvancedGrayScott)
    @assert size(state, 2) == size(out, 2) + 2
    @assert size(state, 3) == size(out, 3) + 2

    for j in axes(out,2), i in axes(out,1)
        @inbounds out[1,i,j] = state[1,i+1,j+1]
        @inbounds out[2,i,j] = state[2,i+1,j+1]
    end
end

function update!(dx, x, params::GrayScottParams, gs::AdvancedGrayScott)
    u = view(x, 1,:,:)
    v = view(x, 2,:,:)
    du = view(dx, 1,:,:)
    dv = view(dx, 2,:,:)

    dx .= 0.0 # zero it out for sanity

    laplacian!(du, u, params.Dᵤ, gs)
    laplacian!(dv, v, params.Dᵥ, gs)

    f, k = params.f, params.k
    @simd for i in eachindex(u)
        @inbounds uv = u[i] * v[i]^2
        @inbounds du[i] += -uv + f * (1 - u[i])
        @inbounds dv[i] += uv - (f+k)*v[i]
    end
end
