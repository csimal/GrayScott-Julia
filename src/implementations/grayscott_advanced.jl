
struct AdvancedGrayScott <: AbstractGrayScott end

function initial_state(init_cond, ::AdvancedGrayScott)
    pad(init_cond, 0.0, (0,1,1))
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

function laplacian!(du, u, D, ::AdvancedGrayScott)
    m, n = size(du)
    tmp = 0.0
    @simd for j in 2:(n-1), i in 2:(m-1)
        @inbounds tmp += 0.25*u[i-1,j-1] + 0.5*u[i,j-1] + 0.25*u[i+1,j-1]
        @inbounds tmp += 0.5*(u[i-1,j] + u[i+1,j])
        @inbounds tmp += 0.5*u[i-1,j+1] + 0.5*u[i,j+1] + 0.25*u[i+1,j+1]
        @inbounds du[i,j] = D * tmp
    end
end
