using SIMD

"""
    SIMDGrayScott{W} <: AbstractGrayScott

A backend for computing the Gray-Scott ODE function using explicit SIMD operations.
W is the SIMD width.
"""
struct SIMDGrayScott{W} <: AbstractGrayScott end

# Constructor that chooses an appropriate SIMD width based on the system
function SIMDGrayScott()
    if SIMD.VecWidth{Float64}() == 8
        SIMDGrayScott{8}()  # AVX-512
    elseif SIMD.VecWidth{Float64}() == 4
        SIMDGrayScott{4}()  # AVX2
    elseif SIMD.VecWidth{Float64}() == 2
        SIMDGrayScott{2}()  # SSE2
    else
        SIMDGrayScott{1}()  # Fallback to scalar operations
    end
end

function initial_state(init_cond, ::SIMDGrayScott{W}) where W
    u = pad(init_cond[1,:,:], 0.0, (1,1))
    v = pad(init_cond[2,:,:], 0.0, (1,1))
    du = similar(u)
    dv = similar(v)
    (u, v), (du, dv)
end

function output!(out, state, ::SIMDGrayScott{W}) where W
    u, v = state
    copyto!(view(out, 1, :, :), view(u, 2:size(u,1)-1, 2:size(u,2)-1))
    copyto!(view(out, 2, :, :), view(v, 2:size(v,1)-1, 2:size(v,2)-1))
end

function simd_laplacian!(du::AbstractMatrix{T}, u::AbstractMatrix{T}, D::T, ::SIMDGrayScott{W}) where {T, W}
    height, width = size(u)
    Vec = SIMD.Vec{W, T}

    for j in 2:width-1
        for i in 2:W:height-1
            u_center = Vec(u, i, j)
            u_left = Vec(u, i-1, j)
            u_right = Vec(u, i+1, j)
            u_up = Vec(u, i, j-1)
            u_down = Vec(u, i, j+1)
            u_upleft = Vec(u, i-1, j-1)
            u_upright = Vec(u, i+1, j-1)
            u_downleft = Vec(u, i-1, j+1)
            u_downright = Vec(u, i+1, j+1)

            tmp = T(0.25) * (u_upleft + u_upright + u_downleft + u_downright) +
                  T(0.5)  * (u_up + u_left + u_right + u_down) -
                  T(3.0)  * u_center

            vstore(tmp * D, du, i, j)
        end
    end
end

function simd_update!(du::AbstractMatrix{T}, dv::AbstractMatrix{T},
                      u::AbstractMatrix{T}, v::AbstractMatrix{T},
                      f::T, k::T, ::SIMDGrayScott{W}) where {T, W}
    height, width = size(u)
    Vec = SIMD.Vec{W, T}

    for j in 2:width-1
        for i in 2:W:height-1
            u_vec = Vec(u, i, j)
            v_vec = Vec(v, i, j)
            du_vec = Vec(du, i, j)
            dv_vec = Vec(dv, i, j)

            uv = u_vec * v_vec * v_vec
            du_vec += -uv + f * (T(1.0) - u_vec)
            dv_vec += uv - (f + k) * v_vec

            vstore(du_vec, du, i, j)
            vstore(dv_vec, dv, i, j)
        end
    end
end

function simd_update_state!(u::AbstractMatrix{T}, v::AbstractMatrix{T},
                            du::AbstractMatrix{T}, dv::AbstractMatrix{T},
                            dt::T, ::SIMDGrayScott{W}) where {T, W}
    Vec = SIMD.Vec{W, T}

    for j in 1:size(u, 2)
        for i in 1:W:size(u, 1)-W+1
            u_vec = Vec(u, i, j)
            v_vec = Vec(v, i, j)
            du_vec = Vec(du, i, j)
            dv_vec = Vec(dv, i, j)

            u_vec += du_vec * dt
            v_vec += dv_vec * dt

            vstore(u_vec, u, i, j)
            vstore(v_vec, v, i, j)
        end
    end
end

function update!((du, dv), (u, v), params::GrayScottParams, gs::SIMDGrayScott{W}) where W
    fill!(du, 0.0)
    fill!(dv, 0.0)

    simd_laplacian!(du, u, params.Dᵤ, gs)
    simd_laplacian!(dv, v, params.Dᵥ, gs)

    simd_update!(du, dv, u, v, params.f, params.k, gs)

    simd_update_state!(u, v, du, dv, params.Δt, gs)
end
