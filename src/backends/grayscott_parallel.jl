
"""
    ParallelGrayScott <: AbstractGrayScott

A parallel backend for computing the Gray-Scott ODE using Julia's native distributed computing library.
"""
struct ParallelGrayScott <: AbstractGrayScott end

function initial_state(init_cond, ::ParallelGrayScott)
    # We need to use a shared array in order to write concurrently from multiple processes
    x = SharedArray(pad(init_cond, 0.0, (1,1,0)))
    x, SharedArray(similar(x)) # similar returns an array, so we need to do this
end

function allocate_output(init_cond, opts::GrayScottOptions, ::ParallelGrayScott)
    SharedArray(zeros(size(init_cond)..., opts.num_output_steps))
end

function output!(out, state, ::ParallelGrayScott)
    @assert size(state, 1) == size(out, 1) + 2
    @assert size(state, 2) == size(out, 2) + 2

    @sync @distributed for j in axes(out,2)
        for i in axes(out,1)
            @inbounds out[i,j,1] = state[i+1,j+1,1]
            @inbounds out[i,j,2] = state[i+1,j+1,2]
        end
    end
end

# compute the laplacian using multiple cores
get_resource(::ParallelGrayScott) = CPUProcesses()

function reaction!(du, dv, u, v, params::GrayScottParams, ::ParallelGrayScott)
    m, n = size(du)
    f, k = params.f, params.k
    @sync @distributed for j in 2:(n-1)
        @simd for i in 2:(m-1)
            uv = u[i,j] * v[i,j]^2
            du[i,j] += -uv + f * (1 - u[i,j])
            dv[i,j] += uv - (f+k)*v[i,j]
        end
    end
end

function laplacian!(du, u, D, ::ParallelGrayScott)
    m, n = size(du)
    @sync @distributed for j in 2:(n-1)
        @simd for i in 2:(m-1)
            tmp = 0.0
            @inbounds tmp += 0.25*u[i-1,j-1] + 0.5*u[i,j-1] + 0.25*u[i+1,j-1]
            @inbounds tmp += 0.5*u[i-1,j] - 3.0*u[i,j] + 0.5*u[i+1,j]
            @inbounds tmp += 0.25*u[i-1,j+1] + 0.5*u[i,j+1] + 0.25*u[i+1,j+1]
            @inbounds du[i,j] = D * tmp
        end
    end
end
