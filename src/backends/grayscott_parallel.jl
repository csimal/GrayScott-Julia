
"""
    ParallelGrayScott <: AbstractGrayScott

A parallel backend for computing the Gray-Scott ODE using Julia's native distributed computing library.
"""
struct ParallelGrayScott <: AbstractGrayScott end

function initial_state(init_cond, ::ParallelGrayScott)
    # We need to use a shared array in order to write concurrently from multiple processes
    x = SharedArray(pad(init_cond, 0.0, (0,1,1)))
    x, SharedArray(similar(x)) # similar returns an array, so we need to do this
end

function allocate_output(init_cond, opts::GrayScottOptions, ::ParallelGrayScott)
    SharedArray(zeros(opts.num_output_steps, size(init_cond)...))
end

function output!(out, state, ::ParallelGrayScott)
    @assert size(state, 2) == size(out, 2) + 2
    @assert size(state, 3) == size(out, 3) + 2

    @sync @distributed for j in axes(out,2)
        for i in axes(out,1)
            @inbounds out[1,i,j] = state[1,i+1,j+1]
            @inbounds out[2,i,j] = state[2,i+1,j+1]
        end
    end
end

# compute the laplacian using multiple cores
get_resource(::ParallelGrayScott) = CPUProcesses()

function update!(dx, x, params::GrayScottParams, backend::ParallelGrayScott)
    u = view(x, 1,:,:)
    v = view(x, 2,:,:)
    du = view(dx, 1,:,:)
    dv = view(dx, 2,:,:)

    dx .= 0.0 # zero it out for sanity

    laplacian!(du, u, params.Dᵤ, backend)
    laplacian!(dv, v, params.Dᵥ, backend)

    f, k = params.f, params.k
    @sync @distributed for i in eachindex(u)
        uv = u[i] * v[i]^2
        du[i] += -uv + f * (1 - u[i])
        dv[i] += uv - (f+k)*v[i]
    end

end
