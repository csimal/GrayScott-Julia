
"""
    AbstractGrayScott

The abstract type for backends to compute the Gray-Scott ODE.

Generic methods for setting up the initial state and data structures are provided, as well as a generic method for `update!`.

Any subtype needs to implement `output!`, `reaction!` and `laplacian!` to have a functional implementation.
"""
abstract type AbstractGrayScott end

function initial_state(init_cond, ::AbstractGrayScott)
    x = pad(init_cond, 0.0, (1,1,0))
    x, similar(x)
end

function allocate_output(init_cond, opts::GrayScottOptions, ::AbstractGrayScott)
    zeros(size(init_cond)..., opts.num_output_steps)
end

# by default, the convolution will be computed on a single core
get_resource(::AbstractGrayScott) = CPU1()

const LAPLACIAN_KERNEL = centered([
    0.25 0.5 0.25;
    0.5 -3.0 0.5;
    0.25 0.5 0.25
])

function ode_step!(dx, x, params::GrayScottParams, ::AbstractGrayScott)
    @. x += dx * params.dt
    clamp!(x, 0.0, 1.0)
end

function update!(dx, x, params::GrayScottParams, backend::AbstractGrayScott)
    u = view(x, :,:,1)
    v = view(x, :,:,2)
    du = view(dx, :,:,1)
    dv = view(dx, :,:,2)

    dx .= 0.0 # zero it out for sanity

    @inline laplacian!(du, u, params.Dᵤ, backend)
    @inline laplacian!(dv, v, params.Dᵥ, backend)

    @inline reaction!(du, dv, u, v, params, backend)
end

function laplacian!(du, u, D, backend::AbstractGrayScott)
    @assert ndims(du) == 2
    @assert ndims(u) == 2
    @assert size(du) == size(u)
    axs = axes(u) # get the index ranges of u along each dimension
    # unfathomably based
    imfilter!(
        get_resource(backend), # specialize the computation based on backend
        du, # output array
        u, # input array
        LAPLACIAN_KERNEL,
        ImageFiltering.NoPad(), # assume arrays are already padded
        (axs[1][begin+1:end-1], axs[2][begin+1:end-1]) # only compute the inner array
    )
    du .*= D
end

const k = StaticKernels.Kernel{(-1:1,-1:1)}(
    @inline w -> 0.25*w[-1,-1] + 0.5*w[0,-1] + 0.25*w[1,-1] + 0.5*w[-1,0] - 3.0*w[0,0] + 0.5*w[1,0] + 0.25*w[-1,1] + 0.5*w[0,1] + 0.25*w[1,1]
)

function laplacian_!(du, u, kernel, D, backend::AbstractGrayScott)
    map!(kernel, du, u)
    du .* D
end
