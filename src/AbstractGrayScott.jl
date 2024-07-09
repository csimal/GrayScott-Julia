
"""
    AbstractGrayScott

The abstract type for backends to compute the Gray-Scott ODE.
"""
abstract type AbstractGrayScott end

initial_state(init_cond, ::AbstractGrayScott) = init_cond

function allocate_output(init_cond, opts::GrayScottOptions, ::AbstractGrayScott)
    zeros(opts.num_output_steps, size(init_cond)...)
end

# by default, the convolution will be computed on a single core
get_resource(::AbstractGrayScott) = CPU1()

const LAPLACIAN_KERNEL = centered([
    0.25 0.5 0.25;
    0.5 -3.0 0.5;
    0.25 0.5 0.25
])

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
