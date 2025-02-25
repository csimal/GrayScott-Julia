# GrayScott-Julia

A modular implementation of the 2D Gray-Scott model simulation in Julia.

This package aims to provide illustrative implementations of a typical semilinear PDE using Julia, as a way to showcase various common optimizations.

## Backends

The simulation of the Gray-Scott PDE is provided by multiple independent backends. The current list of implemented backends is as follows

* `SimpleGrayScott` A baseline single threaded implementation with no specific optimizations.
* `AdvancedGrayScott` A slightly modified version of `SimpleGrayScott` to use some low-hanging optimizations.
* `TurboGrayScott` A single threaded implementation using the LoopVectorization package for autovectorization.
* `ThreadedGrayScott` A multi-threaded backend using the standard library.
* `DistributedGrayScott` A multi-processor backend using the standard library.
* `CUDAGrayScott` A CUDA specific backend using the CUDA.jl package
* `GPUGrayScott` A vendor agnostic GPU backend using the KernelAbstractions.jl package