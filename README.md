
# A Minimal Differentiable and CUDA-Compatible Gaussian Process 

This is a minimal implementation of Gaussian processes that runs on [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and supports [Zygote](https://github.com/FluxML/Zygote.jl).
Currently, there is not proper way to implement Gaussian processes on Julia in a way that supports both GPUs and differentiation.
This package is a minimal attempt to fill the void in the meantime.
For a detailed discussion, see [this blog post]()

## Usage
This package is more like a code snippet and not a proper "package".
Instead, please copy `gp_cuda_utils.jl` to your project and use it or modify for your use case.

Currently, the implementation only has two types of kernels: squared exponential and Matern 5/2 both with automatic relevance determination.

Detailed usage can be found in `main.jl`

## Installation

```sh
git clone CUDAGaussianProcessExample.jl
cd CUDAGaussianProcessExample.jl
```
```julia
> . activate
> instantiate
include("main.jl")
main()
```

## Example

`main.jl` contains an example where MAP-II hyperparameter inference is done using [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) on the Boston housing dataset.
Here is an example result:

```
┌ Info: MAP-II Hyperparameter Optimization Result
│   likelihood_before = -544.3303199616416
│   likelihood_after = -116.86849745187607
│   rmse_before = 0.60338885f0
│   rmse_after = 0.3102568f0
│   lpd_before = -0.8926057396811591
└   lpd_after = -0.16185267732364805
```


## Cholesky Failure
When the Cholesky fails, the current implementation does not throw.
Instead, it spits a `-Inf` for the likelihood and `CUDA.zeros` arrays for the gradients.
