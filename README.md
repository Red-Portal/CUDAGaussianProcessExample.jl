
# A Minimal Differentiable and CUDA-Compatible Gaussian Process 

This is a minimal implementation of Gaussian processes that runs on [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and supports [Zygote](https://github.com/FluxML/Zygote.jl).
Currently, there is not proper way to implement Gaussian processes on Julia in a way that supports both GPUs and differentiation.
This package is a minimal attempt to fill the void in the meantime.
For a detailed discussion, see [this blog post](https://krkim.me/blog/2022/gp_cuda)

## Usage
The main routines are in `gp_cuda_utils.jl`.
This package is more like a code snippet and not a proper "package".
Pplease copy `gp_cuda_utils.jl` to your project and use it or modify it to your use.

Currently, the implementation only has two types of kernels: squared exponential and Matern 5/2 both with automatic relevance determination.

Detailed usage can be found in `main.jl`

## Installation

As usual, for download the package as:
```sh
git clone CUDAGaussianProcessExample.jl
cd CUDAGaussianProcessExample.jl
```

Within the Julia interpreter, enter:
```julia
> . activate
> instantiate
include("main.jl")
main()
```
The results in the blog post are reproducible using `benchmark.jl`

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

## License

```
MIT License

Copyright (c) 2022 Kyurae Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
