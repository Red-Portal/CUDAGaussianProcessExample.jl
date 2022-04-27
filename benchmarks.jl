
#=
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
=## 

using BenchmarkTools
using AbstractGPs
using ProgressMeter
using Plots
using Statistics
using Test

matern_kernel(α², logℓ) =
    α² * (KernelFunctions.Matern52Kernel() ∘ KernelFunctions.ARDTransform(@. exp(-logℓ)))

likelihood_cpu(X, θ) = begin
    N      = size(X, 2)
    ℓσ     = θ[1]
    ℓϵ     = θ[2]
    y      = θ[3:2+N]
    logℓ   = θ[3+N:end]
    kernel = matern_kernel(exp(ℓσ * 2), logℓ)
    gp     = AbstractGPs.GP(kernel)
    fx     = gp(X, exp(ℓϵ * 2))
    logpdf(fx, y)
end

likelihood_gpu(X_dev, θ) = begin
    N  = size(X_dev, 2)
    ℓσ = θ[1]
    ℓϵ = θ[2]
    y  = cu(θ[3:2+N])
    ℓ² = cu(exp.(θ[3+N:end] * 2))
    gp_likelihood(X_dev, y, exp(ℓσ * 2), exp(ℓϵ * 2), ℓ²)
end

gradient_cpu(X, θ) = Zygote.gradient(θ_ -> likelihood_cpu(X, θ_), θ)[1]
gradient_gpu(X, θ) = Zygote.gradient(θ_ -> likelihood_gpu(X, θ_), θ)[1]

Test.@testset "GPU Gaussian process numerical accuracy test" begin
    N     = 128
    D     = 16
    X     = randn(Float32, D, N)
    X_dev = cu(X)
    θ     = randn(Float32, N + D + 2)

    @test likelihood_cpu(X, θ) ≈ likelihood_gpu(X_dev, θ)          atol=1e-4
    @test norm(gradient_cpu(X, θ) - gradient_gpu(X_dev, θ)) ≈ 0.0  atol=1e-4
end

function benchmark()
    CUDA.allowscalar(true)

    D     = 16
    N     = 2048
    X     = randn(Float32, D, N)
    X_dev = cu(X)
    θ     = randn(Float32, N + D + 2)

    bench_likelihood_cpu = @benchmarkable likelihood_cpu(data.X, data.θ) setup = (data = (X = $X,     θ = $θ))
    bench_likelihood_gpu = @benchmarkable likelihood_gpu(data.X, data.θ) setup = (data = (X = $X_dev, θ = $θ))
    bench_gradient_cpu   = @benchmarkable gradient_cpu(data.X, data.θ)   setup = (data = (X = $X,     θ = $θ))
    bench_gradient_gpu   = @benchmarkable gradient_gpu(data.X, data.θ)   setup = (data = (X = $X_dev, θ = $θ))

    @info("Gaussian Process Likelihood Evaluation",
          device="CPU", n_data=N, n_dims=D)
    display(run(bench_likelihood_cpu, samples = 1024, seconds=30))

    @info("Gaussian Process Likelihood Evaluation",
          device="GPU", n_data=N, n_dims=D)
    display(run(bench_likelihood_gpu, samples = 1024, seconds=30))

    @info("Gaussian Process Gradient Evaluation",
          device="CPU", n_data=N, n_dims=D)
    display(run(bench_gradient_cpu, samples = 1024, seconds=30))

    @info("Gaussian Process Gradient Evaluation",
          device="GPU", n_data=N, n_dims=D)
    display(run(bench_gradient_gpu, samples = 1024, seconds=30))
end

function scalability()
    CUDA.allowscalar(true)

    Ns = 2 .^(6:1:12)
    ts = @showprogress map(Ns) do N
        D     = 16
        X     = randn(Float32, D, N)
        X_dev = cu(X)
        θ     = randn(Float32, N + D + 2)

        likelihood_bench_cpu = @benchmarkable likelihood_cpu(data.X, data.θ) setup = (data = (X = $X, θ = $θ))
        likelihood_bench_gpu = @benchmarkable likelihood_gpu(data.X, data.θ) setup = (data = (X = $X_dev, θ = $θ))
        gradient_bench_cpu   = @benchmarkable gradient_cpu(data.X, data.θ)   setup = (data = (X = $X, θ = $θ))
        gradient_bench_gpu   = @benchmarkable gradient_gpu(data.X, data.θ)   setup = (data = (X = $X_dev, θ = $θ))
        likelihood_res_cpu  = run(likelihood_bench_cpu, samples = 32, seconds=8)
        likelihood_res_gpu  = run(likelihood_bench_gpu, samples = 32, seconds=8)
        gradient_res_cpu    = run(gradient_bench_cpu, samples = 32, seconds=8)
        gradient_res_gpu    = run(gradient_bench_gpu, samples = 32, seconds=8)

        likelihood_stat_cpu = quantile(likelihood_res_cpu.times, [0.5, 0.1, 0.9])
        likelihood_stat_gpu = quantile(likelihood_res_gpu.times, [0.5, 0.1, 0.9])
        gradient_stat_cpu   = quantile(gradient_res_cpu.times,   [0.5, 0.1, 0.9])
        gradient_stat_gpu   = quantile(gradient_res_gpu.times,   [0.5, 0.1, 0.9])

        (likelihood_cpu = likelihood_stat_cpu,
         likelihood_gpu = likelihood_stat_gpu,
         gradient_cpu   = gradient_stat_cpu,
         gradient_gpu   = gradient_stat_gpu,
         )
    end
    likelihood_cpu_meds = [t.likelihood_cpu[1] for t ∈ ts] / 1e+9
    likelihood_cpu_err⁺ = [abs(t.likelihood_cpu[2] - t.likelihood_cpu[1]) for t ∈ ts] / 1e+9
    likelihood_cpu_err⁻ = [abs(t.likelihood_cpu[3] - t.likelihood_cpu[1]) for t ∈ ts] / 1e+9

    likelihood_gpu_meds = [t.likelihood_gpu[1] for t ∈ ts] / 1e+9
    likelihood_gpu_err⁺ = [abs(t.likelihood_gpu[2] - t.likelihood_gpu[1]) for t ∈ ts] / 1e+9
    likelihood_gpu_err⁻ = [abs(t.likelihood_gpu[3] - t.likelihood_gpu[1]) for t ∈ ts] / 1e+9

    gradient_cpu_meds = [t.gradient_cpu[1] for t ∈ ts] / 1e+9
    gradient_cpu_err⁺ = [abs(t.gradient_cpu[2] - t.gradient_cpu[1]) for t ∈ ts] / 1e+9
    gradient_cpu_err⁻ = [abs(t.gradient_cpu[3] - t.gradient_cpu[1]) for t ∈ ts] / 1e+9

    gradient_gpu_meds = [t.gradient_gpu[1] for t ∈ ts] / 1e+9
    gradient_gpu_err⁺ = [abs(t.gradient_gpu[2] - t.gradient_gpu[1]) for t ∈ ts] / 1e+9
    gradient_gpu_err⁻ = [abs(t.gradient_gpu[3] - t.gradient_gpu[1]) for t ∈ ts] / 1e+9


    p1 = Plots.plot(Ns, likelihood_cpu_meds,
                    yerr=(likelihood_cpu_err⁺,likelihood_cpu_err⁻,),
                    markerstrokecolor=:auto,
                    xscale=:log10,
                    yscale=:log10,
                    xlabel="N",
                    ylabel="Time (sec)",
                    label="CPU (8 threads)")
    Plots.plot!(p1, Ns, likelihood_gpu_meds,
                yerr=(likelihood_gpu_err⁺,likelihood_gpu_err⁻,),
                markerstrokecolor=:auto,
                title="Likelihood",
                legend=:bottomright,
                xscale=:log10,
                yscale=:log10,
                xlabel="N",
                ylabel="Time (sec)",
                label="GPU (GTX 1050)")

    p2 = Plots.plot(Ns, gradient_cpu_meds,
                    yerr=(gradient_cpu_err⁺,gradient_cpu_err⁻,),
                    markerstrokecolor=:auto,
                    xscale=:log10,
                    yscale=:log10,
                    xlabel="N",
                    ylabel="Time (sec)",
                    label="CPU (8 threads)")
    Plots.plot!(p2, Ns, gradient_gpu_meds,
                yerr=(gradient_gpu_err⁺,gradient_gpu_err⁻,),
                markerstrokecolor=:auto,
                title="Gradient",
                xscale=:log10,
                yscale=:log10,
                legend=:bottomright,
                xlabel="N",
                ylabel="Time (sec)",
                label="GPU (GTX 1050)")
    Plots.plot(p1, p2, layout=(1,2))
end
