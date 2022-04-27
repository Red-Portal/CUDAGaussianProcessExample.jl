
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

using Optim
using Random
using Statistics
using DelimitedFiles
using Distributions

include("gp_cuda_utils.jl")

function load_dataset()
    data = readdlm("housing.csv")
    X    = Array{Float32}(data[:, 1:end-1])
    y    = Array{Float32}(data[:, end])

    n_data      = size(X, 1)
    n_train     = round(Int, n_data*0.9)
    shuffle_idx = Random.shuffle(1:n_data)
    X           = X[shuffle_idx,:]
    y           = y[shuffle_idx]

    X_train     = X[1:n_train,:]
    X_test      = X[n_train+1:end,:]
    y_train     = y[1:n_train]
    y_test      = y[n_train+1:end]

    μ_x = mean(X_train, dims=1)
    σ_x = std( X_train, dims=1)
    μ_y = mean(y_train)
    σ_y = std(y_train)

    X_train .-= μ_x
    X_train ./= σ_x
    X_test  .-= μ_x
    X_test  ./= σ_x

    y_train .-= μ_y
    y_test  .-= μ_y
    y_train ./= σ_y
    y_test  ./= σ_y

    Array(X_train'), Array(X_test'), y_train, y_test
end

function main()
    X_train, X_test, y_train, y_test = load_dataset()

    X_train_dev = cu(X_train)
    X_test_dev  = cu(X_test)
    y_train_dev = cu(y_train)

    N  = size(X_train_dev, 2)
    D  = size(X_train_dev, 1)

    function likelihood(θ_)
        ℓσ     = θ_[1]
        ℓϵ     = θ_[2]
        ℓ²_dev = cu(exp.(θ_[3:end] * 2))
        gp_likelihood(X_train_dev, y_train_dev, exp(ℓσ * 2), exp(ℓϵ * 2), ℓ²_dev)
    end

    function fg!(F,G,θ_)
        if isnothing(G)
            y = likelihood(θ_)
            -y
        else
            y, back = Zygote.pullback(likelihood, θ_)
            ∇like   = back(one(y))[1]
            G[:]    = -Array(∇like) # Gradient is a CuArray
            -y
        end
    end

    function predict_batch(X_pred_dev, θ_)
        σ²     = exp(θ_[1]*2)
        ϵ²     = exp(θ_[2]*2)
        ℓ²_dev = cu(exp.(θ_[3:end] * 2))

        R_train = distance_matrix_gpu(X_train_dev, X_train_dev, ℓ²_dev)
        K_unit  = matern52_gpu(R_train)
        K       = eltype(K_unit)(σ²) * K_unit + eltype(K_unit)(1e-6 + ϵ²) * I
        K_chol  = cholesky(K; check = false)

        R_pred_train      = distance_matrix_gpu(X_pred_dev, X_train_dev, ℓ²_dev)
        K_unit_pred_train = matern52_gpu(R_pred_train)
        K_pred_train      = eltype(K_unit_pred_train)(σ²) * K_unit_pred_train

        μ_f_pred  = Array(K_pred_train * (K_chol \ y_train_dev))
        v         = K_chol.L \ K_pred_train'
        kᵀK⁻¹k    = Array(sum(v .* v, dims = 1)[1, :])
        σ²_f_pred = max.(σ² .+ ϵ² .- kᵀK⁻¹k, eps(eltype(K)))
        μ_f_pred, σ²_f_pred
    end
    
    θ₀  = randn(D + 2)
    opt = optimize(Optim.only_fg!(fg!), θ₀, LBFGS())
    display(opt)
    θ   = Optim.minimizer(opt)

    μ_opt, σ²_opt   = predict_batch(X_test_dev, θ)
    μ_init, σ²_init = predict_batch(X_test_dev, θ₀)

    rmse_opt  = sqrt(mean((μ_opt  - y_test).^2))
    rmse_init = sqrt(mean((μ_init - y_test).^2))
    lpd_opt   = mean(logpdf.(Normal.(μ_opt, sqrt.(σ²_opt)), y_test))
    lpd_init  = mean(logpdf.(Normal.(μ_init, sqrt.(σ²_init)), y_test))
    
    @info("MAP-II Hyperparameter Optimization Result",
          likelihood_before=likelihood(θ₀),
          likelihood_after=likelihood(θ),
          rmse_before=rmse_init,
          rmse_after=rmse_opt,
          lpd_before=lpd_init,
          lpd_after=lpd_opt,
          )
end

