
@eval Zygote begin
    import CUDA
    @adjoint function cholesky(Σ::CUDA.CuArray; check = true)
        C = cholesky(Σ, check = check)
        C, function (Δ::NamedTuple)
            issuccess(C) || throw(PosDefException(C.info))
            U, Ū = C.U, Δ.factors

            U_tru = triu(U.data)
            Ū_tru = triu(Ū.data)

            Σ̄ = similar(U.data)
            Σ̄ = mul!(Σ̄, Ū_tru, U_tru')
            Σ̄ = copytri!(Σ̄, 'U')
            Σ̄ = ldiv!(U, Σ̄)
            Σ̄ = CUDA.CUBLAS.trsm!('R', 'U', 'T', 'N', one(eltype(Σ)), U.data, Σ̄)
            Σ̄[diagind(Σ̄)] ./= 2
            return (UpperTriangular(Σ̄),)
        end
    end
end

@kernel function pairwise_distance_kernel!(
    X_dev::CUDA.CuDeviceMatrix,
    Y_dev::CUDA.CuDeviceMatrix,
    ℓ²::CUDA.CuDeviceVector,
    R_out::CUDA.CuDeviceMatrix,
)
    i, j = @index(Global, NTuple)
    n_dims = size(X_dev, 1)

    ∑Δx² = eltype(X_dev)(0.0)
    @inbounds for k = 1:n_dims
        Δx = X_dev[k, i] - Y_dev[k, j]
        ∑Δx² += Δx * Δx / ℓ²[k]
    end
    r = sqrt(∑Δx²)
    @inbounds R_out[i, j] = r
end

@kernel function matern52_kernel!(R_dev::CUDA.CuDeviceMatrix, K_out::CUDA.CuDeviceMatrix)
    i, j = @index(Global, NTuple)
    ϵ = eps(eltype(R_dev))
    @inbounds r = R_dev[i, j]
    s = sqrt(5) * r
    kᵢⱼ = (1 + s + s * s / 3) * exp(-s)
    @inbounds K_out[i, j] = max(kᵢⱼ, ϵ)
end

@kernel function se_kernel!(R_dev::CUDA.CuDeviceMatrix, K_out::CUDA.CuDeviceMatrix)
    i, j = @index(Global, NTuple)
    ϵ = eps(eltype(R_dev))
    @inbounds r = R_dev[i, j]
    kᵢⱼ = exp(r * r / -2)
    @inbounds K_out[i, j] = max(kᵢⱼ, ϵ)
end

@kernel function matern52_derivative_kernel!(
    R_dev::CUDA.CuDeviceMatrix,
    X_dev::CUDA.CuDeviceMatrix,
    k::Int32,
    σ²::Real,
    ℓ²::CUDA.CuDeviceVector,
    ∂K_out::CUDA.CuDeviceMatrix,
)
    i, j = @index(Global, NTuple)

    ϵ = eps(eltype(R_dev))
    @inbounds r   = R_dev[i, j]
    @inbounds Δxₖ = X_dev[k, i] - X_dev[k, j]
    @inbounds ℓ²ₖ = ℓ²[k]
    ∂r∂ℓ²ₖ        = 1 / 2 / max(r, ϵ) * -Δxₖ * Δxₖ / max(ℓ²ₖ * ℓ²ₖ, ϵ)
    s             = sqrt(5) * r
    ∂k∂ℓ²ₖ        = -sqrt(5) / 3 * σ² * ∂r∂ℓ²ₖ * s * (1 + s) * exp(-s)

    @inbounds ∂K_out[i, j] = ∂k∂ℓ²ₖ
end

@kernel function se_derivative_kernel!(
    K_dev::CUDA.CuDeviceMatrix,
    R_dev::CUDA.CuDeviceMatrix,
    X_dev::CUDA.CuDeviceMatrix,
    k::Int32,
    σ²::Real,
    ℓ²::CUDA.CuDeviceVector,
    ∂K_out::CUDA.CuDeviceMatrix,
)
    i, j = @index(Global, NTuple)

    ϵ = eps(eltype(K_dev))
    @inbounds r = R_dev[i, j]
    @inbounds kᵢⱼ = K_dev[i, j]
    @inbounds Δxₖ = X_dev[k, i] - X_dev[k, j]
    @inbounds ℓ²ₖ = ℓ²[k]
    ∂r∂ℓ²ₖ = 1 / 2 / max(r, ϵ) * -Δxₖ * Δxₖ / max(ℓ²ₖ * ℓ²ₖ, ϵ)
    ∂k∂ℓ²ₖ = σ² * ∂r∂ℓ²ₖ * -r * kᵢⱼ
    @inbounds ∂K_out[i, j] = ∂k∂ℓ²ₖ
end

function distance_matrix_gpu(
    X_dev::CUDA.CuArray{<:Real,2},
    Y_dev::CUDA.CuArray{<:Real,2},
    ℓ²::CUDA.CuArray{<:Real,1},
)
    n_data_x = size(X_dev, 2)
    n_data_y = size(Y_dev, 2)
    R_out    = CUDA.zeros(eltype(X_dev), n_data_x, n_data_y)
    device   = KernelAbstractions.get_device(X_dev)
    n_block  = 32
    kernel!  = pairwise_distance_kernel!(device, n_block)
    ev       = kernel!(X_dev, Y_dev, ℓ², R_out, ndrange = (n_data_x, n_data_y))
    wait(ev)
    R_out
end

function matern52_gpu(R_dev::CUDA.CuArray{<:Real,2})
    K_out   = CUDA.zeros(eltype(R_dev), size(R_dev)...)
    device  = KernelAbstractions.get_device(R_dev)
    n_block = 32
    kernel! = matern52_kernel!(device, n_block)
    ev      = kernel!(R_dev, K_out, ndrange = size(K_out))
    wait(ev)
    K_out
end

function se_gpu(R_dev::CUDA.CuArray{<:Real,2})
    K_out   = CUDA.zeros(eltype(R_dev), size(R_dev)...)
    device  = KernelAbstractions.get_device(R_dev)
    n_block = 32
    kernel! = se_kernel!(device, n_block)
    ev      = kernel!(R_dev, K_out, ndrange = size(K_out))
    wait(ev)
    K_out
end

function gram_matern52_derivative_gpu(
    R_dev::CUDA.CuArray{<:Real,2},
    X_dev::CUDA.CuArray{<:Real,2},
    k::Int32,
    σ²::Real,
    ℓ²::CUDA.CuArray{<:Real,1},
)
    n_data  = size(X_dev, 2)
    ∂K      = CUDA.zeros(eltype(R_dev), n_data, n_data)
    device  = KernelAbstractions.get_device(X_dev)
    n_block = 32
    kernel! = matern52_derivative_kernel!(device, n_block)
    ev      = kernel!(R_dev, X_dev, k, σ², ℓ², ∂K, ndrange = (n_data, n_data))
    wait(ev)
    ∂K
end

function gram_se_derivative_gpu(
    K_dev::CUDA.CuArray{<:Real,2},
    R_dev::CUDA.CuArray{<:Real,2},
    X_dev::CUDA.CuArray{<:Real,2},
    k::Int32,
    σ²::Real,
    ℓ²::CUDA.CuArray{<:Real,1},
)
    n_data  = size(X_dev, 2)
    ∂K      = CUDA.zeros(eltype(K_dev), n_data, n_data)
    device  = KernelAbstractions.get_device(X_dev)
    n_block = 32
    kernel! = se_derivative_kernel!(device, n_block)
    ev      = kernel!(K_dev, R_dev, X_dev, k, σ², ℓ², ∂K, ndrange = (n_data, n_data))
    wait(ev)
    ∂K
end

function gp_likelihood(
    X_dev::CUDA.CuArray{<:Real,2},
    y_dev::CUDA.CuArray{<:Real,1},
    σ²::Real,
    ϵ²::Real,
    ℓ²_dev::CUDA.CuArray{<:Real,1},
)
    n_data = size(X_dev, 2)
    R      = distance_matrix_gpu(X_dev, X_dev, ℓ²_dev)
    #K      = se_gpu(R)
    K      = matern52_gpu(R)
    K_ϵ    = eltype(K)(σ²) * K + eltype(K)(ϵ²) * I
    K_chol = cholesky(K_ϵ; check = false)

    if issuccess(K_chol)
        L⁻¹y = K_chol.L \ y_dev
        yᵀΣ⁻¹y = dot(L⁻¹y, L⁻¹y)
        logdet = 2 * sum(log.(Array(diag(K_chol.U))))
        (yᵀΣ⁻¹y + logdet + n_data * log(2 * π)) / -2
    else
        -Inf
    end
end

function trace_product(A, B)
    sum(A .* B')
end

Zygote.@adjoint function gp_likelihood(
    X_dev::CUDA.CuArray{<:Real,2},
    y_dev::CUDA.CuArray{<:Real,1},
    σ²::Real,
    ϵ²::Real,
    ℓ²_dev::CUDA.CuArray{<:Real,1},
)
    n_dims = size(X_dev, 1)
    n_data = size(X_dev, 2)

    R = distance_matrix_gpu(X_dev, X_dev, ℓ²_dev)
    #K      = se_gpu(R)
    K      = matern52_gpu(R)
    K_ϵ    = eltype(K)(σ²) * K + eltype(K)(ϵ²) * I
    K_chol = cholesky(K_ϵ; check = false)

    if issuccess(K_chol)
        L⁻¹y   = K_chol.L \ y_dev
        yᵀΣ⁻¹y = dot(L⁻¹y, L⁻¹y)
        logdet = 2 * sum(log.(Array(diag(K_chol.U))))
        p_f    = (yᵀΣ⁻¹y + logdet + n_data * log(2 * π)) / -2
        p_f,
        function (Δ)
            L⁻¹     = inv(K_chol.L)
            L⁻¹_trl = tril(L⁻¹)
            K⁻¹     = L⁻¹_trl' * L⁻¹_trl
            α       = K_chol.U \ L⁻¹y
            K⁻¹y    = K⁻¹ * y_dev

            ∂p∂y  = -α
            ∂p∂σ² = (dot(K⁻¹y, K * K⁻¹y) - trace_product(K⁻¹, K)) / 2
            ∂p∂ϵ² = (dot(K⁻¹y, K⁻¹y) - tr(K⁻¹)) / 2
            ∂p∂ℓ² = map(1:n_dims) do k
                ∂K∂ℓ²ᵢ     = gram_matern52_derivative_gpu(R, X_dev, Int32(k), σ², ℓ²_dev)
                ∂K∂ℓ²ᵢK⁻¹y = ∂K∂ℓ²ᵢ * K⁻¹y
                (dot(∂K∂ℓ²ᵢK⁻¹y, K⁻¹y) - trace_product(K⁻¹, ∂K∂ℓ²ᵢ)) / 2
            end
            (nothing, Δ * ∂p∂y, Δ * ∂p∂σ², Δ * ∂p∂ϵ², Δ * ∂p∂ℓ²)
        end
    else
        -Inf, function (Δ)
            ∂p∂y = CUDA.zeros{eltype(X_dev)}(n_data)
            ∂p∂σ² = 0
            ∂p∂ϵ² = 0
            ∂p∂ℓ² = CUDA.zeros{eltype(X_dev)}(n_dims)
            (nothing, Δ * ∂p∂y, Δ * ∂p∂σ², Δ * ∂p∂ϵ², Δ * ∂p∂ℓ²)
        end
    end
end
