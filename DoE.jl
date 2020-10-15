using Wavelets, Images, FileIO, ImageIO, ImageMagick, QuartzImageIO
using LinearAlgebra, FFTW, SparseArrays, StatsBase
using GLMNet, Plots, Statistics

#---

# img = load("SheppLogan_Phantom.png")
# img = load("brain2.jpeg")
# img = load("skull_base.png")
img = load("SheppLogan_Phantom.png")
img = Gray.(img)
img = imresize(img, ratio =1/(2^3))
print("image sparsiy: ", sum(abs.(img) .> 1e-6) / prod(size(img)))
# img = img[1:64,1:64]
img[img .< 4e-1] .= 0
s = sum(abs.(img) .> 1e-6)


imgx = Float64.(img)
x = dwt(imgx, wavelet(WT.haar))

Gray.(x)
N = prod(size(img))
sum(abs.(x) .> 1e-9) / N

#---
# Get the DWT matrix
l_x, l_y = size(img)
W = spzeros(l_x^2, l_y^2)
z = zeros((l_x, l_y))
for i in 1:l_x
    for j in 1:l_y
        global z .= 0
        global z[i,j] = 1
        global W[:,(i-1)*(l_x) + j] = reshape(dwt(z, wavelet(WT.haar), 3), (l_x^2,1))
    end
end

w = reshape(W * reshape(imgx', (l_x*l_y,1)), size(img))

isapprox(x,w)

#---
# Get the DCT matrix
l_x, l_y = size(img)
D = zeros(l_x^2, l_y^2)
z = zeros((l_x, l_y))
for i in 1:l_x
    for j in 1:l_y
        global z .= 0
        global z[i,j] = 1
        global D[:,(i-1)*(l_x) + j] = reshape(FFTW.dct(z), (l_x^2,1))
    end
end

d = reshape(D * reshape(imgx', (l_x*l_y,1)), size(img))
isapprox(d,FFTW.dct(imgx))

#---
# W = I
undersample_rate = 0.50
p = l_x * l_y
n = Int(floor(undersample_rate * p))

X = sqrt(p) .* D
# β = W * reshape(imgx, (l_x*l_y,1))
β = Float64.(reshape(img, (l_x*l_y,1)))

#---

indices = sample(1:p, n, replace = true)
@views y = X[indices, :] * β .+ randn(n)
y = vec(y)
lambda_range = sqrt(log(p)/n) .* collect(1:50) ./ 20
@views cv = glmnetcv(X[indices,:], y,
            intercept = false, lambda = lambda_range,
            standardize=true)
λ_cv = cv.lambda[argmin(cv.meanloss)]

print(cv)
plot(cv.lambda, cv.meanloss, xaxis = "λ", yaxis = "MSE")

#---
λ_theory = sqrt(log(p) / n)
@views fit = glmnet(X[indices,:], y,
 intercept = false, lambda = [λ_cv],
 standardize = true)

β_hat = fit.betas[:,1]

sqrt(sum(abs.(β_hat - β).^2))
sum(abs.(β_hat - β))

β_hat[β_hat .< 0] .= 0
sqrt(sum(abs.(β_hat - β).^2))
sum(abs.(β_hat - β))

Gray.(reshape(β_hat, size(img)))
Gray.(reshape(β, size(img)))

#---
c = spzeros(l_x, l_y)
c[30:40, 30:40] .= 1 / sqrt(121)
γ = (vec(c)' * β)[1]

@views γ_hat = vec(c)' * β_hat + vec(c)' *
            X[indices,:]'*(y - X[indices,:]*β_hat) ./ n

n_rep = 100

γs = zeros(n_rep)

for i in 1:n_rep
    indices = sample(1:p, n, replace = true)
    @views y .= vec(X[indices, :] * β .+ randn(n) ./ 10)
    @views fit = glmnet(X[indices,:], y,
                        intercept = false, lambda = [λ_cv])
    β_hat .= fit.betas[:,1]
    β_hat[β_hat .< 0] .= 0
    γs[i] = vec(c)' * β_hat .+ vec(c)' * X[indices,:]'*(y .- X[indices,:]*β_hat) ./n
    println(i)
end



#---
using Convex, SCS

#---
α = 0.1

w = Variable(p, Positive())

Σ = X * diag(w) * X
obj = matrixfrac(vec(c), Σ)
P = minimize(obj)
P.constraints += [eigmin(Σ) >= α; sum(w)==1]

solve!(P, () -> SCS.Optimizer(verbose=true))


#---
using JuMP, SCS

#---
α = 0.5
model = Model(SCS.Optimizer)
@variable(model, w[1:p])
@constraint(model, w .>= 0)
@constraint(model, sum(w) == 1)
@SDconstraint(model, X' * diagm(w) * X >= α .* I(p))

#---
#Minimzation with the logdet penalty

function gradient(X, Σ, Σ_inv, c)
    N, _ = size(X)
    ∇ = zeros(N)
    for i in 1:N
        @views ∇[i] = X[i,:]' * (I(N) .- Σ_inv * c * c') * Σ_inv * X[i,:]
    end
    return ∇
end

function simplex_project(w::Float64)
    N = length(w)
    u = sort(w, rev = true)
    v = cumsum(u)
    ρ = findlast((u .+ (1 .- v) ./ (1:N)) .> 0)
    λ = (1 - v[ρ]) / ρ
    w_new = w .+ λ
    w_new[w_new .< 0] .= 0
    w_new .= w_new ./ sum(w_new)
    return w_new
end

function gd(X, w_init, step_size, n_iter)
    w = w_init
    Σ = X' * diagm(w) * X
    for i in 1:n_iter
        w .= w .- step_size .* gradient(X, Σ, inv(Σ), vec(c))
        w .= simplex_project(w)
        @views Σ .= X' * diag(w) * X
        println(i / n_iter)
    end
    return w
end

N, _ = size(X)
gd(X,(ones(N) ./ N), 0.01, 2)

#---
# The LP formulation
using JuMP, GLPK

model = Model(GLPK.Optimizer)

@variable(model, μ[1:N])
@variable(model, z[1:N])
@constraint(model, X' * z .== vec(c))
@constraint(model, z .- μ .<= 0)
@constraint(model, -z .- μ .<= 0)

@objective(model, Min, ones(N)' * μ)

optimize!(model)

#---
using JuMP, Clp

model = Model(Clp.Optimizer)
set_optimizer_attribute(model, "PrimalTolerance", 1e-3)
set_optimizer_attribute(model, "DualTolerance", 1e-4)



@variable(model, μ[1:N] >= 0)
@variable(model, z[1:N])
@constraint(model, X' * z .== vec(c))
@constraint(model, z .- μ .<= 0)
@constraint(model, -z .- μ .<= 0)

@objective(model, Min, ones(N)' * μ)

optimize!(model)

w_opt = value.(μ)
w_opt[w_opt .< 0] .= 0
w_opt .= w_opt ./ sum(w_opt)

@views σ_opt = (vec(c)' * inv(X' * diagm(w_opt) * X) * vec(c))
@views σ_unif = (vec(c)' * inv(X' * diagm(ones(N) ./ N) * X) * vec(c))

@views Σ_opt = X' * diagm(w_opt) * X

eigmin(Σ_opt)
det(Σ_opt)

#---
c = spzeros(l_x, l_y)
c[30:40, 30:40] .= 1 / sqrt(121)
γ = (vec(c)' * β)[1]

@views γ_hat = vec(c)' * β_hat + vec(c)' *
            X[indices,:]'*(y - X[indices,:]*β_hat) ./ n

n_rep = 100

γs_opt = zeros(n_rep)

for i in 1:n_rep
    indices = sample(1:p, w_opt, n, replace = true)
    @views y .= vec(X[indices, :] * β .+ randn(n) ./ 10)
    @views fit = glmnet(X[indices,:], y,
                        intercept = false, lambda = [λ_cv])
    β_hat .= fit.betas[:,1]
    β_hat[β_hat .< 0] .= 0
    γs_opt[i] = vec(c)' * β_hat .+ vec(c)' * X[indices,:]'*(y .- X[indices,:]*β_hat) ./n
    println(i)
end
