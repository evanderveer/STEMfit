struct Gaussian{T<:Real} 
    y0::T
    x0::T
    A::T
    a::T
    b::T
    c::T
end

"""
    get_parameters(
        model::Gaussian
    )
    -> Vector{Float32}

Returns the parameters of the Gaussian *model*.
"""
function get_parameters(
    model::Gaussian
)
    (;y0, x0, A, a, b, c) = model
    [y0, x0, A, a, b, c]
end

#Include the gradient and hessian calculated by Symbolics.jl
#include("gaussian_hessian.jl")
#include("gaussian_gradient.jl")

"""
    intensity(
        model::Gaussian{<:Real},
        x:Real,
        y::Real
    )
    -> Real

Returns the intensity of the Gaussian *model* at the point (*x*, *y*).
"""
function intensity(
    model::Gaussian{<:Real},
    x::Real,
    y::Real
    )
    (;y0, x0, A, a, b, c) = model
    A*exp(-(a * (x - x0)^2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)^2))
end

"""
    intensity(
        model::AbstractVector{<:Gaussian{<:Real}},
        x::Real,
        y::Real
    )
    -> Real

Returns the summed intensity of the vector of Gaussians *model* at the point (*x*, *y*).
"""
function intensity(
    model::AbstractVector{<:Gaussian{T}},
    x::Real,
    y::Real
    ) where {T<:Real}
    sum(intensity(g, x, y) for g in model)::T
end

"""
    gradient(
        model::Gaussian,
        x:Real,
        y::Real
    )
    -> Real

Returns the gradient of the Gaussian *model* at the point (*x*, *y*) with respect to its parameters and *x* & *y*.
"""
function gradient(
    model::Gaussian,
    x::Real,
    y::Real
    )
    (;y0, x0, A, a, b, c) = model
    gaussian_gradient([x, y, x0, y0, A, a, b, c])
end

"""
    gradient(
        model_vector::AbstractVector{Gaussian},
        x:Real,
        y::Real
    )
    -> Real

Returns the summed gradient of the vector of Gaussians *model_vector* at the point (*x*, *y*) and *x* & *y*.
"""
function gradient(
    model_vector::Vector{Gaussian},
    x::Real,
    y::Real
    )
    sum = @MVector zeros(Float32,6)
    for model in model_vector
        (;y0, x0, A, a, b, c) = model
        sum += gaussian_gradient([x, y, x0, y0, A, a, b, c])
    end
    sum
end

"""
    hessian(
        model::Gaussian,
        x:Real,
        y::Real
    )
    -> Real

Returns the hessian of the Gaussian *model* at the point (*x*, *y*) with respect to its parameters and *x* & *y*.
"""
function hessian(
    model::Gaussian,
    x::Real,
    y::Real
    )
    (;y0, x0, A, a, b, c) = model
    gaussian_hessian([x, y, x0, y0, A, a, b, c])
end

"""
    hessian(
        model_vector::AbstractVector{Gaussian},
        x:Real,
        y::Real
    )
    -> Real

Returns the summed hessian of the vector of Gaussians *model_vector* at the point (*x*, *y*) and *x* & *y*.
"""
function hessian(
    model_vector::Vector{Gaussian},
    x::Real,
    y::Real
    )
    sum = @MMatrix zeros(Float32, 6, 6)
    for model in model_vector
        (;y0, x0, A, a, b, c) = model
        sum += gaussian_hessian([x, y, x0, y0, A, a, b, c])
    end
    sum
end