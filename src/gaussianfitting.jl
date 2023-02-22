struct GaussianParameters{T,U,V,Y}
    index::Int32
    image_model::ImageModel{T,U,V,Y}
    subimage::AbstractMatrix{T}
    subbackground::AbstractMatrix{T}
    neighbor_image::AbstractMatrix{T}
    range::Tuple{UnitRange, UnitRange}
end

function get_valid_range(
    center::Tuple,
    image_size::Tuple,
    window_size::Real
)

    center_rounded = round.(Int32, center)
    window_size_rounded = round.(Int32, window_size)
    ymin = clamp(center_rounded[1] - window_size_rounded, 1, image_size[1])
    ymax = clamp(center_rounded[1] + window_size_rounded, 1, image_size[1])
    xmin = clamp(center_rounded[2] - window_size_rounded, 1, image_size[2])
    xmax = clamp(center_rounded[2] + window_size_rounded, 1, image_size[2])
    
    xrange = range(xmin, xmax)
    yrange = range(ymin, ymax)
    
    (yrange, xrange)
end

function construct_parameter_set(
    image_model::ImageModel{T,U,V,Y},
    image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}
) where {T,U,V,Y}
    window_size = model_unit_cell_to_window_size(image_model)

    ranges = [get_valid_range(
                                (gaussian[1], gaussian[2]), 
                                size(image_model.background), 
                                window_size
                             )
                             for gaussian in image_model.gaussian_parameters]
    [GaussianParameters(
        Int32(idx),
        image_model,
        Y.(image[range...]),
        image_model.background[range...],
        produce_image(image_model, range..., Int32(idx)),
        range
    ) for (idx,range) in enumerate(ranges)]

    
end

function neighbor_indices(
    image_model,
    range,
    index
)
    indices_full = unique!(sort!(vec(
        image_model.nearest_neighbor_indices_tensor[range..., :]
                    )))
    indices_full[indices_full .!= index]
end

function gaussian_to_optimization(
    gaussian_parameters::AbstractVector{T}
) where {T<:Real}
    
    (y0, x0, A, a, b, c) = gaussian_parameters
    covariance_matrix = [a b;b c]
    (a_t,b_t,c_t) = cholesky(Hermitian(covariance_matrix)).L[BitMatrix([1 0;1 1])]
    SVector{6, T}(y0, x0, A, a_t, b_t, c_t)
end

function optimization_to_gaussian(
    optimization_parameters::AbstractVector{T}
) where T<:Real
    if length(optimization_parameters) != 6
        throw(ArgumentError("parameters vector must have length 6"))
    end
    L_matrix = zeros(T, 2, 2)
    L_matrix[BitMatrix([1 0;1 1])] = optimization_parameters[4:6]

    SVector{6, T}([optimization_parameters[1:3]; (L_matrix*L_matrix')[BitMatrix([1 1;0 1])]])
end

"""
function loss_function(
    u::SVector{6, T},
    p::GaussianParameters
) where {T<:Real}
    p.image_model.gaussian_parameters[p.gaussian_number] = 
        SVector{6,T}(optimization_to_gaussian(u))
    println(u[1])
    submodel = produce_image(p.image_model, p.range...)
    T(residual(submodel, p.subimage))
end
"""

function loss_function(
    u::AbstractVector{T},
    p::GaussianParameters
) where {T<:Real}

    u_g = optimization_to_gaussian(u)
    submodel = produce_image(u_g, p)
    T(residual(submodel, p.sub_image))
end

function fit!(image_model, image; tolerance = 0.001, n = nothing, multithreaded=true, method=BFGS())
    gaussian_parameters = construct_parameter_set(image_model, image)
    initial_parameters = gaussian_to_optimization.(image_model.gaussian_parameters) 
    optf = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())
return (gaussian_parameters, initial_parameters)
    if n === nothing; n = length(initial_parameters); end 
    println("Fitting " * string(n) * " Gaussian functions")

    if multithreaded; fit_gaussians(initial_parameters, gaussian_parameters, optf, n, tolerance, method)
    else; fit_gaussians_st(initial_parameters, gaussian_parameters, optf, n, tolerance, method); end
end

function fit_gaussians(us, ps, optf, n, x_tol, method)

    Threads.@threads for i in 1:n
        prob = OptimizationProblem(optf, us[i], ps[i])
        try
            solve(prob, method, x_tol=x_tol)
        catch 
            solve(prob, method, x_tol=x_tol)
        end
    end
end

function fit_gaussians_st(us, ps, optf, n, x_tol, method)

    Threads.@threads for i in 1:n
        prob = OptimizationProblem(optf, us[i], ps[i])
        try
            solve(prob, method, x_tol=x_tol)
        catch 
            solve(prob, method, x_tol=x_tol)
        end
    end
end

residual(im1::AbstractMatrix{<:Real}, im2::AbstractMatrix{<:Real}) = sum((im1 .- im2) .^ 2)
residual(im1::AbstractMatrix{<:Gray{T}}, im2::AbstractMatrix{<:Real}) where {T<:Real} = sum((T.(im1) .- im2) .^ 2)
residual(im1::AbstractMatrix{<:Real}, im2::AbstractMatrix{<:Gray{T}}) where {T<:Real} = sum((im1 .- T.(im2)) .^ 2)
residual
(im1::AbstractMatrix{<:Gray{T}}, im2::AbstractMatrix{<:Gray{V}}) where {T<:Real, V<:Real} = sum((T.(im1) .- V.(im2)) .^ 2)


model_unit_cell_to_window_size(model::ImageModel) = 0.75* maximum([norm(model.unit_cell.vector_1), 
                                                                   norm(model.unit_cell.vector_2)])

function produce_image(
    u::AbstractVector{T},
    p::GaussianParameters{U}
) where {T, U}

    (yrange, xrange) = p.range
    image_size = (yrange.stop - yrange.start + 1, xrange.stop - xrange.start + 1)
    output_image = zeros(T, image_size...)
    slice = zeros(T, image_size...)
    fill_image!(
        output_image, 
        slice, 
        p.sub_neighbor_tensor, 
        p.image_model.gaussian_parameters,
        yrange,
        xrange
        )
    p.sub_background[yrange, xrange] .+ output_image
end