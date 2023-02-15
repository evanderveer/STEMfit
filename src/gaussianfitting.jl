struct ParameterSet
    image_model
    subimages
    ranges
    neighbor_indices
    ParameterSet(image_model, subimages, ranges, neighbor_indices) = begin
        length(subimages) == 
        length(ranges) == 
        length(neighbor_indices) == 
        length(image_model.gaussians) ||
        throw(ArgumentError('\n'*"""the number of subimages, ranges and neighbor indices 
                             must be the same and equal to the number of gaussians in the model"""))
    new(image_model, subimages, ranges, neighbor_indices)
    end
end

struct GaussianParameters
    gaussian_number
    image_model
    subimage
    range
    neighbor_indices
    GaussianParameters(
        parameter_set::ParameterSet, 
        gaussian::Integer
    ) = begin 
        gaussian > length(parameter_set.subimages) && 
            throw(ArgumentError("there are <"*gaussian*" gaussians in the model"))
        new(
        gaussian,
        parameter_set.image_model,
        parameter_set.subimages[gaussian],
        parameter_set.ranges[gaussian],
        parameter_set.neighbor_indices[gaussian]
        )
    end
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
    image_model::ImageModel{T},
    image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}
) where T
    window_size = model_unit_cell_to_window_size(image_model)

    ranges = [get_valid_range(
                                (gaussian.y0,gaussian.x0), 
                                image_model.model_size, 
                                window_size
                             )
                             for gaussian in image_model.gaussians]

    subimages = [image[range...] for range in ranges];

    neighbor_indices = [knn(
                            image_model.atom_tree, 
                            [gaussian.y0, gaussian.x0], 5
                            )[1] for gaussian in image_model.gaussians]

    ParameterSet(image_model, subimages, ranges, neighbor_indices)
end

function optimization_parameters(
    gaussian::Gaussian{T}
) where {T<:Real}
    
    (; y0, x0, A, a, b, c) = gaussian
    covariance_matrix = [a b;b c]
    (a_t,b_t,c_t) = cholesky(Hermitian(covariance_matrix)).L[BitMatrix([1 0;1 1])]
    [y0, x0, A, a_t, b_t, c_t]
end

function convert_optimization_parameters_to_gaussian(
    parameters::AbstractVector{T}
) where T<:Real
    if length(parameters) != 6
        throw(ArgumentError("parameters vector must have length 6"))
    end
    L_matrix = zeros(T, 2, 2)
    L_matrix[BitMatrix([1 0;1 1])] = parameters[4:6]

    Gaussian([parameters[1:3]; (L_matrix*L_matrix')[BitMatrix([1 1;0 1])]]...)
end

function loss_function(
    u::AbstractVector{T},
    p::AbstractVector{Any}
) where {T<:Real}
    (subimage, image_model, gaussian_number, neighbor_indices, range) = p

    image_model.gaussians[gaussian_number] = convert_optimization_parameters_to_gaussian(u)
    submodel = produce_image(image_model, range..., neighbor_indices)

    T(residual(submodel, subimage))
end

function loss_function(
    u::AbstractVector{T},
    p::GaussianParameters
) where {T<:Real}
    p.image_model.gaussians[p.gaussian_number] = 
        convert_optimization_parameters_to_gaussian(u)
    submodel = produce_image(p.image_model, p.range..., p.neighbor_indices)
    
    T(residual(submodel, p.subimage))
end

function fit_optim!(image_model, image; tolerance = 0.001, n = nothing, multithreaded=true, method=BFGS())
    parameter_set = construct_parameter_set(image_model, image)
    gaussian_parameters = [GaussianParameters(parameter_set, i) 
                                for i in 1:length(image_model.gaussians)]
    initial_parameters = [optimization_parameters(gaussian) 
                                for gaussian in image_model.gaussians];
    optf = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())
    if n === nothing; n = length(image_model.gaussians); end 
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

residual(im1::AbstractMatrix{<:Real}, im2::AbstractMatrix{<:Real}) = sum((im1 .- im2) .^ 2)
residual(im1::AbstractMatrix{<:Gray{T}}, im2::AbstractMatrix{<:Real}) where {T<:Real} = sum((T.(im1) .- im2) .^ 2)
residual(im1::AbstractMatrix{<:Real}, im2::AbstractMatrix{<:Gray{T}}) where {T<:Real} = sum((im1 .- T.(im2)) .^ 2)
residual(im1::AbstractMatrix{<:Gray{T}}, im2::AbstractMatrix{<:Gray{V}}) where {T<:Real, V<:Real} = sum((T.(im1) .- V.(im2)) .^ 2)


model_unit_cell_to_window_size(model::ImageModel) = 0.75* maximum([norm(model.unit_cell.vector_1), 
                                                                   norm(model.unit_cell.vector_2)])