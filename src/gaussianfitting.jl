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

#include("lossfunctions.jl")

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
    image::AbstractMatrix{<:Gray{<:Real}}
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

function fit_A(
    u::AbstractVector{T},
    p::GaussianParameters
) where T
    
    #Calculate the loss at three different A values (0, 0.5, 1)
    parameters = [T.([u[1:2]..., i, u[4:6]...]) for i in 0:0.5:1]
    res = [STEMfit.loss_function(parameter, p) for parameter in parameters]

    #Find the least squares fit to a parabola
    A = T.([2 -4 2;-3 4 -1;1 0 0])
    f = A*res

    #Calculate the minimum of the fitted parabola
    f_min = -f[2]/(2*f[1])

    #Update the gaussian model
    new_g = STEMfit.convert_optimization_parameters_to_gaussian([u[1:2]; f_min; u[4:end]])
    p.image_model.gaussians[p.gaussian_number] = new_g
end

function fit_linear(
    u::AbstractVector{T},
    p::GaussianParameters,
    parameter_to_fit::Symbol;
    step_size::Real = 0.1 
) where T
    
    #Calculate the loss at three different parameter values
    parameter_position = findall(x->x==parameter_to_fit, (:y0,:x0,:A,:a,:b,:c))
    update_vector = zeros(T, 6)
    res = Vector{T}(undef, 3)
    for (i,val) in enumerate((-step_size, 0, step_size))
        update_vector[parameter_position] .= val
        parameter = u .+ T.(update_vector)
        res[i] = loss_function(parameter, p)
    end

    #Find the least squares fit to a parabola
    values = u[parameter_position] .+ (-step_size, 0, step_size)
    A = T.([values.^2 values ones(T, 3)])
    f = inv(A)*res

    #Calculate the minimum of the fitted parabola
    f_min = -f[2]/(2*f[1])

    #Update the gaussian model
    u[parameter_position] .= f_min
    new_g = STEMfit.convert_optimization_parameters_to_gaussian(u)
    p.image_model.gaussians[p.gaussian_number] = new_g
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

residual(im1, im2) = Float64(sum((im1 .- im2) .^ 2))

model_unit_cell_to_window_size(model::ImageModel) = 0.75* maximum([norm(model.unit_cell.vector_1), 
                                                                   norm(model.unit_cell.vector_2)])