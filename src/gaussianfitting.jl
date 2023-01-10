#include("lossfunctions.jl")

function get_valid_range(
    center::Tuple,
    image_size::Tuple,
    window_size::Integer
)

    center_round = round.(Int32, center)
    ymin = clamp(center_round[1] - window_size, 1, image_size[1])
    ymax = clamp(center_round[1] + window_size, 1, image_size[1])
    xmin = clamp(center_round[2] - window_size, 1, image_size[2])
    xmax = clamp(center_round[2] + window_size, 1, image_size[2])
    
    xrange = range(xmin, xmax)
    yrange = range(ymin, ymax)
    
    return (yrange, xrange)
end

function optimization_parameters(
    gaussian::Gaussian
) 
    
    parameters = [getfield(gaussian, field) for field in fieldnames(typeof(gaussian))] 
    covariance_matrix = [parameters[4] parameters[5];parameters[5] parameters[6]]
    parameters[4:6] = cholesky(covariance_matrix).L[BitMatrix([1 0;1 1])]
    parameters
end

function convert_optimization_parameters_to_gaussian(
    parameters::AbstractVector{T}
) where T<:AbstractFloat
    if length(parameters) != 6
        throw(ArgumentError("parameters vector must have length 6"))
    end
    L_matrix = zeros(T, 2, 2)
    L_matrix[BitMatrix([1 0;1 1])] = parameters[4:6]

    Gaussian([parameters[1:3]; (L_matrix*L_matrix')[BitMatrix([1 1;0 1])]]...)
end