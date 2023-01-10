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

function update_gaussian(
    gaussian::Gaussian{T},
    new_parameters::Union{<:Vector{<:AbstractFloat}, <:Tuple{<:AbstractFloat}}
    ) where {T}
    Gaussian{T}(T.(new_parameters)...) 
end

