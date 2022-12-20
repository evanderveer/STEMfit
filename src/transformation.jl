function get_pixel_size(
    uc::UnitCell,
    basis_vector_distances::Tuple{<:Real, <:Real};
    return_two_sizes::Bool = false
)
    ip_pixel_distance = sqrt(+(uc.vector_1 .^ 2 ...))
    oop_pixel_distance = sqrt(+(uc.vector_2 .^ 2 ...))

    ip_pixel_size = basis_vector_distances[1]/ip_pixel_distance
    oop_pixel_size = basis_vector_distances[2]/oop_pixel_distance

    if return_two_sizes
        return (ip_pixel_size, oop_pixel_size)
    end

    if abs(ip_pixel_size-oop_pixel_size)/oop_pixel_size > 0.01
        error("In-plane and out-of-plane pixel sizes too different.
Set return_two_sizes = true to return ip and oop sizes separately.")
    end

    mean([ip_pixel_size, oop_pixel_size])    
end

function transform_positions(
    positions::AbstractMatrix{<:Real},
    pixel_sizes::Tuple{<:Real, <:Real}
)
    Diagonal([pixel_sizes...])*positions
end

function transform_positions(
    positions::AbstractMatrix{<:Real},
    pixel_sizes::AbstractVector{<:Real}
)
    Diagonal(pixel_sizes)*positions
end

function transform_positions(
    positions::AbstractMatrix{<:Real},
    pixel_size::Real
)
    transform_positions(positions, [pixel_size, pixel_size])
end
