function get_pixel_size(
    uc::UnitCell,
    basis_vector_distances::Tuple{<:Real, <:Real};
    return_two_sizes::Bool = false
)
    vec_1_pixel_distance = norm(uc.vector_1)
    vec_2_pixel_distance = norm(uc.vector_2)

    vec_1_pixel_size = basis_vector_distances[1]/vec_1_pixel_distance
    vec_2_pixel_size = basis_vector_distances[2]/vec_2_pixel_distance

    if return_two_sizes
        return (vec_1_pixel_size, vec_2_pixel_size)
    end

    if abs(vec_1_pixel_size-vec_2_pixel_size)/vec_2_pixel_size > 0.01
        error("In-plane and out-of-plane pixel sizes too different.
Set return_two_sizes = true to return horizontal and vertical sizes separately.")
    end

    mean([vec_1_pixel_size, vec_2_pixel_size])    
end

function get_pixel_size(
    reference_latt_param::AbstractMatrix{<:Real},
    basis_vector_distances::Tuple{<:Real, <:Real};
    return_two_sizes::Bool = false
)
    pixel_distances = mean(reference_latt_param, dims=2)

    pixel_sizes = basis_vector_distances ./ pixel_distances

    if return_two_sizes
        return pixel_sizes
    end

    if abs(-(pixel_sizes...))/pixel_sizes[1] > 0.01
        error('\n' * """In-plane and out-of-plane pixel sizes too different.
                        Set return_two_sizes = true to return horizontal and vertical sizes separately.""")
    end

    mean(pixel_sizes)    
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
