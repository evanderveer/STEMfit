#
#File: transformation.jl
#Author: Ewout van der Veer
#
#Description:
# Functions for transforming from px to physical units.
#

"""
    get_pixel_size(
        reference_latt_param::AbstractMatrix{<:Real},
        basis_vector_distances::Tuple{<:Real, <:Real};
        return_two_sizes::Bool = false
    )

Calculates the pixel size based on a known unit cell
"""
function get_pixel_size(
    uc::UnitCell,
    basis_vector_distances::Tuple{<:Real, <:Real};
    return_two_sizes::Bool = false
)
    pixel_distances = norm.((uc.vector_1, uc.vector_2))
    basis_vector_distances ./ pixel_distances
end

"""
    get_pixel_size(
        reference_latt_param::AbstractMatrix{<:Real},
        basis_vector_distances::Tuple{<:Real, <:Real};
        return_two_sizes::Bool = false
    )

Calculates the pixel size based on a reference
"""
function get_pixel_size(
    reference_latt_param::AbstractMatrix{<:Real},
    basis_vector_distances::Tuple{<:Real, <:Real};
    return_two_sizes::Bool = true
)
    pixel_distances = mean(reference_latt_param, dims=2)
    basis_vector_distances ./ pixel_distances  
end

"""
    transform_positions(
        positions::AbstractMatrix{<:Real},
        pixel_sizes::AbstractVector{<:Real}
    )
    -> AbstractMatrix

Transforms the atomic positions based on the given pixel sizes
"""
function transform_positions(
    positions::AbstractMatrix{<:Real},
    pixel_sizes::AbstractVector{<:Real}
)
    if length(pixel_sizes) != 2
        throw(ArgumentError("pixel_sizes should have a length of 2"))
    end
    Diagonal(pixel_sizes)*positions
end

"""
    transform_positions(
        positions::AbstractMatrix{<:Real},
        pixel_sizes::Tuple{<:Real, <:Real}
    )
    -> AbstractMatrix

Transforms the atomic positions based on the given pixel sizes
"""
function transform_positions(
    positions::AbstractMatrix{<:Real},
    pixel_sizes::Tuple{<:Real, <:Real}
)
    transform_positions(positions, [pixel_sizes...])
end

"""
    transform_positions(
        positions::AbstractMatrix{<:Real},
        pixel_sizes::Real
    )
    -> AbstractMatrix

Transforms the atomic positions based on the given pixel sizes
"""
function transform_positions(
    positions::AbstractMatrix{<:Real},
    pixel_sizes::Real
)
    transform_positions(positions, [pixel_sizes, pixel_sizes])
end
