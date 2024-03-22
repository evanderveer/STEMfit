"""
File: strain.jl
Author: Ewout van der Veer

Description:
Functions for calculating the local strain from atomic positions. 
"""

@enum LAYER_DIRECTION y x

"""
    function get_strain_from_lattice_parameters(
        lattice_parameters::AbstractMatrix{T},
        bulk_lattice_parameter_dict::Dict{<:Integer, <:Tuple},
        layer_boundaries::AbstractVector{<:Integer},
        layer_direction::LAYER_DIRECTION = y    
    ) where {T<:Real} -> Matrix{T}

    Calculates the strain from the local lattice parameters values in `lattice_parameters`,
    the bulk lattice_parameters `bulk_lattice_parameter_dict` and layer assignments. 
    `bulk_lattice_parameter_dict` is a dictionary whose keys correspond to the layer indices
    in `layer_assignments` and whose values are tuples of bulk lattice parameters in the same
    directions as the two rows of `lattice_parameters`.
"""
function calculate_true_strain(
    results::Results,
    bulk_lattice_parameter_dict::Dict{<:Integer, <:Tuple},
    layer_boundaries::AbstractVector{<:Integer},
    layer_direction::LAYER_DIRECTION = y    
    ) 
    #Make sure we have the right number of layers
    if length(layer_boundaries) != length(bulk_lattice_parameter_dict) - 1
        throw(ArgumentError("wrong number of layer boundaries provided"))
    end

    layer_numbers = layer_assignments(
                                        results.atom_parameters, 
                                        layer_boundaries,
                                        layer_direction
                                        )

    results.strain = similar(results.lattice_parameters)
    for (i, col) in enumerate(eachcol(results.lattice_parameters))
        if layer_numbers[i] == 0
            results.strain[:, i] = zeros(T, 2)
            continue
        end

        strain = col ./ bulk_lattice_parameter_dict[layer_numbers[i]] .- 1
        results.strain[:, i] = strain
    end
    
end

"""
    layer_assignments(
        atom_positions::AbstractMatrix{T}, 
        layer_boundaries::AbstractVector
    )
    
    Creates a vector of layer indices of each atom in `atom_positions` based on 
    the `layer_boundaries`. Assumes that layer boundaries are horizontal. The values
    in `layer_boundaries` are the y-values of the boundaries in the image.
"""
function layer_assignments(
    atom_parameters::AtomParameters{T}, 
    layer_boundaries::AbstractVector,
    layer_direction::LAYER_DIRECTION
    ) where T

    atom_positions = atom_parameters.centroids

    #Are the layers horizontal or vertical?
    row = layer_direction == y ? 1 : 2

    layer_boundaries = [zero(T), T.(layer_boundaries)..., maximum(atom_positions[row, :])]
    layer_filters = []
    for boundary in eachindex(layer_boundaries[2:end])
        push!(layer_filters, layer_boundaries[boundary] .<
                             atom_positions[row, :] .<= 
                             layer_boundaries[boundary + 1]) 
    end

    #Give each layer a new index incrementally
    layer_assignments = sum(filter .* i for (i, filter) in enumerate(layer_filters), dims=2)

    layer_assignments
end