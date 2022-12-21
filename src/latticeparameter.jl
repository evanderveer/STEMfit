"""
    calculate_lattice_parameters(
        positions::AbstractMatrix{T},
        unit_cell::UnitCell[,
        tolerance::Real = 0.2,
        strictness::Integer = 4]
    ) where {T<:Real} -> Matrix{T}

Calculates the local lattice parameter for each atom in `positions` along the 
basis vectors of `unit_cell`.

`tolerance` determines the maximum deviation of a neighboring atom from its expected
position to be considered a valid neighbor. `strictness` is the number of valid 
neighbors required to calculate the lattice parameter (4 <= `strictness` <= 8).

Returns a matrix of lattice parameters. The first row of the matrix are the lattice
parameters along the first basis vector, the second row are the lattice parameters
along the second basis vector. 
"""
function calculate_lattice_parameters(
    positions::AbstractMatrix{T},
    unit_cell::UnitCell;
    tolerance::Real = 0.2,
    strictness::Integer = 4
) where {T<:Real}

    if strictness > 8 || strictness < 4
        throw(ArgumentError("Strictness must be between 4 and 8"))
    end
    vec_1 = unit_cell.vector_1
    vec_2 = unit_cell.vector_2
    inv_matrix = inv([vec_2 vec_1])
    transformed_positions = inv_matrix * positions

    tree = KDTree(transformed_positions)

    #Matrix is nearest neighbor positions in transformed space
    nn_matrix = [-1 -1 -1 0 0 1 1 1; -1 0 1 -1 1 -1 0 1]
    
    latt_param_matrix = similar(positions)
    
    for i in 1:size(transformed_positions)[2]

        #Get nearest neighbor indices and distances
        nearest_neighbor_idxs = nn(tree, transformed_positions[:, i] .+ nn_matrix)

        #Check that all neighbors are within tolerance of their expected positions
        #otherwise, reject this point
        valid_atoms = nearest_neighbor_idxs[2] .< tolerance
        if sum(valid_atoms) < strictness
            latt_param_matrix[:, i] = zeros(T, 2)
        else
            neighbor_positions = transformed_positions[:, nearest_neighbor_idxs[1]]
            diff_vectors = abs.(neighbor_positions .- transformed_positions[:, i])

            vector_1_positions = BitVector([1,1,1,0,0,1,1,1]) 
            vector_2_positions = BitVector([1,0,1,1,1,1,0,1]) 
            vec_1_latt_param = mean(diff_vectors[1, vector_1_positions .&& valid_atoms])
            vec_2_latt_param = mean(diff_vectors[2, vector_2_positions .&& valid_atoms])

            latt_param_matrix[:, i] = [vec_1_latt_param, vec_2_latt_param]
        end
    end
    #Transform back
    abs.([vec_1 vec_2] * latt_param_matrix)
end