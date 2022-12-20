function calculate_lattice_parameters(
    positions::AbstractMatrix{T},
    unit_cell::UnitCell;
    #require_all_neighbors::Bool = true,
    tolerance::Real = 0.05
) where {T<:Real}
    vec_1 = unit_cell.vector_1
    vec_2 = unit_cell.vector_2
    inv_matrix = inv([vec_1 vec_2])
    transformed_positions = inv_matrix * positions
    tree = KDTree(transformed_positions)

    #Matrix is nearest neighbor positions in transformed space
    nn_matrix = [-1 -1 -1 0 0 1 1 1; -1 0 1 -1 1 -1 0 1]
    
    latt_param_matrix = similar(positions)
    
    for (pos, latt_param) in zip(eachcol(transformed_positions), eachcol(latt_param_matrix))

        #Get nearest neighbor indices and distances
        nearest_neighbor_idxs = nn(tree, pos .+ nn_matrix)

        #Check that all neighbors are within tolerance of their expected positions
        #otherwise, reject this point
        if !all(nearest_neighbor_idxs[2] .< tolerance)
            latt_param = zeros(T, 2)
        else
            neighbor_positions = transformed_positions[:, nearest_neighbor_idxs[1]]
            diff_vectors = abs.(neighbor_positions .- pos)
            vec_1_latt_param = mean(diff_vectors[1, [1,2,3,6,7,8]])
            vec_2_latt_param = mean(diff_vectors[2, [1,3,4,5,6,8]])
            latt_param = [vec_1_latt_param, vec_2_latt_param]
        end
    end
    #Transform back
    [vec_1 vec_2] * latt_param_matrix
end