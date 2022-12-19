struct UnitCell
    volume::Real
    angle::Real
    vector_1::Vector{<:AbstractFloat}
    vector_2::Vector{<:AbstractFloat}
    UnitCell((volume, angle, vectors)) = new(volume, angle, Float32.(vectors[1]), Float32.(vectors[2]))
end

"""
    find_unit_cells(
                    centroids::Matrix{<:Real}[,
                    num_nn::Integer = 10,
                    cluster_radius::Real = 0.2,
                    min_cluster_size::Integer = 50,
                    uc_allowed_areas::UnitRange{<:Real} = 10:Inf,
                    uc_allowed_angles::UnitRange{<:Real} = 5:360,
                    min_neighbor_dist::Real = 5,
                    filter_tolerance::Real = 10
                    ])
                    -> Tuple{Matrix{Any}, Matrix{<:AbstractFloat}, KDTree}

Find unit cells from nearest neighbors. 

Calculates `num_nn` nearest neighbors for each centroid in `centroids`, then clusters these
using a DBSCAN clustering algorithm. The potential unit cells are filtered using the 
`uc_allowed_angles` and `uc_allowed_areas` parameters. 

# Arguments
- `centroids::Matrix{<:Real}`: Matrix of centroids to find a unit cell of
- `num_nn::Integer = 40`: Number of nearest neighbors to find for each centroid
- `cluster_radius::Real = 0.05`: *radius* parameter used by DBSCAN
- `min_cluster_size::Integer = 1000`: *min_cluster_size* parameter used by DBSCAN
- `uc_allowed_areas::Tuple{Real, Real} = (10, Inf)`: Range of allowed unit cell areas
- `uc_allowed_angles::Tuple{Real, Real} = (85, 95)`: Range of allowed unit cell uc_angles
- `min_neighbor_dist::Real = 15`: Minimum neighbor distance, removes spurious neighbors
- `filter_tolerance::Real = 10`: Minimum relative angle and area difference between unit cells
"""
function find_unit_cells(
    centroids::Matrix{<:Real};
    num_nn::Integer = 10,
    cluster_radius::Real = 0.2,
    min_cluster_size::Integer = 50,
    uc_allowed_areas::UnitRange{<:Real} = 10:1000000,
    uc_allowed_angles::UnitRange{<:Real} = 5:360,
    min_neighbor_dist::Real = 5,
    filter_tolerance::Real = 0.1
)

    #Find nearest neighbors for all atoms in centroids
    (neighbors_full, atom_tree) = find_neighbors(centroids, num_nn)


    #Find neighbors using DBSCAN
    neighbors = find_neighbor_clusters(
                                neighbors_full, 
                                radius=cluster_radius, 
                                min_cluster_size=min_cluster_size, 
                                min_neighbor_dist=min_neighbor_dist
                                )

    #Find potential unit cells from the nearest neighbor positions
    sorted_uc = unit_cells_from_nn(
                                    neighbors, 
                                    uc_allowed_angles=uc_allowed_angles, 
                                    uc_allowed_areas=uc_allowed_areas,
                                    tolerance=filter_tolerance
                                    )

    return (sorted_uc, neighbors, atom_tree)
end

"""
    find_neighbors(centroids::Matrix{<:AbstractFloat}[,num_neighbors::Integer])
        -> Tuple{Matrix{Float32}, KDTree}

Find relative positions of the num_neighbors nearest neighbors 
for each cluster in centroids.

The relative positions are rounded to the nearest integer.
"""
function find_neighbors(
    centroids::AbstractMatrix{T},
    num_neighbors::Integer = 40
) where {T<:AbstractFloat}

    atom_tree = KDTree(centroids)

    vectors = Matrix{T}(undef, 2, num_neighbors*size(centroids)[2])

    #Find num_neighbors nearest neighbors for each atom
    idxs::Vector{Vector{Int32}}, _ = knn(atom_tree, centroids, num_neighbors)

    for (i, (neighbors, atom)) in enumerate(zip(idxs, eachcol(centroids)))
        difference_vectors = Float32.(centroids[:, neighbors] .- atom)
        vectors[:, (i-1) * num_neighbors + 1:i * num_neighbors] = difference_vectors
    end

    return (vectors, atom_tree)
end

"""
    find_neighbor_clusters(neighbors::Matrix{T} where T<:AbstractFloat,
                           radius::Real, 
                           min_cluster_size::Integer[,
                           min_neighbor_dist::Real]
                           ) 
                           -> Matrix{<:AbstractFloat}

Finds clusters in a matrix of nearest neighbors using DBSCAN clustering.

`radius` and `min_cluster_size` are parameters used by DBSCAN, 
see the Clustering.jl documentation. Clusters closer than `min_neighbor_dist`
 are removed from the set of clusters.

"""
function find_neighbor_clusters(
    neighbors::AbstractMatrix{T};
    radius::Real,
    min_cluster_size::Integer,
    min_neighbor_dist::Real = 5
)   where {T<:AbstractFloat}

    #TODO: Find a more memory efficient clustering algorithm
    clusters = dbscan(neighbors, radius, min_cluster_size=min_cluster_size)

    #The cluster center is the average of all neighbors belonging to the cluster
    cluster_centers = Matrix{T}(undef, 2, 0)
    for cluster in clusters
        if length(cluster.core_indices) != 0
            #Cast to Float64 so the sum does not exceed Float32 max value
            mean_vector = mean(Float64.(neighbors[:, cluster.core_indices]), dims=2)
            cluster_centers = [cluster_centers T.(mean_vector)]
        end
    end

    #Filter out centers which are too close to each other
    pairwise_clusters = pairwise(Euclidean(), cluster_centers, dims=2)
    cleaned_indices = [!any(0.0 .< col .< min_neighbor_dist) 
                        for col in eachcol(UpperTriangular(pairwise_clusters))]
    cleaned_clusters = cluster_centers[:, cleaned_indices]
    
    return cleaned_clusters
end

"""
    unit_cells_from_nn(
                        neighbors::Matrix{<:AbstractFloat}, 
                        uc_allowed_areas::Tuple{<:Real, <:Real},
                        uc_allowed_angles::Tuple{<:Real, <:Real},
                        tolerance::Real = 0.05
                        )
                        -> Matrix{Any}

Find all possible unit cells from a matrix of nearest neighbor positions. 

Only return unit cells with a size in `uc_allowed_areas` and an angle
in `uc_allowed_angles`. Unit cells whose angle or area is within `tolerance`
of another unit cell are filtered out. Unit cells are returned sorted from 
smallest to largest.
"""
function unit_cells_from_nn(
    neighbors::AbstractMatrix{<:AbstractFloat};
    uc_allowed_areas::UnitRange{<:Real},
    uc_allowed_angles::UnitRange{<:Real},
    tolerance::Real = 0.05
)

    #Each combination of two basis vectors is a potential unit cell edge
    #use getindex.() to flatten the array one level
    unit_cell_edges = getindex.(diff.(collect(combinations(collect(eachcol(neighbors)), 2))), 1)

    #Each combination of two edges makes a unit cell
    unit_cells = collect(combinations(unit_cell_edges, 2))

    #Calculate unit cell volume, round to nearest integer
    uc_areas = round.(Int64, uc_area.(unit_cells))
    uc_angles = round.(Int64, uc_angle.(unit_cells))
    uc_matrix = [uc_areas uc_angles unit_cells]

    #Select only nonzero unit cells
    allowed_area = [i ∈ uc_allowed_areas for i in uc_areas]
    allowed_angle = [i ∈ uc_allowed_angles for i in uc_angles]

    #Only select unit cells which have an allowed angle and area
    allowed_ucs = uc_matrix[allowed_angle .&& allowed_area, :]
    
    #Sort by unit cell volume
    sorted_ucs = allowed_ucs[sortperm(allowed_ucs[:, 1]), :]

    ##TODO: Implement filtering
    filtered_ucs = filter_unit_cells(sorted_ucs, tolerance)
 
    #Return a list of UnitCell structs
    UnitCell.(eachrow(filtered_ucs))::Vector{UnitCell}
end

"""
    filter_unit_cells(
                        unit_cells::AbstractMatrix,
                        tolerance::Real
    )

Takes a sorted matrix of unit cells and filters out unit cells which 
have an angle or area is within `tolerance` of another unit cell in 
the list. `tolerance` is the relative difference between unit cells.  
"""
function filter_unit_cells(
    unit_cells::AbstractMatrix,
    tolerance::Real
)
    filtered_unit_cells = Matrix{Any}(undef, 0, 3)
    #Use permutedims() instead of ' so the third element does not get transposed
    filtered_unit_cells = [filtered_unit_cells; permutedims(unit_cells[1, :])]

    for unit_cell in eachrow(unit_cells)
        #If either the angle or the area are sufficiently different, consider this 
        #a new unit cell.
        if abs(unit_cell[1] - filtered_unit_cells[end,1])/unit_cell[1] > tolerance ||
            abs(unit_cell[2] - filtered_unit_cells[end,2])/unit_cell[2] > tolerance
            filtered_unit_cells = [filtered_unit_cells; permutedims(unit_cell)]
        end
    end
    filtered_unit_cells
end
