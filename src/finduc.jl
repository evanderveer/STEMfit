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
                    num_nn::Integer,
                    cluster_radius::Real,
                    min_cluster_size::Integer,
                    uc_allowed_areas::Tuple{Real, Real},
                    uc_allowed_angles::Tuple{Real, Real},
                    min_neighbor_dist::Real 
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
"""
function find_unit_cells(
    centroids::Matrix{<:Real};
    num_nn::Integer = 40,
    cluster_radius::Real = 0.05,
    min_cluster_size::Integer = 1000,
    uc_allowed_areas::Tuple{Real, Real} = (10, Inf),
    uc_allowed_angles::Tuple{Real, Real} = (85, 95),
    min_neighbor_dist::Real = 15
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
                                    uc_allowed_areas=uc_allowed_areas
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
    centroids::Matrix{<:AbstractFloat},
    num_neighbors::Integer = 40
)

    atom_tree = KDTree(centroids)

    vectors = Matrix{Float32}(undef, 2, num_neighbors*size(centroids)[2])

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
    neighbors::Matrix{T};
    radius::Real,
    min_cluster_size::Integer,
    min_neighbor_dist::Real = 15.0
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
                        uc_allowed_angles::Tuple{<:Real, <:Real}
                        )
                        -> Matrix{Any}

Find all possible unit cells from a matrix of nearest neighbor positions. 

Only return unit cells with a size in `uc_allowed_areas` and an angle
in `uc_allowed_angles`. Unit cells are returned sorted from smallest to largest.
"""
function unit_cells_from_nn(
    neighbors::Matrix{<:AbstractFloat};
    uc_allowed_areas::Tuple{<:Real, <:Real},
    uc_allowed_angles::Tuple{<:Real, <:Real}
)

    #Each combination of two basis vectors is a potential unit cell
    unit_cells = collect(combinations(collect(eachcol(neighbors)), 2))

    #Calculate unit cell volume, round to nearest integer
    uc_areas = round.(Int32, uc_area.(unit_cells))
    uc_angles = round.(Int32, uc_angle.(unit_cells))
    uc_matrix = [uc_areas uc_angles unit_cells]

    #Select only nonzero unit cells
    allowed_area = uc_allowed_areas[1] .< uc_areas .< uc_allowed_areas[2]
    allowed_angle = uc_allowed_angles[1] .< uc_angles .< uc_allowed_angles[2]

    #Only select unit cells which have an allowed angle and area
    allowed_ucs = uc_matrix[allowed_angle .+ allowed_area .== 2, :]
    
    #Sort by unit cell volume
    sorted_ucs = allowed_ucs[sortperm(allowed_ucs[:, 1]), :]

    #Return a list of UnitCell structs
    UnitCell.(eachrow(sorted_ucs))
end



