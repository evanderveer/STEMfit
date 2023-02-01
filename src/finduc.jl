struct UnitCell
    volume::Real
    angle::Real
    vector_1::Vector{<:AbstractFloat}
    vector_2::Vector{<:AbstractFloat}
    UnitCell((volume, angle, vectors)) = new(volume, angle, vectors[1], vectors[2])
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
    num_nn::Integer = 20,
    cluster_radius::Union{<:Real, Symbol} = :auto,
    min_cluster_size::Union{<:Integer, Symbol} = :auto,
    uc_allowed_areas::UnitRange{<:Real} = 10:1000000,
    uc_allowed_angles::UnitRange{<:Real} = 5:360,
    min_neighbor_dist::Union{<:Real, Symbol} = :auto,
    filter_tolerance::Real = 0.1
)

    #Find nearest neighbors for all atoms in centroids
    (neighbors_full, atom_tree, distance_measure) = find_neighbors(centroids, num_nn)

    if cluster_radius == :auto
        cluster_radius = distance_measure/100
    end
    if min_neighbor_dist == :auto
        min_neighbor_dist = distance_measure/2.5
    end
    if min_cluster_size == :auto
        min_cluster_size = round(Int64,size(centroids)[2]/70)
    end

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
    idxs::Vector{Vector{Int32}}, distances = knn(atom_tree, centroids, num_neighbors)

    for (i, (neighbors, atom)) in enumerate(zip(idxs, eachcol(centroids)))
        difference_vectors = Float32.(centroids[:, neighbors] .- atom)
        vectors[:, (i-1) * num_neighbors + 1:i * num_neighbors] = difference_vectors
    end

    distance_measure = mean(vcat(distances...))/num_neighbors^0.5
    return (vectors, atom_tree, distance_measure)
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

    clusters = dbscan(neighbors, radius, min_cluster_size=min_cluster_size)

    #The cluster center is the average of all neighbors belonging to the cluster
    cluster_centers = Matrix{T}(undef, 2, 0)
    for cluster in clusters
        if length(cluster.core_indices) != 0
            mean_vector = mean((neighbors[:, cluster.core_indices]), dims=2)
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
Returns a matrix with columns of unit cell area, angle and basis vectors.
Basis vectors are ordered such that the most in-plane vector always comes first.
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
    uc_squareness = round.(Int64, STEMfit.uc_squareness.(unit_cells))
    
    uc_matrix = [uc_areas uc_angles uc_squareness unit_cells]

    #Select only nonzero unit cells
    allowed_area = [i ∈ uc_allowed_areas for i in uc_areas]
    allowed_angle = [i ∈ uc_allowed_angles for i in uc_angles]

    #Only select unit cells which have an allowed angle and area
    allowed_ucs = uc_matrix[allowed_angle .&& allowed_area, :]
    
    #Sort by unit cell volume
    sorted_ucs = allowed_ucs[sortperm(allowed_ucs[:, 1]), :]

    #Filter out identical unit cells
    filtered_ucs = filter_unit_cells(sorted_ucs, tolerance)
 
    #Optionally flip the basis vectors, so the most in-plane Vector
    #always comes first
    for row in eachrow(filtered_ucs)
        basis_vector_angles = uc_angle.(row[3])
        distances_to_ip = dist_to_ip.(basis_vector_angles)
        if distances_to_ip[1] > distances_to_ip[2]
            row[3][1], row[3][2] = row[3][2], row[3][1]
        end
    end

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
    filtered_unit_cells = Matrix{Any}(undef, 0, 4)
    #Use permutedims() instead of ' so the third element does not get transposed
    filtered_unit_cells = [filtered_unit_cells; permutedims(unit_cells[1, :])]

    for unit_cell in eachrow(unit_cells)
        #If this unit cell is not yet in the list, add it
        if all([is_different_unit_cell(previous_unit_cell, unit_cell, tolerance) 
                    for previous_unit_cell in eachrow(filtered_unit_cells)])
            filtered_unit_cells = [filtered_unit_cells; permutedims(unit_cell)]
        end
    end
    filtered_unit_cells[:, [1,2,4]]
end

"""
    dist_to_ip(
                angle::Real
    )

Calculates the angular distance of `angle` to the in-plane direction   
"""
dist_to_ip(angle::Real) = minimum([abs(angle-270), abs(angle-90)])

function uc_squareness(basis_vectors::AbstractVector{<:AbstractVector{<:Real}})
    (side_1, side_2) = norm.(basis_vectors)
    if side_1 >= side_2
        return side_1/side_2
    else
        return side_2/side_1
    end
end 

function is_different_unit_cell(
    uc_1,
    uc_2,
    tolerance
)
    any([abs.(uc_1[i] .- uc_2[i])./uc_2[i] .> tolerance for i in 1:3])
end