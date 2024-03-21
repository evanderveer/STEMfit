# The contents of this file were adapted from the Clustering.jl package
# https://github.com/JuliaStats/Clustering.jl
#
# DBSCAN Clustering
#
#   References:
#
#       Martin Ester, Hans-peter Kriegel, Jörg S, and Xiaowei Xu
#       A density-based algorithm for discovering clusters
#       in large spatial databases with noise. 1996.
#

"""
    DbscanCluster

    DBSCAN cluster returned by [`dbscan`](@ref) function (point coordinates-based
    implementation)

    # Fields
    * `size::Integer`: number of points in a cluster (core + boundary)
    * `core_indices::Vector{<:Integer}`: indices of points in the cluster *core*
    * `boundary_indices::Vector{<:Integer}`: indices of points on the cluster *boundary*
"""
struct DbscanCluster
    size::Integer                      # number of points in cluster
    core_indices::Vector{<:Integer}      # core points indices
    boundary_indices::Vector{<:Integer}  # boundary points indices
end

## main algorithm

"""
    dbscan(points::AbstractMatrix, radius::Real,
           [leafsize], [min_neighbors], [min_cluster_size]) -> Vector{DbscanCluster}

    Cluster `points` using the DBSCAN (density-based spatial clustering of
    applications with noise) algorithm.

    # Arguments
    - `points`: the ``d×n`` matrix of points. `points[:, j]` is a
    ``d``-dimensional coordinates of ``j``-th point
    - `radius::Real`: query radius

    Optional keyword arguments to control the algorithm:
    - `leafsize::Integer` (defaults to 20): the number of points binned in each
    leaf node in the `KDTree`
    - `min_neighbors::Integer` (defaults to 1): the minimum number of a *core* point
    neighbors
    - `min_cluster_size::Integer` (defaults to 1): the minimum number of points in
    a valid cluster

    # Example
    ``` julia
    points = randn(3, 10000)
    # DBSCAN clustering, clusters with less than 20 points will be discar ded:
    clusters = dbscan(points, 0.05, min_neighbors = 3, min_cluster_size = 20)
    ```
"""
function dbscan(
                points::AbstractMatrix, 
                radius::Real; 
                leafsize::Integer = 20, 
                min_neighbors::Integer = 1, 
                min_cluster_size::Integer = 1,
                kwargs ...
            )
    kdtree = KDTree(points; leafsize=leafsize)
    dim, num_points = size(points)
    num_points <= dim && throw(ArgumentError("points has $dim rows and $num_points columns. Must be a D x N matric with D < N"))
    0 <= radius || throw(ArgumentError("radius $radius must be ≥ 0"))
    1 <= min_neighbors || throw(ArgumentError("min_neighbors $min_neighbors must be ≥ 1"))
    1 <= min_cluster_size || throw(ArgumentError("min_cluster_size $min_cluster_size must be ≥ 1"))

    clusters = Vector{DbscanCluster}()
    visited = falses(num_points)
    cluster_selection = falses(num_points)
    core_selection = falses(num_points)
    to_explore = Vector{Int64}()
    adj_list = Vector{Int64}()
    return _dbscan(
                    kdtree, 
                    num_points, 
                    points, 
                    radius, 
                    clusters,
                    visited, 
                    cluster_selection, 
                    core_selection, 
                    to_explore, 
                    adj_list; 
                    min_neighbors,
                    min_cluster_size,
                    kwargs ...
                    )
end


# An implementation of DBSCAN algorithm that keeps track of both the core and boundary points
function _dbscan(
                 kdtree::KDTree,
                 num_points::Integer,
                 points::AbstractMatrix,
                 radius::Real,
                 clusters::Vector{DbscanCluster},
                 visited::BitVector,
                 cluster_selection::BitVector,
                 core_selection::BitVector,
                 to_explore::Vector{<:Integer},
                 adj_list::Vector{<:Integer};
                 min_neighbors::Integer = 1, 
                 min_cluster_size::Integer = 1
                )  
    if "--cluster" in ARGS || "-c" in ARGS; cluster = true; else; cluster = false; end
    for i = 1:num_points
        #Manually GC so it works on a cluster
        if i % 5000 == 0 && cluster; GC.gc(); end
        visited[i] && continue
        push!(to_explore, i) # start a new cluster
        fill!(core_selection, false)
        fill!(cluster_selection, false)

        while !isempty(to_explore)
            current_index = popfirst!(to_explore)
            visited[current_index] && continue
            visited[current_index] = true
            append!(adj_list, inrange(kdtree, points[:, current_index], radius))
            cluster_selection[adj_list] .= true
            
            # if a point doesn't have enough neighbors it is not a 'core' point and its neighbors are not added to the to_explore list
            if (length(adj_list) - 1) < min_neighbors
                empty!(adj_list)
                continue # query returns the query point as well as the neighbors
            end
            core_selection[current_index] = true
            update_exploration_list!(adj_list, to_explore, visited)
        end
        cluster_size = sum(cluster_selection)
        min_cluster_size <= cluster_size && accept_cluster!(clusters, core_selection, cluster_selection, cluster_size)
    end
    return clusters
end

"""
    update_exploration_list!(adj_list, exploration_list, visited) -> adj_list

    Update the queue for expanding the cluster.

    # Arguments
    - `adj_list::Vector{<:Integer}`: indices of the neighboring points to move to queue
    - `exploration_list::Vector{<:Integer}`: the indices that will be explored in the future
    - `visited::BitVector`: a flag indicating whether a point has been explored already
"""
function update_exploration_list!(adj_list::Vector{T}, exploration_list::Vector{T},
                                  visited::BitVector) where T <: Integer
    for j in adj_list
        visited[j] && continue
        push!(exploration_list, j)
    end
    empty!(adj_list)
end

"""
    accept_cluster!(clusters, core_selection, cluster_selection) -> clusters

    Accept cluster and update the clusters list.

    # Arguments
    - `clusters::Vector{DbscanCluster}`: a list of the accepted clusters
    - `core_selection::Vector{Bool}`: selection of the core points of the cluster
    - `cluster_selection::Vector{Bool}`: selection of all the cluster points
"""
function accept_cluster!(clusters::Vector{DbscanCluster}, core_selection::BitVector,
                         cluster_selection::BitVector, cluster_size::Integer)
    core_idx = findall(core_selection) # index list of the core members
    boundary_selection = cluster_selection .& (~).(core_selection) #TODO change to .~ core_selection
                                                                            # when dropping 0.5
    boundary_idx = findall(boundary_selection) # index list of the boundary members
    push!(clusters, DbscanCluster(cluster_size, core_idx, boundary_idx))
end