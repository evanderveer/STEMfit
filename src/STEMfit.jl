module STEMfit

    import NearestNeighbors: 
                    NNTree, 
                    KDTree, 
                    knn,
                    nn
    import Images: 
                    Gray, 
                    load, 
                    save, 
                    Kernel, 
                    imfilter, 
                    label_components, 
                    component_centroids,
                    component_lengths,
                    mapwindow, 
                    N0f8, 
                    mosaicview
    import ImageBinarization:
                    Niblack,
                    binarize
    import Clustering: 
                    dbscan, 
                    ClusteringResult
    import LinearAlgebra: 
                    svd, 
                    Diagonal,
                    UpperTriangular,
                    cross,
                    dot, 
                    norm,
                    cholesky,
                    Hermitian
    import Combinatorics:
                    combinations,
                    sortperm
    import Statistics:
                    mean
    import Distances: 
                    Euclidean, 
                    PeriodicEuclidean, 
                    euclidean, 
                    pairwise
    import StaticArrays:
                    @MMatrix,
                    @MVector
    import DelimitedFiles:
                    writedlm
    import Plots:
                    scatter,
                    plot,
                    scatter!,
                    plot!,
                    Shape,
                    hline!,
                    vline!
                    
    include("findatoms.jl")
    include("finduc.jl")
    include("gaussianmodel.jl")
    include("latticemodel.jl")
    include("imagemodel.jl")
    include("background.jl")
    include("transformation.jl")
    include("latticeparameter.jl")
    include("gaussianfitting.jl")
    include("utils.jl")

end