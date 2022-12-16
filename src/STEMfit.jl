module STEMfit

    import NearestNeighbors: 
                    NNTree, 
                    KDTree, 
                    knn
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
                    N0f8
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
                    norm
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

                    
    include("findatoms.jl")
    include("finduc.jl")
    include("gaussianmodel.jl")
    include("latticemodel.jl")
    include("imagemodel.jl")
    include("background.jl")
    include("utils.jl")

end