module STEMfit

    import NearestNeighbors: 
                    NNTree, 
                    KDTree, 
                    knn,
                    nn,
                    inrange
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
    import Plots:
                    scatter,
                    plot,
                    scatter!,
                    plot!,
                    Shape,
                    hline!,
                    vline!
                    
    include("findatoms.jl")
    include("dbscan.jl")
    include("finduc.jl")
    include("gaussianmodel.jl")
    include("latticemodel.jl")
    include("imagemodel.jl")
    include("background.jl")
    include("transformation.jl")
    include("latticeparameter.jl")
    include("utils.jl")

end