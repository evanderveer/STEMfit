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
                    N0f8, 
                    mosaicview
    import ImageBinarization:
                    Niblack,
                    Sauvola,
                    binarize
    import ImageBinarization.BinarizationAPI:
                    AbstractImageBinarizationAlgorithm
    import LinearAlgebra: 
                    svd, 
                    Diagonal,
                    UpperTriangular,
                    cross,
                    dot, 
                    norm,
                    cholesky,
                    Hermitian,
                    isposdef
    import Combinatorics:
                    combinations,
                    sortperm
    import Statistics:
                    mean
    import StatsBase:
                    sample
    import Distances: 
                    Euclidean, 
                    PeriodicEuclidean, 
                    euclidean, 
                    pairwise
    import StaticArrays:
                    MVector,
                    SVector
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
    using Optimization
    using ForwardDiff
    import ForwardDiff:
                    Dual #Import explicitly so it can be extended
    import OptimizationOptimJL:
                    Newton,
                    BFGS,
                    LBFGS
    import Base:
                    Float64,
                    Float16

    include("findatoms.jl")
    include("dbscan.jl")
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