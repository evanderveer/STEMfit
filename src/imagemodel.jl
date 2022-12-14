mutable struct ImageModel
    size::Tuple
    lattices::Vector{LatticeModel}
    background::Matrix{Float32} 
    ImageModel(size) = new(size, [], make_background(size, 0)) #Construct initially without any lattices and zero background
end

"""
    add_lattice!(
        model::ImageModel,
        init_pos_list,
        init_A,
        init_a,
        init_b,
        init_c
    )

Adds a new LatticeModel to *model*.lattices.
"""
function add_lattice!( #Probably should get rid of lattices alltogether
    model::ImageModel,
    init_pos_list,
    init_A,
    init_a,
    init_b,
    init_c
    )

    new_lattice = LatticeModel(#FIX
        init_pos_list,
        init_A,
        init_a,
        init_b,
        init_c,
        []
        )
    push!(model.lattices, new_lattice);
end

"""
    reset_lattices!(
        model::ImageModel
    )

Remove all lattices from *model*.lattices.
"""
function reset_lattices!(
    model::ImageModel
    )
    model.lattices = Vector{LatticeModel}[]
end

"""
    initialize!(
        model::ImageModel
    )

Initialize all lattices in *model*.lattices.
"""
function initialize!(
    model::ImageModel
    )
    for lattice in model.lattices
        initialize!(lattice)
    end
end

"""
    intensity(
        model::ImageModel,
        x:Real,
        y::Real,
        mask_dist::Real
    )
    -> Real

Returns the summed intensity of the ImageModel *model* at the point (*x*, *y*). 
Only considers gaussians within *mask_dist* of (x, y).
"""
function intensity(
    model::ImageModel,
    x::Real,
    y::Real,
    mask_size::Real
    )

    intens = model.background[round.(Int64,(y,x))...] 
    for lattice in model.lattices
        intens += intensity(lattice, x, y, mask_size)
    end
    return intens
end

"""
    intensity(
        model::ImageModel,
        x::Real,
        y::Real,
        tree::NNTree,
        num_gaussians::Integer = 10
    )
    -> Real

Returns the summed intensity of the ImageModel *model* at the point (*x*, *y*). 
An NNTree of the Gaussian positions must be passed to find the *num_gaussians* nearest 
neighbors used for the calculation.
"""
function intensity(
    model::ImageModel,
    x::Real,
    y::Real,
    tree::NNTree,
    num_gaussians::Integer = 10
    )

    intens = model.background[round.(Int64,(y,x))...]::Float32
    idxs = knn(tree, [y,x], num_gaussians)[1]
    intens += intensity(model.lattices[1].gaussian_list[idxs], x, y)
    return intens
end

"""
    produce_image(
        model::ImageModel,
        yrange::UnitRange{<:Integer},
        xrange::UnitRange{<:Integer},
        tree::NNTree,
        num_gaussians::Integer = 10
    )
    -> Matrix{Float32}

Returns an image of the *model* on the ranges given by *yrange* and *xrange*. 
If no ranges are given, the full size of *model* is used. *tree* is an NNTree
of Gaussian positions. A new *tree* is computed is none is passed.
"""
function produce_image(
    model::ImageModel,
    yrange::UnitRange{<:Integer},
    xrange::UnitRange{<:Integer},
    tree::NNTree;
    num_gaussians::Integer = 10
    )
    
    image = Matrix{Float32}(undef, length(yrange), length(xrange))

    fill_image!(image, model, xrange, yrange, tree, num_gaussians)
    return image
end

function produce_image(
    model::ImageModel,
    yrange::UnitRange{<:Integer},
    xrange::UnitRange{<:Integer};
    num_gaussians::Integer = 10
    )

    # If no NNTree is passed, construct one
    tree = KDTree(hcat([lat.init_pos_list' for lat in model.lattices]...))
    
    image = Matrix{Float32}(undef, length(yrange), length(xrange))

    fill_image!(image, model, xrange, yrange, tree, num_gaussians)
    return image
end

function produce_image(
    model::ImageModel,
    tree::NNTree;
    num_gaussians::Integer = 10
    )

    xrange = (1:model.size[2])
    yrange = (1:model.size[1])

    image = Matrix{Float32}(undef, length(yrange), length(xrange))

    fill_image!(image, model, xrange, yrange, tree, num_gaussians)
    return image
end

function produce_image(
    model::ImageModel;
    num_gaussians::Integer = 10
    )

    # If no NNTree is passed, construct one
    tree = KDTree(hcat([lat.init_pos_list' for lat in model.lattices]...))

    xrange = (1:model.size[2])
    yrange = (1:model.size[1])

    image = Matrix{Float32}(undef, length(yrange), length(xrange))

    fill_image!(image, model, xrange, yrange, tree, num_gaussians)
    return image
end

"""
    function fill_image!(
        image::Matrix{<:AbstractFloat},
        model::ImageModel,    
        xrange::UnitRange{Int64},
        yrange::UnitRange{Int64},
        tree::NNTree,
        num_gaussians::Int64
    )

Fills a matrix *image* with intensity values of *model*.
"""
function fill_image!(
    image::Matrix{<:AbstractFloat},
    model::ImageModel,    
    xrange::UnitRange{Int64},
    yrange::UnitRange{Int64},
    tree::NNTree,
    num_gaussians::Int64
)
    for (column, ypos) in enumerate(yrange)
        for (row, xpos) in enumerate(xrange)
            image[column,row] = intensity(model, xpos, ypos, tree, num_gaussians)
        end
    end
end




