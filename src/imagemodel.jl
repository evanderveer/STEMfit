mutable struct ImageModel{T<:Real}
    model_size::Tuple
    init_pos_list::AbstractMatrix{T}
    init_A::AbstractVector{T}
    init_a::AbstractVector{T}
    init_b::AbstractVector{T}
    init_c::AbstractVector{T}
    gaussians::Vector{Gaussian}
    unit_cell::UnitCell
    atom_tree::NNTree
    background::Matrix{T} 
    ImageModel(model_size, 
               unit_cell, 
               atom_tree, 
               init_pos_list, 
               init_A,
               init_a, 
               init_b, 
               init_c
               ) = begin
        size(init_pos_list)[2] == 
        length(init_A) == 
        length(init_a) == 
        length(init_b) == 
        length(init_c) || throw(ArgumentError("arguments must have the same length"))
        gaussians = make_gaussians(init_pos_list, init_A, init_a, init_b, init_c)
        #Construct initially without any atoms and zero background
        new{typeof(init_A[1])}(
            model_size, 
            init_pos_list, 
            init_A, 
            init_a, 
            init_b, 
            init_c, 
            gaussians, 
            unit_cell,
            atom_tree,
            zeros(typeof(init_A[1]), model_size)
            ) 
    end
end

"""
    reset_model!(
        model::ImageModel
    )

Reset all gaussians in the model to their initial parameters.
"""
function reset_model!(
    model::ImageModel
    )
    model.gaussians = make_gaussians(
                            model.init_pos_list,
                            model.init_A,
                            model.init_a,
                            model.init_b,
                            model.init_c
                            )
end

"""
    make_gaussians(
        init_pos_list::AbstractMatrix{T}, 
        init_A::AbstractVector{T}, 
        init_a::AbstractVector{T}, 
        init_b::AbstractVector{T}, 
        init_c::AbstractVector{T}
    ) where {T<:Real}
    -> Vector{Gaussian{T}}

Makes a vector of Gaussian structs from lists of initial parameters.
"""
function make_gaussians(
    init_pos_list::AbstractMatrix{T}, 
    init_A::AbstractVector{T}, 
    init_a::AbstractVector{T}, 
    init_b::AbstractVector{T}, 
    init_c::AbstractVector{T}
    ) where {T<:Real}

    V = Float64#Dual{T,T,6} 
    gaussians = Vector{Gaussian{V}}([])
    for (pos, A, a, b, c) in zip(
        eachcol(init_pos_list),
        init_A,
        init_a,
        init_b,
        init_c
    )
        new_gaussian = Gaussian{V}(pos..., A, a, b, c) 
        push!(gaussians, new_gaussian)
    end
    gaussians
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

    intens = model.background[round.(Int64,(y,x))...]
    idxs = knn(tree, [y,x], num_gaussians)[1]
    intens += intensity(model.gaussians[idxs], x, y)
    return intens
end

function intensity(
    model::ImageModel,
    x::Real,
    y::Real,
    gaussian_indices::AbstractVector{<:Integer}
    ) 

    """intens = model.background[round.(Int64,(y,x))...]
    for i in gaussian_indices
        intens += intensity(model.gaussians[i], x, y)
    end
    intens"""
    
    model.background[round.(Int64,(y,x))...] + 
        intensity(model.gaussians[gaussian_indices], x, y)
end

"""
    produce_image(
        model::ImageModel,
        yrange::UnitRange{<:Integer},
        xrange::UnitRange{<:Integer},
        tree::NNTree,
        num_gaussians::Integer = 10
    )
    -> Matrix{Float64}

Returns an image of the *model* on the ranges given by *yrange* and *xrange*. 
If no ranges are given, the full size of *model* is used. *tree* is an NNTree
of Gaussian positions. A new *tree* is computed is none is passed.
"""
function produce_image(
    model::ImageModel{T},
    yrange::UnitRange{<:Integer},
    xrange::UnitRange{<:Integer},
    tree::NNTree;
    num_gaussians::Integer = 10
    ) where {T<:Real}
    
    V = Real
    image = Matrix{V}(undef, length(yrange), length(xrange))

    fill_image!(image, model, xrange, yrange, tree, num_gaussians)
    image
end

function produce_image(
    model::ImageModel{T},
    yrange::UnitRange{<:Integer},
    xrange::UnitRange{<:Integer};
    num_gaussians::Integer = 10
    ) where {T<:Real}

    V = Real
    # If no NNTree is passed, construct one
    tree = model.atom_tree
    
    image = Matrix{V}(undef, length(yrange), length(xrange))

    fill_image!(image, model, xrange, yrange, tree, num_gaussians)
    image
end

function produce_image(
    model::ImageModel{T},
    tree::NNTree;
    num_gaussians::Integer = 10
    ) where {T<:Real}

    V = Real
    xrange = (1:model.model_size[2])
    yrange = (1:model.model_size[1])

    image = Matrix{V}(undef, length(yrange), length(xrange))

    fill_image!(image, model, xrange, yrange, tree, num_gaussians)
    image
end

function produce_image(
    model::ImageModel{T};
    num_gaussians::Integer = 10
    ) where {T<:Real}

    V = Real
    # If no NNTree is passed, construct one
    tree = model.atom_tree

    xrange = (1:model.model_size[2])
    yrange = (1:model.model_size[1])

    image = Matrix{V}(undef, length(yrange), length(xrange))

    fill_image!(image, model, xrange, yrange, tree, num_gaussians)
    image
end

function produce_image(
    model::ImageModel{T},
    gaussian_indices::AbstractVector{<:Integer}
    ) where {T<:Real}

    V = Real
    xrange = (1:model.model_size[2])
    yrange = (1:model.model_size[1])

    image = Matrix{V}(undef, length(yrange), length(xrange))

    fill_image!(image, model, xrange, yrange, gaussian_indices)
    image
end

function produce_image(
    model::ImageModel{T},
    yrange::UnitRange{<:Integer},
    xrange::UnitRange{<:Integer},
    gaussian_indices::AbstractVector{<:Integer}
    ) where {T<:Real}

    V = Real
    image = Matrix{V}(undef, length(yrange), length(xrange))

    fill_image!(image, model, xrange, yrange, gaussian_indices)
    image
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
    image::Matrix{<:Real},
    model::ImageModel,    
    xrange::UnitRange{Int64},
    yrange::UnitRange{Int64},
    tree::NNTree,
    num_gaussians::Int64
)
    Threads.@threads for (column, ypos) in collect(enumerate(yrange))
         for (row, xpos) in collect(enumerate(xrange))
            image[column, row] = intensity(model, xpos, ypos, tree, num_gaussians)
        end
    end
end

function fill_image!(
    image::Matrix{<:Real},
    model::ImageModel,    
    xrange::UnitRange{Int64},
    yrange::UnitRange{Int64},
    gaussian_indices::AbstractVector{<:Integer}
)
    Threads.@threads for (column, ypos) in collect(enumerate(yrange))
         for (row, xpos) in collect(enumerate(xrange))
            image[column, row] = intensity(model, xpos, ypos, gaussian_indices)
        end
    end
end

function transform_gaussian_parameters(
    σ_X::Real,
    σ_Y::Real,
    θ::Real
)
    a = cos(θ)^2/(2*σ_X^2) + sin(θ)^2/(2*σ_Y^2)
    c = sin(θ)^2/(2*σ_X^2) + cos(θ)^2/(2*σ_Y^2)
    b = -sin(2*θ)/(4*σ_X^2) + sin(2*θ)/(4*σ_Y^2)
    (a, b, c)
end

function get_initial_gaussian_parameters(
    σ_X::AbstractVector{T},
    σ_Y::AbstractVector{T},
    θ::AbstractVector{T}
) where {T<:Real}
    length(σ_X) == length(σ_Y) == length(θ) || 
            throw(ArgumentError("arguments must have the same length"))


    a = Vector{T}(undef, length(θ))
    b = Vector{T}(undef, length(θ))
    c = Vector{T}(undef, length(θ))
    for (n,(i,j,k)) in enumerate(zip(σ_X,σ_Y,θ))
        (a[n], b[n], c[n]) = transform_gaussian_parameters(i,j,k)
    end
    (a, b, c)
end

function get_initial_gaussian_parameters(
    σ_X::AbstractVector{T},
    σ_Y::AbstractVector{T}
) where {T<:Real}
    get_initial_gaussian_parameters(σ_X, σ_Y, zeros(T, length(σ_X)))
end

function get_initial_gaussian_parameters(
    σ::AbstractVector{T}
) where {T<:Real}
    get_initial_gaussian_parameters(σ, σ, zeros(T, length(σ)))
end

function check_model(
    image_model;
    fix=true
)
    check_function(g) = check_gaussian(g, 
                        1:image_model.model_size[2], 
                        1:image_model.model_size[1]
                        )
    faulty_gaussians = sum(check_function.(image_model.gaussians))
    println(string(faulty_gaussians) * " Gaussian functions were found to have problems.")
    if fix
        filter!(g -> !check_function(g), image_model.gaussians)
        new_positions = hcat([Float64.([g.y0, g.x0]) for g in image_model.gaussians]...)

        image_model.atom_tree = KDTree(new_positions)
    end
end

function check_gaussian(
    gaussian::Gaussian,
    valid_range_x = -Inf:Inf,
    valid_range_y = -Inf:Inf
)
    0 < gaussian.A < 1 &&
    isposdef([gaussian.a gaussian.b;gaussian.b gaussian.c]) &&
    gaussian.x0 in valid_range_x &&
    gaussian.y0 in valid_range_y
end