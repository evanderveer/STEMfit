mutable struct ImageModel
    model_size::Tuple
    init_pos_list::AbstractMatrix{<:AbstractFloat}
    init_A::AbstractVector{<:AbstractFloat}
    init_a::AbstractVector{<:AbstractFloat}
    init_b::AbstractVector{<:AbstractFloat}
    init_c::AbstractVector{<:AbstractFloat}
    gaussian_list::Vector{Gaussian{<:AbstractFloat}}
    background::Matrix{<:AbstractFloat} 
    ImageModel(model_size, init_pos_list, init_A, init_a, init_b, init_c) = begin
        size(init_pos_list)[2] == 
        length(init_A) == 
        length(init_a) == 
        length(init_b) == 
        length(init_c) || throw(ArgumentError("arguments must have the same length"))
        gaussian_list = make_gaussians(init_pos_list, init_A, init_a, init_b, init_c)
        #Construct initially without any atoms and zero background
        new(
            model_size, 
            init_pos_list, 
            init_A, 
            init_a, 
            init_b, 
            init_c, 
            gaussian_list, 
            zeros(Float64, model_size)
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
    model.gaussian_list = make_gaussians(
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
    init_pos_list::AbstractMatrix{<:Real}, 
    init_A::AbstractVector{<:Real}, 
    init_a::AbstractVector{<:Real}, 
    init_b::AbstractVector{<:Real}, 
    init_c::AbstractVector{<:Real}
    ) 

    gaussian_list = Vector{Gaussian}([])
    for (pos, A, a, b, c) in zip(
        eachcol(init_pos_list),
        init_A,
        init_a,
        init_b,
        init_c
    )
        new_gaussian = Gaussian(Float64.(pos)..., Float64(A), Float64(a), Float64(b), Float64(c)) 
        push!(gaussian_list, new_gaussian)
    end
    gaussian_list
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
    intens += intensity(model.gaussian_list[idxs], x, y)
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
    tree = KDTree(model.init_pos_list)
    
    image = Matrix{Float32}(undef, length(yrange), length(xrange))

    fill_image!(image, model, xrange, yrange, tree, num_gaussians)
    return image
end

function produce_image(
    model::ImageModel,
    tree::NNTree;
    num_gaussians::Integer = 10
    )

    xrange = (1:model.model_size[2])
    yrange = (1:model.model_size[1])

    image = Matrix{Float32}(undef, length(yrange), length(xrange))

    fill_image!(image, model, xrange, yrange, tree, num_gaussians)
    return image
end

function produce_image(
    model::ImageModel;
    num_gaussians::Integer = 10
    )

    # If no NNTree is passed, construct one
    tree = KDTree(model.init_pos_list)

    xrange = (1:model.model_size[2])
    yrange = (1:model.model_size[1])

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
    Threads.@threads for (column, ypos) in collect(enumerate(yrange))
        Threads.@threads for (row, xpos) in collect(enumerate(xrange))
            image[column,row] = intensity(model, xpos, ypos, tree, num_gaussians)
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
    σ_X::AbstractVector{<:Real},
    σ_Y::AbstractVector{<:Real},
    θ::AbstractVector{<:Real}
) 
    length(σ_X) == length(σ_Y) == length(θ) || 
            throw(ArgumentError("arguments must have the same length"))


    a = Vector{Float64}(undef, length(θ))
    b = Vector{Float64}(undef, length(θ))
    c = Vector{Float64}(undef, length(θ))
    for (n,(i,j,k)) in enumerate(zip(σ_X,σ_Y,θ))
        (a[n], b[n], c[n]) = transform_gaussian_parameters(i,j,k)
    end
    (a, b, c)
end

function get_initial_gaussian_parameters(
    σ_X::AbstractVector{<:Real},
    σ_Y::AbstractVector{<:Real}
)
    get_initial_gaussian_parameters(σ_X, σ_Y, zeros(length(σ_X)))
end

function get_initial_gaussian_parameters(
    σ::AbstractVector{<:Real}
)
    get_initial_gaussian_parameters(σ, σ, zeros(length(σ)))
end