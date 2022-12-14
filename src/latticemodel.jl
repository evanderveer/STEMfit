mutable struct LatticeModel{T}
    init_pos_list::Matrix{T}
    init_A::Vector{T}
    init_a::Vector{T}
    init_b::Vector{T}
    init_c::Vector{T}
    gaussian_list::Vector{Gaussian{T}}
end

"""
    initialize!(
        model::LatticeModel
    )

Creates a list of Gaussian structs from the initial parameters stored in *model*.
The Gaussians are stored in *model*.gaussian_list.
"""
function initialize!(
    model::LatticeModel
    )
    model.gaussian_list = Vector{Gaussian}([])
    for (pos, A, a, b, c) in zip(
        eachrow(model.init_pos_list),
        model.init_A,
        model.init_a,
        model.init_b,
        model.init_c
        )
        
        new_gaussian = Gaussian(pos..., A, a, b, c) 
        
        push!(model.gaussian_list, new_gaussian)
    end
end

"""
    intensity(
        model::LatticeModel,
        x:Real,
        y::Real,
        mask_dist::Real
    )
    -> Real

Returns the summed intensity of the LatticeModel *model* at the point (*x*, *y*). 
Only considers gaussians within *mask_dist* of (x, y).
"""
function intensity(
    model::LatticeModel,
    x::Real,
    y::Real, 
    mask_dist::Real
    )
    total_intens = zero(Float32)
    for gaussian in model.gaussian_list
        if abs(x-gaussian.x0) < mask_dist && abs(y-gaussian.y0) < mask_dist
            total_intens += intensity(gaussian, x, y)
        end
    end
    return total_intens
end
