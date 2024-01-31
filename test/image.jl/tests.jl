@testset "load_image" begin
    for image in TEST_IMAGES
        (fn, ext) = splitext(image)
        
        if lowercase(ext) ∈ [".png", ".tif", ".tiff"]
            if fn[end-3:end] == "8bit"
                @test STEMfit.load_image(image) ≈ IMAGE_DATA_8BIT
            else
                @test STEMfit.load_image(image) ≈ IMAGE_DATA_16BIT
            end
        else
            @test_throws ErrorException STEMfit.load_image(image)
        end
    end
end

@testset "load_image downscale" begin
    @test size(STEMfit.load_image(TIF_IMAGE)) == (1258, 1258)
    @test size(STEMfit.load_image(TIF_IMAGE, downscale_factor = 1)) == (1258, 1258)
    @test size(STEMfit.load_image(TIF_IMAGE, downscale_factor = 2)) == (629, 629)
    @test size(STEMfit.load_image(TIF_IMAGE, downscale_factor = 4)) == (314, 314)
    @test size(STEMfit.load_image(TIF_IMAGE, downscale_factor = 8)) == (157, 157)
    @test size(STEMfit.load_image(TIF_IMAGE, downscale_factor = 629)) == (2, 2)
    @test size(STEMfit.load_image(TIF_IMAGE, downscale_factor = 1258)) == (1, 1)
    @test size(STEMfit.load_image(TIF_IMAGE, downscale_factor = 1259)) == (0, 0)


    @test STEMfit.load_image(TIF_IMAGE, downscale_factor = 1) ≈ IMAGE_DATA_16BIT
    @test STEMfit.load_image(TIF_IMAGE, downscale_factor = 2) ≈ Gray{Float64}.(load(joinpath(@__DIR__, "../data/downscaled", "image_d"*string(2)*".tif")))
    @test STEMfit.load_image(TIF_IMAGE, downscale_factor = 4) ≈ Gray{Float64}.(load(joinpath(@__DIR__, "../data/downscaled", "image_d"*string(4)*".tif")))
    @test STEMfit.load_image(TIF_IMAGE, downscale_factor = 8) ≈ Gray{Float64}.(load(joinpath(@__DIR__, "../data/downscaled", "image_d"*string(8)*".tif")))
    @test STEMfit.load_image(TIF_IMAGE, downscale_factor = 629) ≈ Gray{Float64}.(load(joinpath(@__DIR__, "../data/downscaled", "image_d"*string(629)*".tif")))
    @test STEMfit.load_image(TIF_IMAGE, downscale_factor = 1258) ≈ Gray{Float64}.(load(joinpath(@__DIR__, "../data/downscaled", "image_d"*string(1258)*".tif")))

    @test_throws ArgumentError STEMfit.load_image(TIF_IMAGE, downscale_factor = -1)
    @test_throws TypeError STEMfit.load_image(TIF_IMAGE, downscale_factor = 0.5)
    @test_throws TypeError STEMfit.load_image(TIF_IMAGE, downscale_factor = "1")
end

@testset "filter_image" begin
    @test STEMfit.filter_image(STEMfit.load_image(TIF_IMAGE), plot=false) ≈ FILTERED_IMAGE
end