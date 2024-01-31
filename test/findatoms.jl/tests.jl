@testset "find_optimum_threshold" begin
    @test STEMfit.find_optimum_threshold(zeros(2,2)) == 0.05
    @test STEMfit.find_optimum_threshold(ones(2,2)) == 0.05

    @test STEMfit.find_optimum_threshold(zeros(1000,1000)) == 0.05
    @test STEMfit.find_optimum_threshold(ones(1000,1000)) == 0.05
    
    Random.seed!(1)
    @test STEMfit.find_optimum_threshold(rand(1000,1000)) == 0.73
    Random.seed!(1)
    @test STEMfit.find_optimum_threshold(rand(1000,1000) .* 0.5) == 0.36

    @test STEMfit.find_optimum_threshold(FILTERED_IMAGE) ≈ 0.29
    @test STEMfit.find_optimum_threshold(IMAGE_DATA_16BIT) ≈ 0.29
end

@testset "find_atoms" begin
    @test_throws DomainError tp=STEMfit.ThresholdingParameters(threshold=2)
    @test_throws DomainError tp=STEMfit.ThresholdingParameters(threshold=-1)

    tp = STEMfit.ThresholdingParameters(bias=0.5, window_size=3, minimum_atom_size=0, plot=false)
    (atom_parameters, thresholded_image) = STEMfit.find_atoms(FILTERED_IMAGE, tp)
    atom_parameters_correct = STEMfit.load_atomic_parameters(joinpath(DATA_PATH, "positions_com", "atomic parameters.csv"))
    @test maximum(atom_parameters.centroids .- atom_parameters_correct.centroids) ≈ 0 atol=0.1
    @test maximum(atom_parameters.sizes .- atom_parameters_correct.sizes) ≈ 0 atol=0.1
    @test maximum(atom_parameters.intensities .- atom_parameters_correct.intensities) ≈ 0 atol=0.1

    tp = STEMfit.ThresholdingParameters(bias=0.5, window_size=25, minimum_atom_size=0, plot=false)
    ellipticity_image = STEMfit.load_image(joinpath(DATA_PATH, "binarized", "ellipticity test 25 full.png"))
    (atom_parameters, thresholded_image) = STEMfit.find_atoms(ellipticity_image, tp)
    atom_parameters_correct = STEMfit.load_atomic_parameters(joinpath(DATA_PATH, "positions_com", "ellipticity test full.csv"))
    @test maximum(atom_parameters.centroids .- atom_parameters_correct.centroids) ≈ 0 atol=0.1
    @test maximum(atom_parameters.sizes .- atom_parameters_correct.sizes) ≈ 0 atol=0.1
    @test maximum(atom_parameters.intensities .- atom_parameters_correct.intensities) ≈ 0 atol=0.1
    @test maximum(atom_parameters.angles .- atom_parameters_correct.angles) ≈ 0 atol=0.1
end