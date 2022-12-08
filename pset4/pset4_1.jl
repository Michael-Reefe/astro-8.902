# Import modules
using CSV
using DataFrames
using Cosmology
using Distributions
using Optim
using PyPlot
using PyCall
using Printf
using ProgressMeter

plt.switch_backend("agg")
@pyimport mpl_toolkits.mplot3d as mplot3d

# Load in the catalog data from the CSV file
data = CSV.read("MCXC.tsv", DataFrame, delim='|', comment="#", skipto=4)

# Prepare our reference Flat LambdaCDM cosmology
cosmo = cosmology(h=0.70, OmegaM=0.3, OmegaK=0.0, OmegaR=0.0)

# Function for calculating the physical separation between two galaxy clusters
# given both of their cooridnates in decimal degrees, and their redshifts
function separation(ra1::Float64, dec1::Float64, z1::Float64, ra2::Float64, dec2::Float64, z2::Float64;
    method=:z)

    # First, get the angular separation using (ra,dec) as spherical coordinates
    cosβ = cos(dec1 * π/180) * cos(dec2 * π/180) * cos((ra2 - ra1) * π/180) + sin(dec1 * π/180) * sin(dec2 * π/180)

    if method == :z
        # Convert the redshifts to angular diameter distances using Flat LambdaCDM cosmology
        d1 = angular_diameter_dist(cosmo, z1).val
        d2 = angular_diameter_dist(cosmo, z2).val
    elseif method == :half
        d1 = angular_diameter_dist(cosmo, z1).val
        d2 = z2
    elseif method == :r
        d1 = z1
        d2 = z2
    else
        error("Unrecognized method $method")
    end

    # Get the physical separation √[(x-x')^2 + (y-y')^2 + (z-z')^2] using spherical coordinates
    sep = √(d1^2 + d2^2 - 2*d1*d2*cosβ)

    return sep

end

function get_pairsep(names::Union{Vector{String},Vector{String15}}, α::Vector{Float64}, δ::Vector{Float64}, z::Vector{Float64};
    method::Symbol=:z)
        
    # Get the separations between each pair of clusters
    n = length(names)
    # Total number of unique pairwise combinations
    n_pair = n*(n-1) ÷ 2   # (floor division)

    object1 = Vector{String}(undef, n_pair)
    object2 = Vector{String}(undef, n_pair)
    sep = Vector{Float64}(undef, n_pair)
    prog = Progress(n; showspeed=true)

    # k is the overall iterator
    k = 1
    # Iterate over i
    for i ∈ 1:n

        # Iterate over j > i
        for j ∈ (i+1):n

            # Add the object names and separation to the vectors
            object1[k] = names[i]
            object2[k] = names[j]
            sep[k] = separation(α[i], δ[i], z[i], α[j], δ[j], z[j]; method=method)
            
            # Increment k for every step
            k += 1

        end
        next!(prog)
    end

    separations = DataFrame(name_1=object1, name_2=object2, sep=sep)

    return separations
end

function get_crosspairs(names1::Union{Vector{String},Vector{String15}}, names2::Union{Vector{String},Vector{String15}},
    α1::Vector{Float64}, δ1::Vector{Float64}, z1::Vector{Float64},
    α2::Vector{Float64}, δ2::Vector{Float64}, z2::Vector{Float64}; 
    method::Symbol=:half)

    n1 = length(names1)
    n2 = length(names2)
    n_pair = n1 * n2

    object1 = Vector{String}(undef, n_pair)
    object2 = Vector{String}(undef, n_pair)
    sep = Vector{Float64}(undef, n_pair)
    prog = Progress(n1; showspeed=true) 

    k = 1
    for i ∈ 1:n1
        for j ∈ 1:n2
            object1[k] = names1[i]
            object2[k] = names2[j]
            sep[k] = separation(α1[i], δ1[i], z1[i], α2[j], δ2[j], z2[j]; method=method)
            k += 1
        end
        next!(prog)
    end

    separations = DataFrame(name_1=object1, name_2=object2, sep=sep)

    return separations
end

# Perform redshift cut
min_z = 0.02
max_z = 0.167
min_r = angular_diameter_dist(cosmo, min_z).val
max_r = angular_diameter_dist(cosmo, max_z).val

# Perform coordinate cut

filter = min_z .≤ data[!, :z] .≤ max_z

# Real data
println("Calculating real pairwise separations...")
separations_data = get_pairsep(data[filter, :MCXC], data[filter, :_RAJ2000], data[filter, :_DEJ2000], data[filter, :z], method=:z)

# Create the simulated data between lots of "clusters"
n_sim = sum(filter)
name_sim = ["sim_$i" for i ∈ 1:n_sim]
ni = 0
ra_sim = Vector{Float64}()
dec_sim = Vector{Float64}()
r_sim = Vector{Float64}()
while ni < n_sim
    global r_sim
    global ra_sim
    global dec_sim
    global ni

    xi, yi, zi = rand(Uniform(-max_r, max_r), 3)
    ri = √(xi^2 + yi^2 + zi^2)
    rai = atan(yi, xi) * 180/π
    deci = (π/2 - atan(√(xi^2 + yi^2), zi)) * 180/π
    if min_r ≤ ri ≤ max_r
        r_sim = [r_sim; ri]
        ra_sim = [ra_sim; rai]
        dec_sim = [dec_sim; deci]
        ni += 1
    end
end

println("Calculating simulated pairwise separations...")
separations_sim = get_pairsep(name_sim, ra_sim, dec_sim, r_sim, method=:r)

println("Calculating real-sim cross pairwise separations...")
separations_cross = get_crosspairs(data[filter, :MCXC], name_sim, data[filter, :_RAJ2000], data[filter, :_DEJ2000], data[filter, :z],
    ra_sim, dec_sim, r_sim, method=:half)

# Get histogram of separations
function histogram(values, bin_edges)
    n = length(bin_edges)-1
    bin_counts = zeros(n)
    for i ∈ 1:n
        bin_counts[i] = sum(bin_edges[i] .≤ values .< bin_edges[i+1])
    end
    return bin_counts
end


# Bin size (in Mpc)
Δr = 1
# Round maximum to nearest 10
min_val = 0.1
max_val = 600

# Make histograms of separations
bin_edges = exp10.(range(log10(min_val), log10(max_val), length=100))
# Geometric mean
bin_midpts = .√(bin_edges[1:end-1] .* bin_edges[2:end])

# bin_edges = min_val:Δr:max_val
# bin_midpts = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2

DD = histogram(separations_data[!, :sep], bin_edges)
RR = histogram(separations_sim[!, :sep], bin_edges)
DR = histogram(separations_cross[!, :sep], bin_edges)

nD = length(separations_data[!, :sep])
nR = length(separations_sim[!, :sep])

# Get the cumulative probabilities at each bin 
ξ = (nR ./ nD) .* (DD ./ DR) .- 1
# ξ = 1 ./ RR .* (DD .* (nR./nD).^2 .- 2 .* DR .* (nR./nD) .+ RR)
# ξ = (DD .* RR) ./ DR.^2 .- 1

filter = ξ .> 0 .&& isfinite.(ξ)
ξ = ξ[filter]
bin_midpts = bin_midpts[filter]
bin_left = bin_edges[1:end-1][filter]

function power_law(p, r)
    return (r ./ p[1]) .^ p[2]
end

function residuals(p)
    model = power_law(p, bin_left)
    return sum((ξ[bin_left .> 30] .- model[bin_left .> 30]).^2)
end

guess = [5.59, -1.84]
fit = optimize(residuals, [1., -Inf], [Inf, 0.], guess)
p_best = fit.minimizer

ξ_model = power_law(p_best, bin_left)
ξ_gal = power_law([5.59, -1.84], bin_left)

X = r_sim .* cos.(dec_sim .* π/180) .* cos.(ra_sim .* π/180)
Y = r_sim .* cos.(dec_sim .* π/180) .* sin.(ra_sim .* π/180)
Z = r_sim .* sin.(dec_sim .* π/180)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(X, Y, Z, c="r")
ax.set_xlabel("\$x\$")
ax.set_ylabel("\$y\$")
ax.set_zlabel("\$z\$")
ax.set_xlim(-max_r, max_r)
ax.set_ylim(-max_r, max_r)
ax.set_zlim(-max_r, max_r)
plt.savefig("random_points.pdf", dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots()
ax.stairs(DD ./ sum(DD), collect(bin_edges), color="r", linestyle="-", label="DD")
ax.stairs(RR ./ sum(RR), collect(bin_edges), color="b", linestyle="-", label="RR")
ax.stairs(DR ./ sum(DR), collect(bin_edges), color="g", linestyle="-", label="DR")
ax.set_xlim(minimum(bin_edges), maximum(bin_edges))
ax.set_xlabel("\$r\$ (Mpc)")
ax.set_ylabel("Bin Fraction")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-5, 0.2)
ax.legend()
plt.savefig("pairwise_separation.pdf", dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots()
ax.loglog(bin_left, ξ, "k.")
ax.loglog(bin_left, ξ_model, "r-", label="\$(r/$(@sprintf "%.2f" p_best[1]){\\rm\\ Mpc})^{$(@sprintf "%.2f" p_best[2])}\$")
ax.loglog(bin_left, ξ_gal, "g-", label="\$(r/5.59{\\rm\\ Mpc})^{-1.84}\$")
# ax.plot(collect(min_val:1:max_val), ξ_gal, "b-", label="\$(r/5{\\rm\\ Mpc})^{-1.67}\$")
# ax.set_xlim(minimum(bin_edges), maximum(bin_edges))
ax.legend()
# ax.set_ylim(-1, maximum(ξ)+1)
ax.set_xlabel("\$r\$ (Mpc)")
ax.set_ylabel("\$\\xi(r)\$")
plt.savefig("2pt_correlation.pdf", dpi=300, bbox_inches="tight")
plt.close()
