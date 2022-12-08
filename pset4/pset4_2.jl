using Roots
using Cosmology
using QuadGK

Ωm = 0.3
Ωr = 0.0
ΩK = 0.0
ΩΛ = 0.7
cosmo = cosmology(h=0.7, OmegaM=Ωm, OmegaK=ΩK, OmegaR=Ωr)

# Constants 
η = 5e-10           # unitless
k = 8.61733326e-11  # MeV/K
mₙ = 939.56542052   # MeV

# Function to find the roots for
f(kT) = 1 - 6.7η * (kT / mₙ)^(3/2) * exp(1.0 / kT)

kT = find_zero(f, (0.001, 1.))
T = kT/k

# Convert temperature to age in seconds - Carroll & Ostlie eqn 29.89
t = (T/1e10)^(-2)

println("Deuterium fusion begins at kT = $kT MeV")
println("This corresponds to T = $T K")
println("This happens at t = $t s")