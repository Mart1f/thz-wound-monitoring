# S11 of a wave going through air -> plaster -> skin

import numpy as np
import matplotlib.pyplot as plt


# Parameters
c = 3e8                      # speed of light [m/s]
eps_aire = 1.0               # air permittivity
eps_piel = 4.0 - 3.0j        # skin permittivity
tan_delta = 0.05             # plaster loss tangent
eps_escayola = 2.5 * (1 - 1j * tan_delta)   # plaster permittivity
d = 5e-3                     # plaster thickness [m]
f = 275e9                    # frequency [Hz]


# Refractive indices
n1 = np.sqrt(eps_aire)
n2 = np.sqrt(eps_escayola)
n3 = np.sqrt(eps_piel)

print("n air     =", n1)
print("n plaster =", np.round(n2, 4))
print("n skin    =", np.round(n3, 4))


# Fresnel reflection coefficients (formule book)
r12 = (n1 - n2) / (n1 + n2)   # air -> plaster
r23 = (n2 - n3) / (n2 + n3)   # plaster -> skin

print("\n|r12| air->plaster  =", round(abs(r12), 3))
print("|r23| plaster->skin =", round(abs(r23), 3))


# Wavenumber in air (applying formule)
k0 = 2 * np.pi * f / c

# Round-trip propagation factor in the plaster (phase + attenuation)
viaje = np.exp(-2j * k0 * n2 * d)

print("\nRound-trip attenuation =",
      round(20 * np.log10(abs(viaje)), 2), "dB")


# Total reflection coefficient of a single layer (Ulaby, 2-10)
S11 = (r12 + r23 * viaje) / (1 + r12 * r23 * viaje)

# S11 in dB
S11_dB = 20 * np.log10(abs(S11))
print("\nS11 at 275 GHz, d = 5 mm:", round(S11_dB, 2), "dB")


# Plot 1: S11 vs frequency
frecuencias = np.linspace(220e9, 330e9, 1000)

plt.figure()
for td in [0.01, 0.05, 0.1]:
    n2 = np.sqrt(2.5 * (1 - 1j * td))
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    k0 = 2 * np.pi * frecuencias / c
    viaje = np.exp(-2j * k0 * n2 * d)
    S11 = (r12 + r23 * viaje) / (1 + r12 * r23 * viaje)
    plt.plot(frecuencias / 1e9, 20 * np.log10(abs(S11)), label=f"tan δ = {td}")

plt.xlabel("Frequency [GHz]")
plt.ylabel("|S11| [dB]")
plt.title("Air / 5 mm plaster / skin")
plt.grid()
plt.legend()
plt.savefig("simple_vs_frecuencia.png", dpi=150)


# Plot 2: S11 vs thickness
espesores = np.linspace(0, 10e-3, 1000)
k0 = 2 * np.pi * 275e9 / c

plt.figure()
for td in [0.01, 0.05, 0.1]:
    n2 = np.sqrt(2.5 * (1 - 1j * td))
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    viaje = np.exp(-2j * k0 * n2 * espesores)
    S11 = (r12 + r23 * viaje) / (1 + r12 * r23 * viaje)
    plt.plot(espesores * 1e3, 20 * np.log10(abs(S11)), label=f"tan δ = {td}")

plt.xlabel("Thickness [mm]")
plt.ylabel("|S11| [dB]")
plt.title("275 GHz")
plt.grid()
plt.legend()
plt.savefig("simple_vs_espesor.png", dpi=150)

plt.show()
