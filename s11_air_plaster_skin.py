"""
Task 1 - Normal-incidence reflection coefficient S11 of an air / plaster / skin structure.

Structure (plane wave, normal incidence):
    medium 1: air            (semi-infinite)
    medium 2: plaster cast   (thickness d)
    medium 3: skin           (semi-infinite)

Time convention: exp(+j*omega*t)  ->  eps = eps' - j*eps'',  propagation exp(-j*k0*n*z),
so the complex refractive index n = sqrt(eps) must have Im(n) <= 0.

Reference: Ulaby & Long (2014), Sections 2-6 and 2-10.
"""

import numpy as np
import matplotlib.pyplot as plt

C0 = 299_792_458.0  # speed of light in vacuum [m/s]

# Illustrative permittivities (from contenu_dossier.docx)
EPS_AIR = 1.0
EPS_SKIN = 4.0 - 3.0j
EPS_PLASTER_REAL = 2.5
TAN_DELTAS = [0.01, 0.05, 0.1]


def refractive_index(eps):
    """Complex refractive index with Im(n) <= 0 (exp(+jwt) convention)."""
    n = np.sqrt(np.asarray(eps, dtype=complex))
    return np.where(n.imag > 0, np.conj(n), n)


def fresnel_normal(n_i, n_j):
    """Reflection coefficient at the interface i -> j, normal incidence."""
    return (n_i - n_j) / (n_i + n_j)


def s11_single_slab(freq, d, eps_slab, eps_sub, eps_inc=EPS_AIR):
    """
    S11 of incident medium / slab (thickness d) / semi-infinite substrate.

    Gamma = (G12 + G23*exp(-2j*k0*n2*d)) / (1 + G12*G23*exp(-2j*k0*n2*d))

    freq and d can be scalars or numpy arrays (broadcasting).
    """
    n1 = refractive_index(eps_inc)
    n2 = refractive_index(eps_slab)
    n3 = refractive_index(eps_sub)
    k0 = 2 * np.pi * np.asarray(freq) / C0
    g12 = fresnel_normal(n1, n2)
    g23 = fresnel_normal(n2, n3)
    phase = np.exp(-2j * k0 * n2 * np.asarray(d))
    return (g12 + g23 * phase) / (1 + g12 * g23 * phase)


def to_db(x):
    return 20 * np.log10(np.abs(x))


def sanity_checks():
    """Limit cases that the model must satisfy."""
    n_air, n_skin = refractive_index(EPS_AIR), refractive_index(EPS_SKIN)
    eps_p = EPS_PLASTER_REAL * (1 - 0.05j)

    # 1) d = 0 -> plaster disappears, Gamma = air/skin Fresnel coefficient
    g_d0 = s11_single_slab(275e9, 0.0, eps_p, EPS_SKIN)
    assert np.isclose(g_d0, fresnel_normal(n_air, n_skin)), "d=0 check failed"

    # 2) very thick lossy slab -> only the air/plaster interface is seen
    g_thick = s11_single_slab(275e9, 1.0, eps_p, EPS_SKIN)
    g_ap = fresnel_normal(n_air, refractive_index(eps_p))
    assert np.isclose(g_thick, g_ap), "thick-slab check failed"

    # 3) slab with same permittivity as skin -> air/skin Fresnel coefficient
    g_same = s11_single_slab(275e9, 5e-3, EPS_SKIN, EPS_SKIN)
    assert np.isclose(g_same, fresnel_normal(n_air, n_skin)), "matched-slab check failed"

    # 4) passive structure -> |S11| <= 1
    f = np.linspace(220e9, 330e9, 501)
    assert np.all(np.abs(s11_single_slab(f, 5e-3, eps_p, EPS_SKIN)) <= 1), "passivity check failed"

    print("Sanity checks passed.")
    print(f"  |Gamma| air/skin    = {abs(fresnel_normal(n_air, n_skin)):.3f} "
          f"({to_db(fresnel_normal(n_air, n_skin)):.2f} dB)")
    print(f"  |Gamma| air/plaster = {abs(g_ap):.3f} ({to_db(g_ap):.2f} dB)")


def print_link_budget(f=275e9, d=5e-3):
    """Rough estimate: how strong is the skin echo compared to the surface echo."""
    n_air, n_skin = refractive_index(EPS_AIR), refractive_index(EPS_SKIN)
    k0 = 2 * np.pi * f / C0
    print(f"\nLink budget at {f/1e9:.0f} GHz, d = {d*1e3:.1f} mm:")
    for tand in TAN_DELTAS:
        n2 = refractive_index(EPS_PLASTER_REAL * (1 - 1j * tand))
        g12, g23 = fresnel_normal(n_air, n2), fresnel_normal(n2, n_skin)
        att_db = to_db(np.exp(-2j * k0 * n2 * d))  # two-way slab loss
        skin_echo_db = to_db((1 - g12**2) * g23) + att_db
        print(f"  tan_d = {tand:<5}: two-way loss = {att_db:7.2f} dB | "
              f"surface echo = {to_db(g12):6.2f} dB | skin echo = {skin_echo_db:7.2f} dB")
    n2 = refractive_index(EPS_PLASTER_REAL)
    print(f"  Ripple period: {C0/(2*n2.real*d)/1e9:.1f} GHz in frequency, "
          f"{C0/f/(2*n2.real)*1e3:.3f} mm in thickness")


def main():
    sanity_checks()
    print_link_budget()

    freq = np.linspace(220e9, 330e9, 2001)
    thick = np.linspace(0, 10e-3, 2001)
    d_fixed, f_fixed = 5e-3, 275e9

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    for tand in TAN_DELTAS:
        eps_p = EPS_PLASTER_REAL * (1 - 1j * tand)
        ax1.plot(freq / 1e9, to_db(s11_single_slab(freq, d_fixed, eps_p, EPS_SKIN)),
                 label=rf"$\tan\delta$ = {tand}")
        ax2.plot(thick * 1e3, to_db(s11_single_slab(f_fixed, thick, eps_p, EPS_SKIN)),
                 label=rf"$\tan\delta$ = {tand}")

    ax1.set(title=f"Air / plaster ({d_fixed*1e3:.0f} mm) / skin - normal incidence",
            xlabel="Frequency [GHz]", ylabel=r"$|S_{11}|$ [dB]")
    ax2.set(title=f"Air / plaster / skin at {f_fixed/1e9:.0f} GHz",
            xlabel="Plaster thickness [mm]", ylabel=r"$|S_{11}|$ [dB]")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.4)
        ax.legend()
    fig.tight_layout()
    fig.savefig("s11_air_plaster_skin.png", dpi=150)
    print("\nFigure saved to s11_air_plaster_skin.png")
    plt.show()


if __name__ == "__main__":
    main()
