# =============================================================================
# VERSIÓN SIMPLE / VERSÃO SIMPLES
# Reflexión de una onda en: aire -> escayola -> piel
# Reflexão de uma onda em: ar -> gesso -> pele
# =============================================================================

import numpy as np                 # cálculos con números complejos / cálculos com números complexos
import matplotlib.pyplot as plt    # gráficas / gráficos


# -----------------------------------------------------------------------------
# PARTE 1: DATOS DEL PROBLEMA / DADOS DO PROBLEMA
# -----------------------------------------------------------------------------

c = 3e8          # velocidad de la luz [m/s] / velocidade da luz [m/s]

# Permitividad de cada material (número complejo, j = imaginario en Python)
# Permissividade de cada material (número complexo, j = imaginário em Python)
eps_aire = 1.0               # aire / ar
eps_piel = 4.0 - 3.0j        # piel (dada por el profesor) / pele (dada pelo professor)
tan_delta = 0.05             # pérdidas de la escayola (prueba 0.01, 0.05, 0.1)
                             # perdas do gesso (teste 0.01, 0.05, 0.1)
eps_escayola = 2.5 * (1 - 1j * tan_delta)   # escayola / gesso

d = 5e-3         # espesor de la escayola = 5 mm / espessura do gesso = 5 mm
f = 275e9        # frecuencia = 275 GHz / frequência = 275 GHz


# -----------------------------------------------------------------------------
# PARTE 2: ÍNDICE DE REFRACCIÓN  n = raíz(eps)
#          ÍNDICE DE REFRAÇÃO    n = raiz(eps)
# -----------------------------------------------------------------------------
# n dice cuánto se frena la onda (parte real) y cuánto se atenúa (parte imaginaria).
# n diz quanto a onda desacelera (parte real) e quanto ela atenua (parte imaginária).
# Con la convención exp(+jwt), la parte imaginaria de n debe ser NEGATIVA.
# Com a convenção exp(+jwt), a parte imaginária de n deve ser NEGATIVA.
# np.sqrt ya devuelve la raíz con parte imaginaria negativa aquí.
# np.sqrt já devolve a raiz com parte imaginária negativa aqui.

n1 = np.sqrt(eps_aire)        # aire / ar
n2 = np.sqrt(eps_escayola)    # escayola / gesso
n3 = np.sqrt(eps_piel)        # piel / pele

print("n aire / ar       =", n1)
print("n escayola / gesso =", np.round(n2, 4))
print("n piel / pele     =", np.round(n3, 4))


# -----------------------------------------------------------------------------
# PARTE 3: REFLEXIÓN EN CADA FRONTERA (Fresnel)
#          REFLEXÃO EM CADA FRONTEIRA (Fresnel)
# -----------------------------------------------------------------------------
# Cuando la onda pasa de un material a otro, una parte rebota:
# Quando a onda passa de um material para outro, uma parte reflete:
#       r = (n_antes - n_después) / (n_antes + n_después)
#       r = (n_antes - n_depois)  / (n_antes + n_depois)

r12 = (n1 - n2) / (n1 + n2)   # frontera aire -> escayola / fronteira ar -> gesso
r23 = (n2 - n3) / (n2 + n3)   # frontera escayola -> piel / fronteira gesso -> pele

print("\n|r12| aire->escayola / ar->gesso   =", round(abs(r12), 3))
print("|r23| escayola->piel / gesso->pele =", round(abs(r23), 3))


# -----------------------------------------------------------------------------
# PARTE 4: EL VIAJE DENTRO DE LA ESCAYOLA (ida y vuelta)
#          A VIAGEM DENTRO DO GESSO (ida e volta)
# -----------------------------------------------------------------------------
# k0 = número de onda en el aire / número de onda no ar
k0 = 2 * np.pi * f / c

# La onda recorre 2*d (va hasta la piel y vuelve).
# A onda percorre 2*d (vai até a pele e volta).
# Este factor hace DOS cosas / Este fator faz DUAS coisas:
#   - gira la fase (por la parte real de n2)      -> oscilaciones / oscilações
#   - reduce la amplitud (por la parte imag. de n2) -> pérdidas / perdas
viaje = np.exp(-2j * k0 * n2 * d)

print("\nAtenuación ida y vuelta / Atenuação ida e volta =",
      round(20 * np.log10(abs(viaje)), 2), "dB")


# -----------------------------------------------------------------------------
# PARTE 5: REFLEXIÓN TOTAL S11 (suma de todos los rebotes)
#          REFLEXÃO TOTAL S11 (soma de todas as reflexões)
# -----------------------------------------------------------------------------
# Fórmula de una capa (Ulaby, sección 2-10):
# Fórmula de uma camada (Ulaby, seção 2-10):
#       S11 = (r12 + r23*viaje) / (1 + r12*r23*viaje)
# El numerador = eco de la superficie + eco de la piel.
# O numerador = eco da superfície + eco da pele.
# El denominador = los rebotes múltiples dentro de la escayola.
# O denominador = as reflexões múltiplas dentro do gesso.

S11 = (r12 + r23 * viaje) / (1 + r12 * r23 * viaje)

# Pasar a dB (20 porque es amplitud de campo, no potencia)
# Converter para dB (20 porque é amplitude de campo, não potência)
S11_dB = 20 * np.log10(abs(S11))
print("\nS11 a 275 GHz, d = 5 mm:", round(S11_dB, 2), "dB")


# -----------------------------------------------------------------------------
# PARTE 6: GRÁFICA 1 -> S11 vs FRECUENCIA (d = 5 mm)
#          GRÁFICO 1 -> S11 vs FREQUÊNCIA (d = 5 mm)
# -----------------------------------------------------------------------------
# Repetimos el cálculo para muchas frecuencias y para cada tan_delta.
# Repetimos o cálculo para muitas frequências e para cada tan_delta.

frecuencias = np.linspace(220e9, 330e9, 1000)   # 1000 puntos / 1000 pontos

plt.figure()
for td in [0.01, 0.05, 0.1]:
    n2 = np.sqrt(2.5 * (1 - 1j * td))           # nueva escayola / novo gesso
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    k0 = 2 * np.pi * frecuencias / c             # ahora k0 es un vector / agora k0 é um vetor
    viaje = np.exp(-2j * k0 * n2 * d)
    S11 = (r12 + r23 * viaje) / (1 + r12 * r23 * viaje)
    plt.plot(frecuencias / 1e9, 20 * np.log10(abs(S11)), label=f"tan δ = {td}")

plt.xlabel("Frecuencia / Frequência [GHz]")
plt.ylabel("|S11| [dB]")
plt.title("Aire / escayola 5 mm / piel  —  Ar / gesso 5 mm / pele")
plt.grid()
plt.legend()
plt.savefig("simple_vs_frecuencia.png", dpi=150)


# -----------------------------------------------------------------------------
# PARTE 7: GRÁFICA 2 -> S11 vs ESPESOR (f = 275 GHz)
#          GRÁFICO 2 -> S11 vs ESPESSURA (f = 275 GHz)
# -----------------------------------------------------------------------------
espesores = np.linspace(0, 10e-3, 1000)   # de 0 a 10 mm / de 0 a 10 mm
k0 = 2 * np.pi * 275e9 / c                # frecuencia fija / frequência fixa

plt.figure()
for td in [0.01, 0.05, 0.1]:
    n2 = np.sqrt(2.5 * (1 - 1j * td))
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    viaje = np.exp(-2j * k0 * n2 * espesores)   # ahora cambia el espesor / agora muda a espessura
    S11 = (r12 + r23 * viaje) / (1 + r12 * r23 * viaje)
    plt.plot(espesores * 1e3, 20 * np.log10(abs(S11)), label=f"tan δ = {td}")

plt.xlabel("Espesor / Espessura [mm]")
plt.ylabel("|S11| [dB]")
plt.title("275 GHz")
plt.grid()
plt.legend()
plt.savefig("simple_vs_espesor.png", dpi=150)

# -----------------------------------------------------------------------------
# CÓMO LEER LAS GRÁFICAS / COMO LER OS GRÁFICOS
# -----------------------------------------------------------------------------
# - Oscilaciones grandes = el eco de la piel llega fuerte -> SÍ vemos la piel.
#   Oscilações grandes = o eco da pele chega forte -> SIM vemos a pele.
# - Curva plana (~ -13 dB) = solo vemos la superficie -> NO vemos la piel.
#   Curva plana (~ -13 dB) = só vemos a superfície -> NÃO vemos a pele.

plt.show()
