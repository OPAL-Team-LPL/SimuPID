import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import control
import os

# ===========================
# CONFIGURATION PAGE
# ===========================
st.set_page_config(page_title="Simulation PID", layout="wide")
st.title("ENERGETIQUE 2 - Simulation PID")

# ===========================
# IMAGES (si présentes)
# ===========================
if os.path.exists("logo-sup-galilee.png"):
    st.image("logo-sup-galilee.png")

# ===========================
# NOTATIONS
# ===========================
st.markdown("## 📘 Notations")

st.markdown("**Fonction de transfert du système :**")
st.latex(r"G(p) = \frac{b_1 p + b_0}{a_2 p^2 + a_1 p + a_0}")

st.markdown("**Fonction de transfert du capteur :**")
st.latex(r"H(p) = \frac{c_1 p + c_0}{d_2 p^2 + d_1 p + d_0}")

st.markdown("**Fonction de transfert du correcteur PID :**")
st.latex(r"C(p) = K_p\left(1 + \frac{1}{T_i p} + \frac{T_d p}{1 + \frac{T_d}{N}p}\right)")

st.markdown("**Fonction de transfert en boucle fermée :**")
st.latex(r"T(p) = \frac{C(p)G(p)}{1 + C(p)G(p)H(p)}")

if os.path.exists("regulation.gif"):
    st.image("regulation.gif", use_container_width=True, caption="Boucle fermée PID")

# ===========================
# PARAMÈTRES PID
# ===========================
st.sidebar.header("Paramètres du PID")

mode_pid = st.sidebar.radio("Mode de réglage PID", ("Kp/Ki/Kd", "Kp/Ti/Td"))

Kp = st.sidebar.slider("Kp", 0.0, 50.0, 1.0, 0.1)
st.sidebar.write(f"Kp = {Kp:.2f}")

if mode_pid == "Kp/Ki/Kd":
    Ki = st.sidebar.slider("Ki", 0.0, 1000.0, 10.0, 0.1)
    Kd = st.sidebar.slider("Kd", 0.0, 50.0, 0.0, 0.01)

    Ti = Kp / Ki if Ki != 0 else 1e6
    Td = Kd / Kp if Kp != 0 else 0

else:
    Ti = st.sidebar.slider("Ti", 0.001, 1000.0, 1.0, 0.01)
    Td = st.sidebar.slider("Td", 0.0, 50.0, 0.0, 0.01)

    Ki = Kp / Ti if Ti != 0 else 1e6
    Kd = Kp * Td

N = st.sidebar.slider("N (filtre dérivatif)", 1.0, 50.0, 10.0, 1.0)

# ===========================
# CORRECTEUR PID
# ===========================
num_C = [Kp * Ti * Td * (1 + 1/N), Kp * Ti + Kp * Td / N, Kp]
den_C = [Td * Ti / N, Ti, 0]

C = control.TransferFunction(num_C, den_C)

# ===========================
# SYSTÈME G(p)
# ===========================
st.sidebar.header("Paramètres du système G(p)")

b1 = st.sidebar.number_input("b1", value=0.0, step=0.1)
b0 = st.sidebar.number_input("b0", value=1.0, step=0.1)
a2 = st.sidebar.number_input("a2", value=1.0, step=0.1)
a1 = st.sidebar.number_input("a1", value=1.0, step=0.1)
a0 = st.sidebar.number_input("a0", value=1.0, step=0.1)

G = control.TransferFunction([b1, b0], [a2, a1, a0])

# ===========================
# CAPTEUR H(p)
# ===========================
st.sidebar.header("Paramètres du capteur H(p)")

c1 = st.sidebar.number_input("c1", value=0.0, step=0.1)
c0 = st.sidebar.number_input("c0", value=1.0, step=0.1)
d2 = st.sidebar.number_input("d2", value=0.0, step=0.1)
d1 = st.sidebar.number_input("d1", value=0.0, step=0.1)
d0 = st.sidebar.number_input("d0", value=1.0, step=0.1)

H = control.TransferFunction([c1, c0], [d2, d1, d0])

# ===========================
# BOUCLE FERMÉE
# ===========================
L = control.series(C, G)
T = control.feedback(L, H)

# ===========================
# SIMULATION
# ===========================
st.markdown("---")
st.markdown("## 📈 Réponse indicielle")

t = np.linspace(0, 20, 1000)
u = np.ones_like(t)

try:
    t_out, y = control.forced_response(T, T=t, U=u)
    y = np.asarray(y).flatten()
except Exception as e:
    st.error(f"Erreur simulation : {e}")
    y = np.zeros_like(t)
    t_out = t

fig, ax = plt.subplots()
ax.plot(t_out, y)
ax.set_xlabel("Temps")
ax.set_ylabel("Réponse")
ax.set_title("Réponse indicielle")
ax.grid(True)

st.pyplot(fig)

# ===========================
# INDICATEURS
# ===========================
st.markdown("## 📊 Indicateurs de performance")

if np.any(np.isnan(y)) or np.any(np.isinf(y)):
    st.error("Réponse non définie (instabilité probable)")
else:
    val_finale = float(y[-1])
    y_max = float(np.max(y))

    st.write("Valeur finale :", round(val_finale, 4))
    st.write(f"Kp={Kp:.2f} | Ki={Ki:.2f} | Kd={Kd:.2f}")

    if abs(val_finale) > 1e-8:
        depassement = (y_max - val_finale) / abs(val_finale) * 100
        st.write("Dépassement (%) :", round(depassement, 2))

        bande = 0.05 * abs(val_finale)
        indices = np.where(np.abs(y - val_finale) > bande)[0]
        tr5 = t_out[indices[-1]] if len(indices) > 0 else 0

        st.write("Temps de réponse à 5% :", round(float(tr5), 3), "s")
    else:
        st.write("Dépassement : non défini")