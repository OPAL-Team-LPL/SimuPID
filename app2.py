import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

st.set_page_config(page_title="Simulation PID", layout="wide")
st.title("ENERGETIQUE 2 - Simulation PID")

# ===========================
# NOTATIONS
# ===========================
st.markdown("## 📘 Notations")
st.latex(r"G(p) = \frac{b_1 p + b_0}{a_2 p^2 + a_1 p + a_0}")
st.latex(r"H(p) = \frac{c_1 p + c_0}{d_2 p^2 + d_1 p + d_0}")
st.latex(r"C(p) = K_p\left(1 + \frac{1}{T_i p} + T_d p\right)")

# ===========================
# PID
# ===========================
st.sidebar.header("Paramètres PID")

Kp = st.sidebar.slider("Kp", 0.0, 50.0, 1.0, 0.1)
Ti = st.sidebar.slider("Ti", 0.001, 50.0, 1.0, 0.01)
Td = st.sidebar.slider("Td", 0.0, 10.0, 0.0, 0.01)

Ki = Kp / Ti
Kd = Kp * Td

# PID C(s) = Kp + Ki/s + Kd s
num_C = [Kd, Kp, Ki]
den_C = [1, 0]

# ===========================
# SYSTEME G
# ===========================
st.sidebar.header("Système G(p)")

b1 = st.sidebar.number_input("b1", value=0.0)
b0 = st.sidebar.number_input("b0", value=1.0)
a2 = st.sidebar.number_input("a2", value=1.0)
a1 = st.sidebar.number_input("a1", value=1.0)
a0 = st.sidebar.number_input("a0", value=1.0)

num_G = [b1, b0]
den_G = [a2, a1, a0]

# ===========================
# CAPTEUR H
# ===========================
st.sidebar.header("Capteur H(p)")

c1 = st.sidebar.number_input("c1", value=0.0)
c0 = st.sidebar.number_input("c0", value=1.0)
d2 = st.sidebar.number_input("d2", value=0.0)
d1 = st.sidebar.number_input("d1", value=0.0)
d0 = st.sidebar.number_input("d0", value=1.0)

num_H = [c1, c0]
den_H = [d2, d1, d0]

# ===========================
# BOUCLE FERMEE
# ===========================

# C*G
num_CG = np.polymul(num_C, num_G)
den_CG = np.polymul(den_C, den_G)

# C*G*H
num_CGH = np.polymul(num_CG, num_H)
den_CGH = np.polymul(den_CG, den_H)

# Fermeture : T = CG / (1 + CGH)
num_T = np.polymul(num_CG, den_H)
den_T = np.polyadd(den_CGH, num_CGH)

system = signal.TransferFunction(num_T, den_T)

# ===========================
# SIMULATION
# ===========================
st.markdown("## 📈 Réponse indicielle")

t = np.linspace(0, 20, 1000)
t_out, y = signal.step(system, T=t)

fig, ax = plt.subplots()
ax.plot(t_out, y)
ax.set_xlabel("Temps")
ax.set_ylabel("Réponse")
ax.grid(True)

st.pyplot(fig)

# ===========================
# INDICATEURS
# ===========================
st.markdown("## 📊 Indicateurs")

val_finale = float(y[-1])
y_max = float(np.max(y))

st.write("Valeur finale :", round(val_finale, 4))

if abs(val_finale) > 1e-8:
    depassement = (y_max - val_finale) / abs(val_finale) * 100
    st.write("Dépassement (%) :", round(depassement, 2))