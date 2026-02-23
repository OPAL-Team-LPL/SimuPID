import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import control

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(page_title="Simulation PID", layout="wide")
st.title("ENERGETIQUE 2 - Simulation PID")
st.image("logo-sup-galilee.png")  # taille originale

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
st.markdown("### 🔁 Schéma bloc")
st.image("regulation.gif", use_container_width=True, caption="Boucle fermée PID")

# ===========================
# PID PARAMS
# ===========================
st.sidebar.header("Paramètres du PID")
mode_pid = st.sidebar.radio("Mode de réglage PID", ("Kp/Ki/Kd", "Kp/Ti/Td"))

# ---- Sliders PID ----
Kp = st.sidebar.slider("Kp", 0.0, 50.0, 1.0, 0.1)
st.sidebar.write(f"Kp = {Kp:.2f}")

if mode_pid == "Kp/Ki/Kd":
    Ki = st.sidebar.slider("Ki", 0.0, 1000.0, 10.0, 0.1)
    st.sidebar.write(f"Ki = {Ki:.2f}")
    Kd = st.sidebar.slider("Kd", 0.0, 50.0, 0.0, 0.01)
    st.sidebar.write(f"Kd = {Kd:.2f}")
    Ti = Kp / Ki if Ki != 0 else 1e6
    Td = Kd / Kp
else:
    Ti = st.sidebar.slider("Ti", 0.001, 1000.0, 1.0, 0.01)
    st.sidebar.write(f"Ti = {Ti:.3f}")
    Td = st.sidebar.slider("Td", 0.0, 50.0, 0.0, 0.01)
    st.sidebar.write(f"Td = {Td:.3f}")
    Ki = Kp / Ti if Ti != 0 else 1e6
    Kd = Kp * Td

# Filtre dérivatif
N = st.sidebar.slider("N (filtre dérivatif)", 1.0, 50.0, 10.0, 1.0)

# Correcteur PID
num_C = [Kp * Ti * Td * (1 + 1/N), Kp * Ti + Kp * Td / N, Kp]
den_C = [Td * Ti / N, Ti, 0]
C = control.TransferFunction(num_C, den_C)

# ===========================
# SYSTEME G(p)
# ===========================
st.sidebar.header("Paramètres du système G(p)")
b1 = st.sidebar.number_input("b1 (coef p)", value=0.0, step=0.1)
b0 = st.sidebar.number_input("b0 (constante)", value=1.0, step=0.1)
a2 = st.sidebar.number_input("a2 (coef p²)", value=1.0, step=0.1)
a1 = st.sidebar.number_input("a1 (coef p)", value=1.0, step=0.1)
a0 = st.sidebar.number_input("a0 (constante)", value=1.0, step=0.1)
G = control.TransferFunction([b1, b0], [a2, a1, a0])

# ===========================
# CAPTEUR H(p)
# ===========================
st.sidebar.header("Paramètres du capteur H(p)")
c1 = st.sidebar.number_input("c1 (coef p)", value=0.0, step=0.1)
c0 = st.sidebar.number_input("c0 (constante)", value=1.0, step=0.1)
d2 = st.sidebar.number_input("d2 (coef p²)", value=0.0, step=0.1)
d1 = st.sidebar.number_input("d1 (coef p)", value=0.0, step=0.1)
d0 = st.sidebar.number_input("d0 (constante)", value=1.0, step=0.1)
H = control.TransferFunction([c1, c0], [d2, d1, d0])

# ===========================
# BOUCLE FERMEE
# ===========================
L = control.series(C, G)
T = control.feedback(L, H)

# ===========================
# SIMULATION
# ===========================
st.markdown("---")
st.markdown("## 📈 Réponse indicielle")
t = np.linspace(0, 20, 5000)
u = np.ones_like(t)
t_out, y = control.forced_response(T, T=t, U=u)
y = np.asarray(y).flatten()

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
    st.error("La réponse contient des valeurs non définies (instabilité probable).")
else:
    val_finale = float(y[-1])
    y_max = float(np.max(y))
    st.write("Valeur finale :", round(val_finale, 4))
    st.write(f"Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}, Ti={Ti:.3f}, Td={Td:.3f}")

    if abs(val_finale) > 1e-8:
        depassement = (y_max - val_finale) / abs(val_finale) * 100
        st.write("Dépassement (%) :", round(depassement, 2))
        bande = 0.05 * abs(val_finale)
        indices = np.where(np.abs(y - val_finale) > bande)[0]
        tr5 = t_out[indices[-1]] if len(indices) > 0 else 0
        st.write("Temps de réponse à 5% :", round(float(tr5), 3), "s")
    else:
        st.write("Dépassement (%) : non défini")

    poles = control.poles(T)
    st.write("Pôles de la boucle fermée :", poles)
    if np.all(np.real(poles) < 0):
        st.success("Système stable")
    else:
        st.error("Système instable ou limite de stabilité")