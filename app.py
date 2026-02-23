import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import control

st.set_page_config(page_title="Simulation PID", layout="wide")
st.title("Simulation PID - Réponse indicielle")

# ==========================================================
# NOTATIONS
# ==========================================================
st.markdown("## 📘 Notations")

st.markdown("**Fonction de transfert du système :**")
st.latex(r"G(p) = \frac{N_G(p)}{D_G(p)}")

st.markdown("**Fonction de transfert du capteur :**")
st.latex(r"H(p) = \frac{N_H(p)}{D_H(p)}")

st.markdown("**Fonction de transfert du correcteur PID :**")
st.latex(r"C(p) = K_p\left(1 + \frac{1}{T_i p} + \frac{T_d p}{1 + \frac{T_d}{N}p}\right)")

st.markdown("**Fonction de transfert en boucle fermée :**")
st.latex(r"T(p) = \frac{C(p)G(p)}{1 + C(p)G(p)H(p)}")

st.markdown("### 🔁 Schéma bloc d'une boucle de régulation")

st.image(
    "regulation.gif",
    caption="Schéma bloc d'une boucle fermée avec correcteur, système et capteur",
    use_container_width=True
)

# ==========================================================
# PARAMETRES PID
# ==========================================================
st.sidebar.header("Paramètres du PID")

Kp = st.sidebar.slider("Kp", 0.0, 50.0, 1.0)
Ki = st.sidebar.slider("Ki", 0.0, 1000.0, 10.0)
Td = st.sidebar.slider("Td", 0.0, 5.0, 0.0)
N = st.sidebar.slider("N (filtre dérivatif)", 1.0, 20.0, 10.0)

Ti = Kp / Ki if Ki != 0 else 1e6

num_C = [Kp * Ti * Td * (1 + (1/N)), Kp * Ti + Kp * Td/N , Kp]
den_C = [Td*Ti/N, Ti, 0]
C = control.TransferFunction(num_C, den_C)

# ==========================================================
# TYPE DE SYSTEME
# ==========================================================
st.sidebar.header("Type de système")

type_systeme = st.sidebar.selectbox(
    "Choisir le type de système",
    ("1er ordre", "2ème ordre")
)

st.sidebar.header("Paramètres du système G(p)")
gain_G = st.sidebar.number_input("Gain N_G", value=10.0)

if type_systeme == "1er ordre":
    st.sidebar.markdown("D_G(p) = a p + b")
    a = st.sidebar.number_input("a (coef p)", value=10.0)
    b = st.sidebar.number_input("b (constante)", value=1.0)
    num_G = [gain_G]
    den_G = [a, b]
else:
    st.sidebar.markdown("D_G(p) = a p² + b p + c")
    a = st.sidebar.number_input("a (coef p²)", value=1.0)
    b = st.sidebar.number_input("b (coef p)", value=10.0)
    c = st.sidebar.number_input("c (constante)", value=1.0)
    num_G = [gain_G]
    den_G = [a, b, c]

G = control.TransferFunction(num_G, den_G)

# ==========================================================
# CAPTEUR
# ==========================================================
st.sidebar.header("Paramètres du capteur H(p)")

gain_H = st.sidebar.number_input("Gain N_H", value=1.0)
alpha = st.sidebar.number_input("α (coef p)", value=0.0)

num_H = [gain_H]
den_H = [alpha, 1]

H = control.TransferFunction(num_H, den_H)

# ==========================================================
# BOUCLE FERMEE
# ==========================================================
L = control.series(C, G)
T = control.feedback(L, H)

# ==========================================================
# SIMULATION
# ==========================================================
t = np.linspace(0, 20, 5000)
u = np.ones_like(t)

t_out, y = control.forced_response(T, T=t, U=u)

# Correction du problème NamedSignal
y = np.asarray(y).flatten()

fig, ax = plt.subplots()
ax.plot(t_out, y)
ax.set_xlabel("Temps")
ax.set_ylabel("Réponse")
ax.set_title("Réponse indicielle")
ax.grid(True)

st.pyplot(fig)

# ==========================================================
# INDICATEURS
# ==========================================================
st.markdown("## 📊 Indicateurs de performance")

if np.any(np.isnan(y)) or np.any(np.isinf(y)):
    st.error("La réponse contient des valeurs non définies (instabilité probable).")
else:
    val_finale = float(y[-1])
    y_max = float(np.max(y))

    st.write("Valeur finale :", round(val_finale, 4))

    if val_finale != 0:
        depassement = (y_max - val_finale) / abs(val_finale) * 100
        depassement = max(depassement, 0)
        st.write("Dépassement (%) :", round(float(depassement), 2))
    else:
        st.write("Dépassement (%) : non défini (valeur finale nulle)")

    poles = control.poles(T)

    if np.all(np.real(poles) < 0):
        st.success("Système stable (tous les pôles ont une partie réelle négative)")
    else:
        st.error("Système instable ou limite de stabilité")