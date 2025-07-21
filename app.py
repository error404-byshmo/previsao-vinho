import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("Qualidade do Vinho 🍷")

# Escolha do tipo de vinho
tipo_vinho = st.sidebar.selectbox("Tipo de vinho", ["Tinto", "Branco"])

# Carrega o modelo correto
if tipo_vinho == "Tinto":
    modelo = joblib.load("modelo_vinho_red.pkl")
    limites = {
        "fixed acidity": (4.0, 16.0, 7.4),
        "volatile acidity": (0.1, 1.5, 0.7),
        "citric acid": (0.0, 1.0, 0.0),
        "residual sugar": (0.5, 15.0, 1.9),
        "chlorides": (0.01, 0.2, 0.076),
        "free sulfur dioxide": (1, 72, 11),
        "total sulfur dioxide": (6, 289, 34),
        "density": (0.9900, 1.0050, 0.9978),
        "pH": (2.5, 4.5, 3.5),
        "sulphates": (0.3, 2.0, 0.56),
        "alcohol": (8.0, 15.0, 10.0)
    }
else:
    modelo = joblib.load("modelo_vinho_white.pkl")
    limites = {
        "fixed acidity": (4.0, 13.0, 6.9),
        "volatile acidity": (0.1, 1.0, 0.3),
        "citric acid": (0.0, 1.0, 0.3),
        "residual sugar": (0.5, 65.0, 5.0),
        "chlorides": (0.01, 0.15, 0.045),
        "free sulfur dioxide": (1, 72, 35),
        "total sulfur dioxide": (6, 440, 138),
        "density": (0.987, 1.004, 0.994),
        "pH": (2.8, 4.2, 3.2),
        "sulphates": (0.2, 1.2, 0.49),
        "alcohol": (8.0, 15.0, 10.5)
    }

# Cria os sliders dinamicamente com base nos limites definidos
st.sidebar.header("Características químicas do vinho")
dados = {}
for atributo, (min_v, max_v, default) in limites.items():
    if atributo == "density":
        dados[atributo] = st.sidebar.slider(atributo.capitalize(), float(min_v), float(max_v), float(default), step=0.0001)
    else:
        dados[atributo] = st.sidebar.slider(atributo.capitalize(), float(min_v), float(max_v), float(default))

dados_usuário = pd.DataFrame([dados])

st.subheader("Dados inseridos")
st.write(dados_usuário)

if st.button("Prever qualidade"):
    previsao = modelo.predict(dados_usuário)[0]
    st.subheader("Resultado da previsão")
    st.info(f"A qualidade prevista do vinho {tipo_vinho.lower()} é: **{previsao}**")

    if hasattr(modelo, "predict_proba"):
        prob = modelo.predict_proba(dados_usuário)[0]
        st.subheader("Distribuição das probabilidades")
        fig, ax = plt.subplots()
        ax.bar([str(i) for i in modelo.classes_], prob)
        ax.set_ylabel("Probabilidade")
        ax.set_xlabel("Qualidade")
        st.pyplot(fig)
