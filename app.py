import streamlit as st
import pandas as pd
import joblib

#  PRIMER comando de Streamlit
st.set_page_config(page_title="¿Es legendario?", page_icon="🐉")

# =========================
# Cargar modelo
# =========================
@st.cache_resource
def load_model():
    return joblib.load("modelo_app_pokemon.pkl")

model = load_model()

app_features = [
    "base_total",
    "base_egg_steps",
    "base_happiness",
    "is_genderless",
    "experience_growth",
    "capture_rate",
    "hp",
    "sp_attack",
    "sp_defense",
    "generation",
]

# =========================
# Interfaz
# =========================
st.title("¿Este Pokémon sería legendario? 🐉✨")
st.write(
    "App basada en un modelo de regresión logística entrenado con datos de Pokémon."
)

st.sidebar.header("Características del Pokémon")

base_total = st.sidebar.slider("Total de stats base (base_total)", 200, 780, 500)
base_egg_steps = st.sidebar.slider("Pasos para eclosionar", 1000, 40000, 10000, step=500)
base_happiness = st.sidebar.slider("Felicidad base", 0, 140, 70)
is_genderless = st.sidebar.checkbox("Sin género (is_genderless)", value=False)
experience_growth = st.sidebar.slider("Experiencia para crecer", 600000, 1640000, 1000000, step=50000)
capture_rate = st.sidebar.slider("Tasa de captura (capture_rate)", 3, 255, 45)
hp = st.sidebar.slider("HP", 1, 255, 80)
sp_attack = st.sidebar.slider("Ataque especial (sp_attack)", 10, 200, 90)
sp_defense = st.sidebar.slider("Defensa especial (sp_defense)", 10, 200, 90)
generation = st.sidebar.selectbox("Generación", options=[1, 2, 3, 4, 5, 6, 7])

input_df = pd.DataFrame([{
    "base_total": base_total,
    "base_egg_steps": base_egg_steps,
    "base_happiness": base_happiness,
    "is_genderless": 1 if is_genderless else 0,
    "experience_growth": experience_growth,
    "capture_rate": capture_rate,
    "hp": hp,
    "sp_attack": sp_attack,
    "sp_defense": sp_defense,
    "generation": generation,
}], columns=app_features)

st.subheader("Características seleccionadas")
st.write(input_df)

if st.button("Calcular probabilidad 🧠"):
    proba = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]

    st.subheader("Resultado del modelo")
    st.metric("Probabilidad de ser legendario", f"{proba*100:.1f} %")

    if pred == 1:
        st.success("El modelo clasifica este Pokémon como **LEGENDARIO** ✨")
    else:
        st.info("El modelo clasifica este Pokémon como **NO legendario**.")
