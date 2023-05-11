import streamlit as st
import pickle

@st.cache
def load_model():
        pkl = open("modelo_ifs.pkl", "rb")
        model = pickle.load(pkl)

        return model


program_categorizer = load_model()

st.set_page_config(page_title="Predictor Unitario", page_icon="1")

st.markdown("# Predictor Unitario")
st.sidebar.header("Predictor Unitario")

st.write(
    """En esta aplicación se ingresa el nombre del programa para predecir su categoría"""
)

text_input = st.text_input('Nombre del programa', placeholder="Favor de ingresar el nombre del programa")

if text_input:
        pred=program_categorizer.categorize_program(text_input)

        st.write("Nombre del programa: ", text_input)
        st.write('Categoría predicha: ', pred)
        st.write('Categoría real')