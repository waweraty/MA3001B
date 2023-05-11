import streamlit as st
import re
import pickle

@st.cache
def load_model():
        with open('modelo_ifs_2.pkl', 'rb') as f:
            model = pickle.load(f)
            f.close()

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