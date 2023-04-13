import streamlit as st

st.set_page_config(page_title="Predictor Unitario", page_icon="1")

st.markdown("# Predictor Unitario")
st.sidebar.header("Predictor Unitario")

st.write(
    """En esta aplicación se ingresa el nombre del programa para predecir su categoría"""
)

text_input = st.text_input('Nombre del programa', placeholder="Favor de ingresar el nombre del programa")

if text_input:
        st.write("Nombre del programa: ", text_input)
        st.write('Categoría predicha:')
        st.write('Categoría real')