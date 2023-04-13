import streamlit as st

st.set_page_config(page_title="Predictor Múltiple", page_icon="🦾")

st.markdown("# Predictor Múltiple")
st.sidebar.header("Predictor Múltiple")

st.write(
    """En esta aplicación se ingresa un archivo .csv o de Excel para predecir múltiples instancias y
    se regresa un archivo con una nueva columna con la categoría predicha"""
)