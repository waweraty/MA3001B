import streamlit as st
import pandas as pd
from modelo_ifs import *

st.set_page_config(page_title="Predictor Múltiple", page_icon="🦾")

@st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

program_categorizer= ProgramCategorizer()

st.markdown("# Predictor Múltiple")
st.sidebar.header("Predictor Múltiple")

st.write(
    """En esta aplicación se ingresa un archivo .csv o de Excel para predecir múltiples instancias y
    se regresa un archivo con una nueva columna con la categoría predicha"""
)

uploaded_file = st.file_uploader("Sube el archivo .csv con columna program_name", type=['csv'])

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)

    edited_df = st.experimental_data_editor(dataframe['program_name'], num_rows="dynamic")

    if st.button('Predecir valores'):
        pred=program_categorizer.categorize_program(dataframe['program_name'])
        dataframe['Predictions']=pred
        dataframe
        
        csv = convert_df(dataframe)

        st.download_button(
            "Descargar csv",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
            )