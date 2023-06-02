import streamlit as st
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
import datetime
import numpy as np
import os
import pandas as pd
from sklearn.metrics import r2_score


st.set_page_config(
    page_title="Hola",
    page_icon="👋",
)

st.title('🌼🌼 Equipo 1 🌼🌼')
st.write('Ernesto Borbón, Gerardo Villegas, José de Jesús Gutiérrez, Luis Felipe Villaseñor')

st.write("## Visualizador y Predictor de Categoría de Programas Para Market Development Funds")
st.write('''
        HP, el socio formador, desea saber a dónde están destinados sus fondos para las campañas de marketing.
        Para esto, ellos se apoyan de un documento de Excel donde tienen todos los datos de estas campañas, con los cuales se 
        busca identificar el tipo de campaña según su título de programa. Esta clasificación se ha estado llevando de manera manual,
        lo que representa una gran pérdida de tiempo para la empresa, por lo que es fundamental encontrar la
        manera de automatizar este proceso, reduciendo así los tiempos de clasificación sin sacrificar el desempeño de la
        clasificación de los programas.\n

        Para resolver esta problemática, se realizó un modelo hecho a mano basado en reglas más representativas de cada categoría,
        al igual que realizar una jerarquía de importancia de estas reglas. Con este modelo se logró categorizar correctamente el 100%
        de los datos para la variable categórica de *activity_subtype* y el 99%% de los datos fueron correctamente categorizados para 
        la variable categórica de *activity_subtype_id*.\n


        ''')