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

st.header('Ramos Blancos')
st.header("Predictor de Categoría de Programas Para Market Development Funds")
st.write('Favor de seleccionar la aplicación deseada en el menú de la izquierda')