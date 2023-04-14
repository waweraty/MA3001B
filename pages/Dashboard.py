import streamlit as st
import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard", page_icon="📈")

st.markdown("# Dashboard")
st.sidebar.header("Dashboard")

st.write(
    """Primeras visualizaciones"""
)

@st.cache_data
def get_data():
    df = pd.read_csv('datosMDF_US.csv')
    return df

df = get_data()
st.write(df)

fig, ax = plt.subplots()
fig.set_size_inches(10, 15)
sns.countplot(x=df["activity_subtype_id"],ax=ax)
st.pyplot(fig)