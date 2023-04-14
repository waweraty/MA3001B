import streamlit as st
import numpy as np
import time
import pandas as pd
import seaborn as sns

st.set_page_config(page_title="Dashboard", page_icon="📈")

st.markdown("# Dashboard")
st.sidebar.header("Dashboard")

st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

@st.cache_data
def get_data():
    df = pd.read_csv('datosMDF_US.csv')
    return df

df = get_data()
st.write(df)

st.pyplot(sns.countplot(x=df["activity_subtype_id"]))


progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")