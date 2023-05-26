import streamlit as st
import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import urllib.request
import pickle
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from scipy.spatial.distance import braycurtis
import plotly.express as px

st.set_page_config(page_title="Dashboard", page_icon="📈")

st.markdown("# Dashboard")
st.sidebar.header("Dashboard")

st.write(
    """Primeras visualizaciones"""
)

@st.cache_resource
def download_umap():
	url='https://github.com/waweraty/MA3001B/releases/download/1.0.0/umap.pickle'

	filename = url.split('/')[-1]
	if not os.path.exists(filename):
		urllib.request.urlretrieve(url, filename)
	
@st.cache_resource
def get_umap():
    fit=pickle.load(open('umap.pickle','rb'))
    return fit

@st.cache_resource
def get_vect():
    vectorizer=pickle.load(open('vectorizador_pro3000.pickle','rb'))
    return vectorizer

@st.cache_data
def get_embedding():
    u=pickle.load(open('embedding.csv','rb'))
    return u

@st.cache_data
def get_data():
    data=pickle.load(open('data_sd.csv','rb'))
    data.reset_index(drop = True, inplace = True)
    return data

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\d+\w*\b|\b\w*\d+\b', '', text)
    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)

    lemmas = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tagged_words]
    text = ' '.join(lemmas)
    return text

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a' # adjective
    elif treebank_tag.startswith('V'):
        return 'v' # verb
    elif treebank_tag.startswith('N'):
        return 'n' # noun
    elif treebank_tag.startswith('R'):
        return 'r' # adverb
    else:
        return 'n' # assume noun as default

def visKnear(data, texto, umap,cattype, fit, K=10):
    vectorizer = get_vect()

    umap=umap.values
    textoarray = (vectorizer.transform(texto)).toarray()
    distarr=[braycurtis(r, textoarray[0]) for r in data.iloc[:,:-3].values]
    res = sorted(range(len(distarr)), key=lambda sub: distarr[sub])[:K]

    u_consulta = fit.transform(textoarray)
    tu_consulta_row = pd.DataFrame({'program_name':texto[0], cattype:'Tu Consulta'},index=[0])

    fig = px.scatter(data.iloc[res], x=umap[res,0], y=umap[res,1], color=cattype, hover_name="program_name", log_x=False)
    fig.update_traces(marker_size=10)  # Set the initial marker size for all points

    # Create a separate trace for the "Tu Consulta" point with a bigger marker size
    tu_consulta_trace = px.scatter(tu_consulta_row, x=[u_consulta[0][0]], y=[u_consulta[0][1]], color=cattype, hover_name="program_name")

    # Update the marker size for the "Tu Consulta" trace
    tu_consulta_trace.update_traces(marker=dict(size=20, color = 'OrangeRed', line=dict(color='Crimson', width=3), symbol = 'star'))

    # Add the "Tu Consulta" trace to the figure
    fig.add_traces(tu_consulta_trace.data)
    fig.update_layout(title=("10 programas con títulos más similares a tu consulta, por " + cattype))
    fig.show()

download_umap()

data = get_data()
u = get_embedding()
fit= get_umap()
st.write(data)

st.write(
    """En esta aplicación se ingresa el nombre del programa para ubicar a los 10 programas con títulos más cercanos"""
)

text_input = st.text_input('Nombre del programa', placeholder="Favor de ingresar el nombre del programa")

if text_input:
    visKnear(data, [text_input], u, 'activity_subtype', fit, 10)
    visKnear(data, [text_input], u, 'activity_subtype_id', fit, 10)