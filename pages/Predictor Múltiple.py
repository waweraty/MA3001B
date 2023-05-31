import streamlit as st
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import os
import urllib.request
import pickle
import umap
import umap.umap_ as umap_
import re
import nltk
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import braycurtis
import plotly.express as px
from modelo_ifs import *

st.set_page_config(page_title="Predictor Múltiple", page_icon="🦾")

@st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_resource
def start_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    lemmatizer = WordNetLemmatizer()

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
    u=pd.read_pickle('embedding.csv')
    return u

@st.cache_data
def get_data():
    data=pd.read_pickle('datos_sd.csv')
    data.reset_index(drop = True, inplace = True)
    return data

@st.cache_data
def get_pred():
    pred=pd.read_csv('df_pred.csv')
    pred['program_name']=pred['program_name'].str.strip()
    pred.drop_duplicates(subset='program_name',inplace=True)
    return pred

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
    pred=get_pred()
    vectorizer = get_vect()
    u=umap.values

    textoarray = (vectorizer.transform(texto)).toarray()
    distarr=[braycurtis(r, textoarray[0]) for r in data.iloc[:,:-3].values]
    res = sorted(range(len(distarr)), key=lambda sub: distarr[sub])[:K]

    u_consulta = fit.transform(textoarray)
    tu_consulta_row = pd.DataFrame({'program_name':texto[0], cattype:'Tu Consulta'},index=[0])

    fig = px.scatter(pred.loc[pred['program_name'].isin(data.iloc[res]['program_name'].values)], x=u[res,0], y=u[res,1], color=('pred_'+cattype), hover_name="program_name", log_x=False)
    fig.update_traces(marker_size=10)  # Set the initial marker size for all points

    # Create a separate trace for the "Tu Consulta" point with a bigger marker size
    tu_consulta_trace = px.scatter(tu_consulta_row, x=[u_consulta[0][0]], y=[u_consulta[0][1]], color=cattype, hover_name="program_name")

    # Update the marker size for the "Tu Consulta" trace
    tu_consulta_trace.update_traces(marker=dict(size=20, color = 'OrangeRed', line=dict(color='Crimson', width=3), symbol = 'star'))

    # Add the "Tu Consulta" trace to the figure
    fig.add_traces(tu_consulta_trace.data)
    fig.update_layout(title=("10 programas con títulos más similares a tu consulta, por " + cattype))
    st.plotly_chart(fig, use_container_width=True)

start_nltk()
download_umap()
data = get_data()
u = get_embedding()
fit= get_umap()

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

    edited_df = st.experimental_data_editor(dataframe['program_name'].to_frame(), num_rows="dynamic", use_container_width = True)

else:
    df_empty = pd.DataFrame({'program_name' : []})
    df_empty['program_name']=df_empty['program_name'].astype('str')
    edited_df = st.experimental_data_editor(df_empty, num_rows = "dynamic", use_container_width = True)


col1, inter_cols_pace, col2 = st.columns((4, 8, 2))

with col1:
    button_pred=st.button('Predecir valores', key='but_p', disabled= edited_df.empty)

with col2:
    button_vis=st.button('Visualizar', key='but_v', disabled= edited_df.empty)

if button_pred:
    st.session_state['button'] = False

    pred=program_categorizer.categorize_program(edited_df['program_name'])
    pred=pred.add_prefix('pred_')
    df=pd.concat([edited_df,pred],axis=1)
    st.dataframe(df, use_container_width = True)
    #dataframe=dataframe.merge(df, how='outer', on='program_name')
    #st.write(dataframe)
    
    csv = convert_df(df)

    st.download_button(
        "Descargar csv",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
        )


if st.session_state.get('button') != True:
    st.session_state['button'] = button_vis

if st.session_state['button']==True:
    K = st.slider('Selecciona el número de vecinos', 1, 100, 10)

    if st.button('Correr visualización'):
        visKnear(data, edited_df['program_name'].values,u, 'activity_subtype', fit, K)
        visKnear(data, edited_df['program_name'].values,u, 'activity_subtype_id', fit, K)
