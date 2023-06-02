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
import plotly.express as px
from plotly.validators.scatter.marker import SymbolValidator
from modelo_ifs import *

st.set_page_config(page_title="Predicción y Visualización", page_icon="🦾")

@st.experimental_memo
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_resource
def start_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    lemmatizer = WordNetLemmatizer()
    return lemmatizer

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
    pred=pred.iloc[:,1:]
    pred.reset_index(drop = True, inplace = True)
    return pred

@st.cache_data
def transform_input(df):
    vectorizer = get_vect()
    textoarray = (vectorizer.transform(df['program_name'])).toarray()
    u_consulta = fit.transform(textoarray)

    return (textoarray, u_consulta)

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

def visKnear(data, texto, umap, u_consulta,cattype, textoarray, K=10):
    pred=get_pred()
    u=umap.values

    dists=np.stack([np.sum(np.absolute(r-textoarray), axis=1)/np.sum(np.absolute(r+textoarray), axis=1) for r in data.iloc[:,:-3].values],axis=1)
    res = [sorted(range(len(d)), key=lambda sub: d[sub])[:K] for d in dists]
    fullpred=pd.DataFrame()

    for i,r in enumerate(res):
        smp=pred.iloc[r].copy()
        smp['n_cons']=i
        fullpred=pd.concat([fullpred, smp])

    tu_consulta_row = pd.DataFrame({'program_name':texto, cattype:'Tu Consulta'},index=np.arange(len(texto)))
    resf=[item for sublist in res for item in sublist]

    fig = px.scatter(fullpred, x=u[resf,0], y=u[resf,1], color=('pred_'+cattype), hover_name="program_name", log_x=False,symbol='n_cons')
    fig.update_traces(marker=dict(size=10))

    # Create a separate trace for the "Tu Consulta" point with a bigger marker size

    tu_consulta_trace = px.scatter(tu_consulta_row, x=u_consulta[:,0], y=u_consulta[:,1], color=cattype, hover_name="program_name",symbol=tu_consulta_row.index)
    tu_consulta_trace.update_traces(marker=dict(size=15, color = 'OrangeRed', line=dict(color='Crimson', width=3)))
    fig.add_traces(tu_consulta_trace.data)

    fig.update_layout(title=(str(K)+ " programas con títulos más similares a tu consulta, por "+ cattype))
    #fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

lemmatizer=start_nltk()
download_umap()
data = get_data()
u = get_embedding()
fit= get_umap()

program_categorizer= ProgramCategorizer()

st.markdown("# Predicción y Visualización")
st.sidebar.header("Predicción y Visualización")

st.write(
    """En esta aplicación se ingresa un archivo .csv para predecir múltiples instancias o visualizar estos 
    junto con sus vecinos, también si lo desea, puede ingresar manualmente uno o más nombres de programa
    que desee.
    """
)

uploaded_file = st.file_uploader("Sube el archivo .csv con columna program_name", type=['csv'])
st.write('Esta tabla de abajo es editable')

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

if np.logical_and(st.session_state['button']==True, 0<edited_df.shape[0]<=55):
    textoarray, u_consulta = transform_input(edited_df)

    K = st.slider('Selecciona el número de vecinos', 1, 100, 10)

    if st.button('Correr visualización'):
        visKnear(data, edited_df['program_name'].values,u, u_consulta, 'activity_subtype', textoarray, K)
        visKnear(data, edited_df['program_name'].values,u, u_consulta, 'activity_subtype_id', textoarray, K)

elif np.logical_and(st.session_state['button']==True,edited_df.shape[0]>55):
    st.error('Error: No se pueden visualizar más de 55 datos a la vez')
