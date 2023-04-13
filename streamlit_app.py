import streamlit as st
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow import keras
import datetime
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.metrics import r2_score
from keras.preprocessing.image import ImageDataGenerator

@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None},ttl=6*3600)
def predict_value(image,model):   
	img_width, img_height = 512, 512
	img = keras.preprocessing.image.load_img(image)
	img = tf.image.central_crop(img, central_fraction=0.5)
	img = tf.image.resize(img,[img_width, img_height])
	img = tf.image.resize(img,[150, 150])
	#img = keras.utils.img_to_array(img)

	img = np.expand_dims(img, axis = 0)
	res=model.predict(img/255)
	
	return res

@st.cache
def load_data(path):
	df=pd.read_csv(path)
	df = df.set_index('Time')
	df.index = pd.to_datetime(df.index)
	df.sort_index(inplace = True)
	df2 = df[~df.index.duplicated(keep='first')]
	return (df,df2)

st.header('GRIP Team')
st.header("Stage predictor using images")
st.write("Upload image to get its corresponding stage")

uploaded_file = st.file_uploader("Choose an image...")

df, df2=load_data('small_df.csv')

Classifier=keras.models.load_model('Classifier')
CNN=keras.models.load_model('CNN', custom_objects = {"r2_score": r2_score})
VGG=keras.models.load_model('VGG', custom_objects = {"r2_score": r2_score})

if uploaded_file is not None:
	#src_image = load_image(uploaded_file)
	upimage = Image.open(uploaded_file)	

	st.image(upimage, caption='Input Image', use_column_width=True)

	pred_class=predict_value(uploaded_file,Classifier)


	if int(pred_class[0][0].round()==0):
		st.write('Image loaded is not suitable for the prediction model')
		d = st.date_input("Please select the date of the Stage to load",
		datetime.date(2012, 6, 9),
		min_value=datetime.date(2012, 6, 9),
		max_value=datetime.date(2019, 10, 11))
		t = st.time_input('Select the time for that date',datetime.time(0, 0))
		#st.write('The closest Stage to your date is:', d,t)	
		if st.button('Select'):	
			date=datetime.datetime.combine(d,t)
			idx=df2.index.get_indexer([date], method='nearest')
			s = df2.iloc[idx,1]
			st.write('The closest Stage to your date is: ',s[0], ' which corresponds to: ',s.index[0])
	elif int(pred_class[0][0].round()==1):
		st.write('Predicted Stage:')
		if uploaded_file.name in df['Filename'].unique():
			st.write('Real:',df.loc[df['Filename'].isin([uploaded_file.name])]['Stage'][0])

		predVGG=predict_value(uploaded_file,VGG)
		predCNN=predict_value(uploaded_file,CNN)
		st.write('Our model:',round(predCNN[0][0],2))
		st.write('VGG:',round(predVGG[0][0],2))
else:
	sampleimg='StateLineWeir_20180915_Farrell_160.jpg'
	upimage=Image.open(f'SampleIMG/{sampleimg}')
	st.image(upimage, caption='Sample Image', use_column_width=True)

	st.write('Predicted Stage:')
	st.write('Real:',df.loc[df['Filename'].isin([sampleimg])]['Stage'][0])

	predVGG=predict_value(f'SampleIMG/{sampleimg}',VGG)
	predCNN=predict_value(f'SampleIMG/{sampleimg}',CNN)
	st.write('Our model:',round(predCNN[0][0],2))
	st.write('VGG:',round(predVGG[0][0],2))
