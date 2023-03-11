# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 21:55:28 2023

@author: 96546
"""
import streamlit as st
# other libs
import numpy as np
import pandas as pd
import pickle
# import pyautogui # for reset button: pip install pyautogui

# load the model.pkl
# path = r"C:\Users\96546\Documents\何宇婷\学习\master\DSSI\LRmodel.pkl"
with open('LRmodel.pkl', "rb") as f:
	model = pickle.load(f)
    
    


@st.cache()
def prediction(sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm):
	# Making predictions
	prediction = model.predict([[sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm]])
	if prediction == 0:
		pred = 'setosa'
	else:
		pred = 'others (none setosa)'
	return pred



# putting the app related codes in main()
def main():
	# -- Set page config
	apptitle = 'iris data for prediction'
	st.set_page_config(page_title=apptitle, page_icon='random', 
		layout= 'wide', initial_sidebar_state="expanded")
	# random icons in the browser tab

	# give a title to your app
	st.title('Solution Implementation')
	#front end elements of the web page 
	# pick colors from: https://www.w3schools.com/tags/ref_colornames.asp
	html_temp = """ <div style ="background-color:AntiqueWhite;padding:15px"> 
       <h1 style ="color:black;text-align:center;">Predicting type of iris based on features</h1> 
       </div> <br/>"""

    #display the front end aspect
	st.markdown(html_temp, unsafe_allow_html = True)
	# let us make infrastructure to provide inputs
	# we will add the inputs to side bar
	st.sidebar.info('Provide input using the panel')
	st.info('Click Predict button below')

   ###################### declare inputs for UI #####################
	sepal_length_cm = st.sidebar.slider('sepal_length_cm', 0.1, 10.0, 0.1)
	st.write('input sepal length (cm)', sepal_length_cm)
	sepal_width_cm = st.sidebar.slider('sepal_width_cm', 0.1, 6.0, 0.1)
	st.write('input sepal_width_cm', sepal_width_cm)
	petal_length_cm = st.sidebar.slider('petal_length_cm', 0.1, 10.0, 0.1)
	st.write('input petal_length_cm', petal_length_cm)
	petal_width_cm = st.sidebar.slider('petal_width_cm', 0.1, 5.0, 0.1)
	st.write('input petal_width_cm', petal_width_cm)


   ###################### create a button to predict on UI ##############
	# assessment button
	if st.button("Predict"):
		assessment = prediction(sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm)
		st.success('**System assessment says:** {}'.format(assessment))


	st.success("App is working!!") # other tags include st.error, st.warning, st.help etc.

if __name__ == '__main__':
	main()