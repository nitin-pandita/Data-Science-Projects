# -*- coding: utf-8 -*

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('E:/Data Science Projects/Heart Disease Model/trained_model.sav', 'rb'))


# creating a function for Prediction

def heart_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
def main():
    
    
    # giving a title
    st.title('Heart Disease Prediction Web App')
    
    
    # getting the input data from the user
    
    
    age = st.text_input('Enter your age: ')
    sex = st.text_input('Enter your Gender 1 for male 0 for female: ')
    cp = st.text_input('Enter cp: ')
    trestbps = st.text_input('Enter trestbps: ')
    chol = st.text_input('Enter chol: ')
    fbs = st.text_input('Enter fbs: ')
    restecg = st.text_input('Enter restecg: ')
    thalach = st.text_input('Enter thalach: ')
    exang = st.text_input('Enter exang: ')
    oldpeak = st.text_input('Enter oldpeak: ')
    slope = st.text_input('Enter slop: ')
    ca = st.text_input('Enter ca: ')
    thal = st.text_input('Enter thal: ')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = heart_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()