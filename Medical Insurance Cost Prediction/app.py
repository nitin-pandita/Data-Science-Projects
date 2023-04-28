import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('E:\Data Science Projects\Medical Insurance Cost Prediction\Model.pkl', 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction[0])
  
    
  
def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user
    
    # age,sex,bmi,children,smoker,region,charges
    age = st.text_input('Age')
    sex = st.text_input('Sex')
    bmi = st.text_input('BMI')
    children = st.text_input('Children')
    smoker = st.text_input('Smoker')
    region = st.text_input('Region')


    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([age,sex,bmi,children,smoker,region])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    