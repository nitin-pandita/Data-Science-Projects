# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import numpy as np


loaded_model = pickle.load(open('E:\Data Science Projects/Heart Disease Model/trained_model.sav','rb')) 

input_data = (56,1,2,130,256,1,0,142,1,0.6,1,1,1)

# changing the input data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshaping the data
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshape)
print(prediction)

if(prediction[0] == 0):
    print("The Person don't have any heart Disease")
else:
    print('The Person is suffering from heart Disease')