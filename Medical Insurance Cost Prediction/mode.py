
import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('E:\Data Science Projects\Medical Insurance Cost Prediction\Model.pkl', 'rb'))


input_data = (19,1,27.9,0,1,1)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction[0])
