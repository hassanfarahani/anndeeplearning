import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('columnTransformer.pkl', 'rb') as file:
    ct = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = [[
    credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary
]]

# One-hot encode 'Geography'
input_data_transformed = np.array(ct.transform(input_data))

# Scale the input data
input_data_transformed[:, 5:]=scaler.transform(input_data_transformed[:, 5:])
input_data_scaled = input_data_transformed

input_data_scaled = input_data_scaled.astype('float32')
# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = float(prediction[0][0])

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
