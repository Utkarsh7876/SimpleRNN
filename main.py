#Step 1: Import all the libraries and load the model

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


#Load the imdb dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}

#Load the pre trained model with Relu activation 
model = load_model('simplernn_imdb_model.h5')

##Step 2:Helper Functions 
#function to decode reviews
def decode_review(text):
    return ' '.join([reverse_word_index.get(i-3, '?')for i in encoded_review])

#Function to preprocess user input

def preprocess_text(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word, 2)+ 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# ##Step 3: Get user input and make predictions
# def predict_sentiment(review):
#     preprocessed_input=preprocess_text(review)

#     prediction = model.predict(preprocessed_input)

#     sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'

#     return sentiment, prediction[0][0]


import streamlit as st

##streamlit app

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as posititive or negative.")

#user input
user_input=st.text_area("Movie Review")

if st.button("Classify"):
    preprocessed_input=preprocess_text(user_input)

    #Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'

    #display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]:.4f}')
else:
    st.write('Please enter a movie review.')