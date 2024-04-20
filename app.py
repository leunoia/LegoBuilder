#imports for app
import panel as pn
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import random
import param
import streamlit as st

#imports for model 
import pandas as pd
import string
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku

#read data
lego = pd.read_csv('lego_data_clean_translated.csv')
lego.head()

toy_name_en = lego['toy_name_en'].values
print(toy_name_en)

def clean_text(txt):
    txt = "".join(t for t in txt if t not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii", 'ignore')
    return txt

toy_name_en_clean = [clean_text(x) for x in toy_name_en]
toy_name_en_clean[:10]

tokenizer = Tokenizer()
def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    ## convert data to a token sequence
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(toy_name_en_clean)
inp_sequences[:10]

def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer):
    # If the seed_text is empty, randomly pick a word from the tokenizer's word index
    if seed_text == "":
        seed_text = random.choice(list(tokenizer.word_index.keys()))

    seed_text += " " + random.choice(list(tokenizer.word_index.keys()))

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted, axis=-1)  # Get the index of the maximum prediction

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text.title()

# Load the saved TensorFlow/Keras model --> USING lego_nlp_model_new.ipynb
modelH5 = tf.keras.models.load_model('nlp_model.h5')
modelKeras = tf.keras.models.load_model('nlp_model.keras')

st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#HTML CONTENT
styles = {
    'background-color': '#F6F6F6', 'border': '2px solid black',
    'border-radius': '5px', 'padding': '10px'
} 

#GET USER INPUT 
# Define global variables
global seedtext
global numw
seedtext = ""
numw = 1

# def add(event):
#     seedtext = f"You selected {select.value}"
#     return choice

def updateSeed(event):
    choice = seedUI.value
    #print(seedtext)
    return choice

def updateNW(event):
    choice = nwUI.value
    #print(numw)
    return choice



# Create TextInput and IntInput widgets
seedUI = pn.widgets.TextInput(name='Enter text to generate a cool lego name', placeholder='Enter string here')
nwUI = pn.widgets.IntInput(name='Enter the number of words for our LegoBuilder to generate name', value=0, step=1, start=0, end=10, width=400)

#Get use input
#seedres = pn.bind(updateSeed, val=seedUI, x=seedtext)
seedres = pn.bind(updateSeed, seedUI)
nmwres = pn.bind(updateNW, nwUI)
#button = pn.widgets.Button(name="Confirm")

setname = generate_text(seedres(), nmwres(), modelH5, max_sequence_len, tokenizer)
print(setname)

import os
import openai
# from openai import OpenAI
from app_utils import API_KEY
openai.api_key = API_KEY

#GENERATE IMAGE USING NLP MODEL
response = openai.Image.create(
  model="dall-e-3",
  prompt="Generate image of a Lego set with the box in the background with the title: " + setname + " DO NOT GENERATE ANY COPYRIGHT CONTENT",
  n=1,
    #   size='512x512',
  quality="standard"
)
image_url = response.data[0].url

image_url = 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-2qYe8WP5V1Bj4t8QqB4ezmit/user-yipV9ZqeKHB5Jhj262EOamDg/img-XgRcgqB4aXpP2xDSjbFX0LZQ.png?st=2024-04-20T17%3A04%3A37Z&se=2024-04-20T19%3A04%3A37Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2024-04-20T07%3A00%3A13Z&ske=2024-04-21T07%3A00%3A13Z&sks=b&skv=2021-08-06&sig=6Y2dGPqnI3zCpzK3SudDlgeKXgCJuFdC9W%2BXh8g9mV4%3D'
print(image_url)
#PANEL
picture = pn.pane.Image(
    image_url, sizing_mode='scale_width'
)
#picture.servable()

# #DEFINE TEMPLATE 
# template = pn.template.FastListTemplate(
#     title="LegoBuilder",
#     #sidebar=["Hello Sidebar", "This is text for the"],
#     main=[layout],
# )
# # template.main.append(picture)
# # #RENDER APP
# template.servable()
layout = pn.Column(seedUI, nwUI, seedres,nmwres, picture)
layout.servable()

 