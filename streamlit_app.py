import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from PIL import Image
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')

stem = PorterStemmer()
lem = nltk.WordNetLemmatizer()
stop_word = nltk.corpus.stopwords.words("english")

model = pickle.load(open("disastertweetsmodel", "rb"))
countvect = pickle.load(open("countvectorizer", "rb"))

st.sidebar.title("Disaster Tweets")
st.sidebar.write("This page is created for classifying tweets whether they are related with disasters or not.")
st.sidebar.write("Model is trained with data from Natural Language Processing with Disaster Tweets competition in Kaggle.")

st.write("\n")
st.write("\n")
st.write("\n")
tweet = st.text_area("Please enter your tweet in this area!!", height=200)

def converter(x):
    """This function applies multiple tasks sequentially"""
    
    # Tokenize the words
    process_1 = word_tokenize(x.lower())
    
    # Removing "#" from text 
    process_2 = [re.sub("#", "", i) for i in process_1]
    
    # Removing non-alphanumeric strings
    process_3 = [i for i in process_2 if i.isalpha()]
    
    # Removing stopwords
    process_4 = [i for i in process_3 if i not in stop_word]
    
    # Lemmatize words
    process_5 = [lem.lemmatize(i) for i in process_4]
    
    process_6 = [i for i in process_5 if i not in ["http", "https"]]
    
    return " ".join(process_6)

if st.button("Search"):
    new_tweet = converter(tweet)
    final_tweet = countvect.transform(np.array([new_tweet]))
    result = model.predict(final_tweet)
    if list(result)[0] == 0:
        st.success("This tweet is not related with disasters."+":tada:")
    else:
        st.warning("This tweet is related with disasters!!!"+":rotating_light:"*3)
    
        if "earthquake" or "aftershock" in new_tweet:
            im = Image.open("earthquake.jpg")
            st.image(im, width=700, caption="Earthquake")
    
        elif "flood" in new_tweet:
            im = Image.open("flood.jpg")
            st.image(im, width=700, caption="Flood")
    
        elif "fire" in new_tweet:
            im = Image.open("fire.jpg")
            st.image(im, width=700, caption="Fire")

        elif "evacuation" in new_tweet:
            im = Image.open("evacuation.jpg")
            st.image(im, width=700, caption="Evacuation")
    
        elif "tornado" in new_tweet:
            im = Image.open("tornado.jpg")
            st.image(im, width=700, caption="Tornado")

        elif "heat wave" in new_tweet:
            im = Image("heat wave.jpg")
            st.image(im, width=700, caption="Heat Wave")

        elif "ablaze" in new_tweet:
            im = Image.open("ablaze.jpg")
            st.image(im, width=700, caption="Ablaze")
            
        elif "car crash"  or "crash" in new_tweet:
            im = Image.open("car crash.jpg")
            st.image(im, width=700, caption="Car Crash")
            
        elif "accident" in new_tweet:
            im = Image.open("accident.jpg")
            st.image(im, width=700, caption="Accident")
        
        elif "shoot" in new_tweet:
            im = Image.open("shooting.jpg")
            st.image(im, width=700, caption="Shooting")



