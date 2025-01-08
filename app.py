import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
#from PIL import Image



st.title("Spam SMS Classifier App")

# Load and preprocess data
sms_spam = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['Label', 'SMS'])
sms_spam['Label'] = sms_spam['Label'].map({'spam': 1, 'ham': 0})
sms_spam["SMS_Copy"] = sms_spam["SMS"]
ps = PorterStemmer()
sms_spam["SMS"] = sms_spam["SMS"].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))
X = sms_spam['SMS']
Y = sms_spam['Label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
feature_extraction = CountVectorizer(stop_words='english')
X_train_features_bow = feature_extraction.fit_transform(X_train)
Y_train = Y_train.astype('int')

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features_bow, Y_train)

# Create Streamlit app


#image = Image.open('techma.png')
#st.sidebar.image(image, width=120)
st.sidebar.header("User Input")
user_input = st.sidebar.text_area("Enter a message:", "Congratulations! You've won a $1000 cash prize. Claim your prize now by clicking the link: http://example.com/claim", height=300)

# Make prediction on the user input
input_data_features = feature_extraction.transform([user_input])
prediction = model.predict(input_data_features)

st.subheader("Prediction:")
if prediction[0] == 1:
    st.error("Spam SMS")
    st.subheader("Similar Spam Messages:")
    query_vec = feature_extraction.transform([user_input])
    word_similarity = cosine_similarity(query_vec, X_train_features_bow)[0]
    most_similar = sorted(list(enumerate(word_similarity)), reverse=True, key=lambda x: x[-1])[:5]
    for i in most_similar:
        st.write(sms_spam.iloc[X_train.index[i[0]], 2])
else:
    st.success("Non-Spam SMS")

