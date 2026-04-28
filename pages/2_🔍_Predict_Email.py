import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import re

st.title("🔍 Email Anomaly Prediction")

#=====================================LOAD SAVED FILES=============================================
model = joblib.load("E:\Desktop\GUVI\Project\Development of a Temporal-Aware NLP Pipeline/isolation_forest_model.pkl")
scaler = joblib.load('E:\Desktop\GUVI\Project\Development of a Temporal-Aware NLP Pipeline/scaler.pkl')
feature_cols = joblib.load(r"E:\Desktop\GUVI\Project\Development of a Temporal-Aware NLP Pipeline\feature_cols.pkl")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

#========================================Input====================================================
st.subheader('Enter Email Details')

email_text = st.text_area('Email Content')

sender = st.text_input('Sender Email Address')
receivers = st.text_input('Receivers Email Address(comma separated)')
hour = st.slider('Hour of Email', 0, 23, 12)
day_num = st.selectbox('Day of Week (0=Mon)', list(range(7)))
email_count_rolling = st.number_input('Recent Email Count', value = 1)

#======================================Cleaning===================================================
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

#=================================Feature Engineering=============================================
if st.button('Predict'):

    clean_body = clean_text(email_text)

    # Text Features(Bert)
    X_text = bert_model.encode([clean_body])
    
    #Behavioral Features
    email_length = len(email_text)
    word_count = len(email_text.split())
    num_recipients = len(receivers.split(',')) if receivers else 0

    # simple keyword flag
    keywords = ['deal', 'money', 'transfer', 'urgent', 'confidential']
    keyword_flag = int(any(k in clean_body for k in keywords))

    # Create dataframe
    X_other = pd.DataFrame([[
        email_length,
        word_count,
        num_recipients,
        keyword_flag,
        hour,
        day_num,
        email_count_rolling
    ]], columns = feature_cols)

    #Scale
    X_other_scaled = scaler.transform(X_other)

    #Combine
    X_final = np.hstack((X_text, X_other_scaled))

    #Predict
    label = model.predict(X_final)[0]
    score = model.decision_function(X_final)[0]

    #================Output=====================
    st.subheader('Result')

    if label == -1:
        st.error(f'🚨 Anomalous Email (Score:{score:.4f})')
    else:
        st.success(f'✅ Normal Email ({score:.4f})')