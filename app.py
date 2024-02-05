import streamlit as st
import pickle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import sqlite3
import pandas as pd

# Sample function for preprocessing input text (e.g., tokenization, vectorization)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)


# Function to predict whether the input text is spam or ham and store in the database
def predict_and_store(text, model):
    preprocessed_text = preprocess_text(text)

    # Load the model and the TF-IDF vectorizer
    if model == ':rainbow[Naive Bayes]':
        file_name = 'saved_nb_model_vectorizer.pkl'
    else:
        file_name = 'saved_model_with_vectorizer.pkl'

    with open(file_name, 'rb') as file:
        classifier_model = pickle.load(file)
        tfidf_vectorizer = pickle.load(file)

    # Transform the input text
    X = tfidf_vectorizer.transform([preprocessed_text])

    # Make prediction
    prediction = classifier_model.predict(X)[0]

    # Convert numeric prediction to text
    if prediction == 0:
        prediction_text = 'Ham'
    else:
        prediction_text = 'Spam'

    # Store the response in the database
    conn = sqlite3.connect('spam_ham_responses.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            model TEXT,
            prediction TEXT
        )
    ''')
    cursor.execute('INSERT INTO responses (text, model, prediction) VALUES (?, ?, ?)',
                   (text, model, prediction_text))
    conn.commit()
    conn.close()

    return prediction_text


# Function to display analytics
def display_analytics():
    # Connect to the database
    conn = sqlite3.connect('spam_ham_responses.db')
    cursor = conn.cursor()

    # Retrieve data for analytics
    cursor.execute('SELECT model, prediction, COUNT(*) FROM responses GROUP BY model, prediction')
    results = cursor.fetchall()
    conn.close()

    # Display analytics
    st.title('Analytics')
    st.subheader('Number of Responses:')
    df = pd.DataFrame(results, columns=['Model', 'Prediction', 'Count'])

    # Display overall counts of Ham and Spam
    ham_count = df[df['Prediction'] == 'Ham']['Count'].sum()
    spam_count = df[df['Prediction'] == 'Spam']['Count'].sum()

    st.write(f'Overall Ham Count: {ham_count}')
    st.write(f'Overall Spam Count: {spam_count}')

    # Bar chart for model-wise predictions
    st.subheader('Model-wise Predictions')
    st.bar_chart(df.pivot(index='Model', columns='Prediction', values='Count'))


# Streamlit app
st.title('Spam or Ham Classifier')

user_input = st.text_input('Enter a text to classify')

# Radio button for model selection
model = st.radio(
    "Which model will you choose: ",
    [":rainbow[Naive Bayes]", ":rainbow[SVM]"])

# Main prediction and analytics display
if st.button('Predict'):
    if user_input:
        prediction = predict_and_store(user_input, model)

        # Display prediction result in the "Model" section
        st.title('Prediction')
        if prediction == 'Ham':
            st.write('Ham')
        else:
            st.write('Spam')

        # Display analytics in the "Analytics" section
        display_analytics()

    else:
        st.write('Please enter a text to classify')
