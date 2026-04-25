import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ✅ Download required resources (first time only, cached after that)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)   # 🔥 important for Python 3.12+
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# Function for preprocessing text
def transform_text(text):
    text = text.lower()                       # lowercase
    text = nltk.word_tokenize(text)           # tokenize

    # keep only alphanumeric
    y = [i for i in text if i.isalnum()]

    # remove stopwords & punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("📧 Email Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first!")
    else:
        # preprocess
        transformed_sms = transform_text(input_sms)

        # vectorize
        vector_input = tfidf.transform([transformed_sms])

        # predict
        result = model.predict(vector_input)[0]

        # display
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")
