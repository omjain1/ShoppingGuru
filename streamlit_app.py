import streamlit as st
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import openai
import toml
import os
from google.oauth2 import service_account
from streamlit_chat import message
api = st.sidebar.text_input("Your API Key", type="password")

openai.api_key= st.secrets["openai"]["api_key"]
# Load the service account credentials from the downloaded JSON file

credentials_path = 'portfolio-388719-602e459f77d8.json'
credentials = service_account.Credentials.from_service_account_file(
    credentials_path, scopes=['https://www.googleapis.com/auth/drive']
)

# Set the credentials as environment variables for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
import pandas as pd
import gdown

# Define the file ID of the CSV file on Google Drive
file_id = '159VHNpovIyJfISBtbRnY2oCoIs2HrAPY'
file_url = f'https://drive.google.com/uc?id={file_id}'

file_id1 = '19NALCJBXeNJDPD7SmyLDxlyIlRyVQaO6'
file_url1 = f'https://drive.google.com/uc?id={file_id}'

# Download the file into memory
response = gdown.download(file_url, quiet=False)

# Load the CSV file using pandas
df_train_cleaned = pd.read_csv(response)

# Download the file into memory
response1 = gdown.download(file_url1, quiet=False)

# Load the CSV file using pandas
df_val_cleaned = pd.read_csv(response1)



# df_train_cleaned = pd.read_csv('X_train_cleaned.csv')
X_train_cleaned_series = pd.Series(df_train_cleaned['text'].dropna())

# df_val_cleaned = pd.read_csv('X_val_cleaned.csv')
X_val_cleaned_series = pd.Series(df_val_cleaned['text'].dropna())

# df_X_val_padded = pd.read_csv('X_val_padded.csv')
# df_X_train_padded = pd.read_csv('X_train_padded.csv')

max_words = 10000
max_sequence_length = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_cleaned_series)
embedding_dim = 100
lstm_units = 128
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(units=lstm_units))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the pre-trained model weights
model.load_weights('ltsm_sentiment.h5')

def generate_response(prompt, conversation_memory):
    messages = [
        {"role": "system", "content": "You are Prodify. Your job is to assist the user with its queries related to any product and provide suggestions about the product"},
        {"role": "user", "content": prompt}
    ]
    
    if conversation_memory:
        messages.extend(conversation_memory)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    return response.choices[0].message.content


st.title('Sentiment Analysis & Chatbot')

# Get user input
user_input = st.text_input('Enter a review:')

if user_input:
    # Preprocess the review
    review_sequence = tokenizer.texts_to_sequences([user_input])
    review_padded = pad_sequences(review_sequence, maxlen=max_sequence_length)

    # Perform sentiment analysis prediction
    prediction = model.predict(review_padded)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    st.write('Sentiment:', sentiment)
    st.write('Prediction Score:', prediction)

# Get user input for chatbot
# chat_input = st.text_input('Chat with the Chatbot:')

# if chat_input:
#     response = generate_response(chat_input)
#     st.write('Chatbot:', response)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'conversation_memory' not in st.session_state:
    st.session_state['conversation_memory'] = []

def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = generate_response(user_input, st.session_state['conversation_memory'])
    # Store the output 
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

    # Update conversation memory
    st.session_state['conversation_memory'].append({"role": "user", "content": user_input})
    st.session_state['conversation_memory'].append({"role": "assistant", "content": output})

if api:
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state['generated'][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

else:
    st.error("API Error")
