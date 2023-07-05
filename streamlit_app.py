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

openai.api_key= st.secrets["openai"]["api_key"]
# Load the service account credentials from the downloaded JSON file

# Load the service account credentials from the environment variable
credentials_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
credentials = service_account.Credentials.from_service_account_file(
    credentials_path, scopes=['https://www.googleapis.com/auth/drive'
)
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
# tokenizer.fit_on_texts(X_train_cleaned_series)
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

def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "If user asks about anything other than related to products on amazon than reply it with : I'm sorry, I can only provide information about Amazon products. Otherwise if user asks about any product then please give them information about product's price, specifications and reviews"},
            {"role": "user", "content": prompt}
        ]
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
chat_input = st.text_input('Chat with the Chatbot:')

if chat_input:
    response = generate_response(chat_input)
    st.write('Chatbot:', response)

# # Run the Streamlit app
# if __name__ == '__main__':
#     app()
