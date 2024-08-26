import gradio as gr
from tensorflow.keras.utils import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model
import pickle


max_length = 80
trunc_type='post'
padding_type='post'


# model = tf.keras.models.load_model('models/saved_model')
model = load_model("models/model.keras")

with open('tokenizer/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def predict_sentiment(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    prediction = model.predict(padded)
    sentiment = np.argmax(prediction, axis=1)[0]
    labels = ['Irrelevant', 'Negative', 'Neutral', 'Positive']
    return labels[sentiment]


demo = gr.Interface(
    title="Sentiment Analysis of X (a.k.a. Twitter)",
    description="Predict sentiments of tweets!",
    fn=predict_sentiment,
    inputs=["text"],
    outputs=["text"],
)

demo.launch()

