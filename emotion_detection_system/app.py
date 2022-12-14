import streamlit as st
from tensorflow import keras
import numpy as np
import librosa
from sklearn.preprocessing import OneHotEncoder #scikit-learn**

model = keras.models.load_model('speech_emo_recognition.h5')

def predict_emotion(audio):
    # Extract features from audio
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)   # MFCC Feature extraction
    result = mfcc

    x = np.array(result)

    # Frame data to make it compatible for the model
    x = np.expand_dims(x, axis=3)
    x = np.swapaxes(x, 1, 2)
    x = np.expand_dims(x, axis=3)

    return x

st.title("Emotion Detection System")

saved_audio = st.file_uploader('Upload an audio file')

st.audio(saved_audio)

X = predict_emotion(saved_audio)

y = model.predict(X)

emotion = encoder.inverse_transform(y)

st.text("Emotion: ", emotion)