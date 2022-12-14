import streamlit as st
from tensorflow import keras
import numpy as np
import librosa
from sklearn.preprocessing import OneHotEncoder

model = keras.models.load_model('speech_emo_recognition.h5')

emotions = ['angry', 'calm', 'disgust', 'fear', 'happy',
            'neutral', 'sad', 'surprise']  # Emotions as ordered by the training one hot encoder

# One hot encode the emotions
encoder = OneHotEncoder()
encoder.fit_transform(np.array(emotions).reshape(-1, 1)).toarray()


def extract_features(audio):
    # Extract features from audio
    x_value = []
    data, sample_rate = librosa.load(audio, duration=2.5, offset=0.6)
    # MFCC Feature extraction
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)
    result = mfcc

    x = np.array(result)

    # Frame data to make it compatible for the model
    x = np.expand_dims(x, axis=2)
    x = np.swapaxes(x, 0, 1)
    x = np.expand_dims(x, axis=2)

    x_value.append(x)
    x_value = np.array(x_value)

    return x_value


# Driver Code
st.title("Emotion Detection System")

saved_audio = st.file_uploader('Upload an audio file', type=['wav'])

if saved_audio is not None:

    st.audio(saved_audio)

    X = extract_features(saved_audio)

    y = model.predict(X)

    emotion = encoder.inverse_transform(y)[0][0]

    if emotion in ['angry', 'disgust', 'fear',
                   'sad']:
        st.write("Emotion: ", emotion)
        st.write("Be alert! Something's wrong.")
    else:
        st.write("Emotion: ", emotion)
        st.write("Looks like all's good!")
