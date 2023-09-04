import os
import sounddevice as sd
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import time
import wave
import matplotlib as plt
from itertools import groupby

def load_audio(filename, target_sample_rate=16000):
    # Detect file extension
    _, file_extension = os.path.splitext(filename)
    
    if file_extension == ".wav":
        return load_wav_16k_mono(filename)
    elif file_extension == ".mp3":
        return load_mp3_16k_mono(filename)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

#MP3 wird geladen in ein Float-Tensor konvertiert und auf 16Khz gesampelt. Mono
def load_mp3_16k_mono(filename):
    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess_wav_for_model(wav):
    wav = wav[:48000] #Alles ab 4800 wird abgeschnitten
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32) #Wenns zu kurz ist, wird mit zeros gefüllt.
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)#Umwandlung zum Spektrogramm
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram


def get_input_shape_from_data(dataset_path):
    sample_file_path = os.path.join(dataset_path, os.listdir(dataset_path)[0])
    sample_wav = load_audio(sample_file_path)
    sample_spectrogram = preprocess_wav_for_model(sample_wav)
    return sample_spectrogram.shape

def predict_from_wav(model, wav):
    processed_data = preprocess_wav_for_model(wav)
    prediction = model.predict(processed_data)
    return prediction

def predict_with_saved_model(model_path, wav_file_path):
    loaded_model = tf.keras.models.load_model(model_path)
    wav = load_audio(wav_file_path)
    processed_wav = preprocess_wav_for_model(wav)
    processed_wav_batched = tf.expand_dims(processed_wav, axis=0)
    predictions = loaded_model.predict(processed_wav_batched)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    return predicted_class

def live_audio_classification(model):
    
    print(sd.query_devices())

    
    # Wartezeit von 3 Sekunden vor der Aufnahme
    print("Warten Sie 3 Sekunden vor der Aufnahme...")
    time.sleep(3)

    # 10 Sekunden Audioaufnahme
    print("Nehmen Sie 10 Sekunden Audio auf...")
    recording_duration = 10  # in Sekunden
    samplerate = 16000  # Samplerate von 16 kHz, kann nach Bedarf angepasst werden

    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32') as stream:
        audio_data, overflowed = stream.read(int(samplerate * recording_duration))
        
    # Speichern Sie die aufgenommene Audio-Daten als WAV-Datei, um sie zu überprüfen
    with wave.open('aufnahme.wav', 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())


    # Vorverarbeitung der Audio-Daten
    print("Verarbeiten der Audio-Daten...")
    audio_data_reshaped = tf.reshape(audio_data, [-1])
    processed_data = preprocess_wav_for_model(audio_data_reshaped)


    # Klassifikation der Daten mit dem Modell
    print("Klassifizierung der Audio-Daten...")
    prediction = model.predict(np.expand_dims(processed_data, axis=0))

    # Anzeige der Klassifikationsergebnisse
    print("Klassifikationsergebnis:", prediction)
    return print("Die vorhergesagte Klasse ist:", np.argmax(prediction))