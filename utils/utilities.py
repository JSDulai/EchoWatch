import os
import tensorflow as tf
import tensorflow_io as tfio
from itertools import groupby

#WAV wird geladen in ein Float-Tensor konvertiert und auf 16Khz gesampelt. Mono
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

#Vorbereitung fürs Model
def preprocess_wav_for_model(wav):
    wav = wav[:48000] #Alles ab 4800 wird abgeschnitten
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32) #Wenns zu kurz ist, wird mit zeros gefüllt.
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)#Umwandlung zum Spektrogramm
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

def get_input_shape_from_data(dataset_path):
    # Choose a sample file from the dataset
    sample_file_path = os.path.join(dataset_path, os.listdir(dataset_path)[0])
    
    # Load and preprocess the sample file
    sample_wav = load_wav_16k_mono(sample_file_path)
    sample_spectrogram = preprocess_wav_for_model(sample_wav)
    
    return sample_spectrogram.shape

def predict_from_wav(model_path, wav):
    # Modell laden
    model = tf.keras.models.load_model(model_path)
    
    # .wav Datei laden und verarbeiten (dies hängt von Ihrer spezifischen Vorverarbeitung ab)
    processed_data = preprocess_wav_for_model(wav)
    
    # Vorhersage treffen
    prediction = model.predict(processed_data)
    
    return prediction


def predict_with_saved_model(model_path , wav_file_path):
    # Load the saved model
    loaded_model = tf.keras.models.load_model(model_path)
    
    # Process the WAV file
    wav = load_wav_16k_mono(wav_file_path)
    processed_wav = preprocess_wav_for_model(wav)
    processed_wav_batched = tf.expand_dims(processed_wav, axis=0)
    
    # Make predictions
    predictions = loaded_model.predict(processed_wav_batched)
    
    # Return the class with the highest probability
    #predicted_class = tf.argmax(predictions[0]).numpy()

    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    return print(predicted_class)

#def postprocess_results(results):
    postprocessed = {}
    for file, logits in results.items():
        scores = [1 if prediction > 0.1 else 0 for prediction in logits]
        postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()
    return postprocessed
