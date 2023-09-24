import os
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from pydub import AudioSegment
import sounddevice as sd
import csv
import time
import wave


def load_audio(filepaths):
    unique_extensions = set([os.path.splitext(path)[1] for path in filepaths])

    if len(unique_extensions) != 1:
        raise ValueError("Der Datensatz enthält unterschiedliche Dateiendungen.")
    
    file_extension = list(unique_extensions)[0]
    if file_extension == ".wav":
        return load_wav_16k_mono
    elif file_extension == ".mp3":
        return load_mp3_16k_mono
    else:
        raise ValueError(f"Dateiendung wird nicht unterstützt: {file_extension}")


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
    sample_wav = load_wav_16k_mono(sample_file_path)
    sample_spectrogram = preprocess_wav_for_model(sample_wav)
    return print(sample_spectrogram.shape)

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
    print("Aufnahme startet in 3 Sekunden!") #3 Sekunden warten, da nicht immer sofort alles passt.
    time.sleep(3)
    # 10 Sekunden Audioaufnahme
    print("Audio wird aufgenommen...")
    recording_duration = 10 # in sekunden
    samplerate = 16000 

    with sd.InputStream(device=3,samplerate=samplerate, channels=1, dtype='float32') as stream:  #je nach device von der query, muss device=3 angepasst werden
        audio_data, overflowed = stream.read(int(samplerate * recording_duration))
        
    with wave.open('aufnahme.wav', 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    # Preproccesing
    audio_data_reshaped = tf.reshape(audio_data, [-1])
    processed_data = preprocess_wav_for_model(audio_data_reshaped)
    # Klassifikation der Daten mit dem Modell
    prediction = model.predict(np.expand_dims(processed_data, axis=0))

    # Anzeige der Klassifikationsergebnisse
    print("Klassifikationsergebnis:", prediction)
    return print("Die vorhergesagte Klasse ist:", np.argmax(prediction))


def split_and_save_audio_chunks(audio_path, target, split_len=10):

    audio = AudioSegment.from_wav(audio_path)
    split_len_ms = split_len * 1000 
    num_chunks = len(audio) // split_len_ms

    base_dir = os.path.dirname(audio_path)
    split_dir = os.path.join(base_dir, target)
    if not os.path.exists(split_dir):
        os.mkdir(split_dir)

    base_filename = os.path.splitext(os.path.basename(audio_path))[0]

    for i in range(num_chunks):
        start_time = i * split_len_ms
        end_time = (i + 1) * split_len_ms
        chunk = audio[start_time:end_time]
        chunk.export(os.path.join(split_dir, f"{base_filename}_{i + 1}.wav"), format="wav")
    
    if len(audio) % split_len_ms:
        last_chunk = audio[num_chunks * split_len_ms:]
        last_chunk.export(os.path.join(split_dir, f"{base_filename}_{num_chunks + 1}.wav"), format="wav")

def predict_with_saved_model(model_path , wav_file_path):
    
    loaded_model = tf.keras.models.load_model(model_path)
    
    wav = load_wav_16k_mono(wav_file_path)
    processed_wav = preprocess_wav_for_model(wav)
    processed_wav_batched = tf.expand_dims(processed_wav, axis=0)
    predictions = loaded_model.predict(processed_wav_batched)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    return print(predicted_class)

def save_predictions_to_csv(model, data_path, output_filename, binary=True):
    results = {}
    for file in os.listdir(data_path):
        FILEPATH = os.path.join(data_path, file)
        wav = load_wav_16k_mono(FILEPATH)
        
        processed_wav = preprocess_wav_for_model(wav)
        processed_wav_batched = tf.expand_dims(processed_wav, axis=0)
        predictions = model.predict(processed_wav_batched)
        
        if binary:
            binary_prediction = 1 if predictions[0][0] > 0.5 else 0
            results[file] = binary_prediction
        else:
            predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
            print(predicted_class)
            results[file] = predicted_class

    
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dateiname', 'Vorhersage'])  
        for key, value in results.items():
            writer.writerow([key,value])