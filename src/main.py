import os
import csv
import sys
sys.path.append("../EchoWatch")
import tensorflow as tf
from models.model import create_model
from utils.utilities import load_mp3_16k_mono, preprocess_wav_for_model
from utils.preparation import prepare_data, split_data

def main():
    #Datensatz wird geladen, vorbereitet und geteilt.
    data_path = os.path.join('data', 'pump', 'test')
    data = prepare_data(data_path)
    train, test = split_data(data)

    #Model wird erstellt.
    model = create_model()
    hist = model.fit(train, epochs=4, validation_data=test)

    #Ergebnisse werden in result gespeichert.
    results = {}
    for file in os.listdir(data_path):
        FILEPATH = os.path.join(data_path, file)
        wav = load_mp3_16k_mono(FILEPATH)
        processed_wav = preprocess_wav_for_model(wav)
        processed_wav_batched = tf.expand_dims(processed_wav, axis=0)
        predictions = model.predict(processed_wav_batched)
        
        #Konvertieren der Vorhersagen in 0 oder 1 basierend auf einem Schwellenwert von 0.5
        binary_prediction = 1 if predictions[0][0] > 0.5 else 0
        results[file] = binary_prediction

    #Result wird in eine csv abgespeichert.
    with open('predictions.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dateiname', 'Vorhersage'])  # Überschrift
        for key, value in results.items():
            writer.writerow([key, value])

    return 'Fertig! Ergebnisse in predictions.csv gespeichert.'

main()