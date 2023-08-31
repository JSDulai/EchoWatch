
import os
import csv
import sys
sys.path.append("../EchoWatch")
import tensorflow as tf
from models.model import create_model, create_optimized_model, create_moodel, model_create, create_moodel_pt, modified_model_create, cnn_model_create, revised_cnn_model_create, revised_cnn_model_create1
from utils.utilities import load_mp3_16k_mono, preprocess_wav_for_model, load_wav_16k_mono
from utils.preparation import prepare_data, split_data, prepare_data1, get_label_from_filename1, split_data1


def main():
    #Datensatz wird geladen, vorbereitet und geteilt.
    data_path = os.path.join('data', 'pump', "test")
    #klassennamen = ["A_1000", "B_1000", "C_1000", "D_1000"]
    #get_label_from_filepath1(data_path)
    data = prepare_data(data_path)
    #data = prepare_data(data_path, klassennamen)
    train, test = split_data(data)


    #Model wird erstellt.
    model = modified_model_create()
    model.summary()
    hist = model.fit(train, epochs=30, validation_data=test)


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


#main()





def main1():
    #Datensatz wird geladen, vorbereitet und geteilt.
    data_path = os.path.join('data', 'PT500')
    data = prepare_data1(data_path)
    train, val, test = split_data1(data)



    #Model wird erstellt.
    model = revised_cnn_model_create()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


    model.summary()
    hist = model.fit(train, epochs=15, validation_data=val, callbacks=early_stopping)


    # Angenommen, Ihr Modell heißt "model" und Ihr Testdatensatz "test_data"
    loss, accuracy, recall, precision = model.evaluate(test)


    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    
    test_labels_list = []
    predicted_labels_list = []


    for test_features, test_labels in test:
        predicted_scores = model.predict(test_features)
        predicted_batch_labels = tf.argmax(predicted_scores, axis=1).numpy()


        test_labels_list.extend(tf.argmax(test_labels, axis=1).numpy())


        predicted_labels_list.extend(predicted_batch_labels)


    confusion_mat = tf.math.confusion_matrix(test_labels_list, predicted_labels_list)
    print(confusion_mat)


    #Ergebnisse werden in result gespeichert.
    results = {}
    for file in os.listdir(data_path):
        FILEPATH = os.path.join(data_path, file)
        wav = load_wav_16k_mono(FILEPATH)
        processed_wav = preprocess_wav_for_model(wav)
        processed_wav_batched = tf.expand_dims(processed_wav, axis=0)
        predictions = model.predict(processed_wav_batched)

        predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
        results[file] = predicted_class


    #Result wird in eine csv abgespeichert.
    with open('predictions.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dateiname', 'Zustand'])  # Überschrift
        for key, value in results.items():
            writer.writerow([key, ['Normal', 'Innenringschaden', 'Außenringschaden', 'Verschleiß'][value]])


    return 'Fertig! Ergebnisse in predictions.csv gespeichert.'


main1()