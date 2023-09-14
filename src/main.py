
import os
import csv
import sys
sys.path.append("../EchoWatch")
import tensorflow as tf
from models.model import create_model,revised_cnn_model_create, piczak_modell
from utils.utilities import load_audio, preprocess_wav_for_model, get_input_shape_from_data, predict_from_wav, load_wav_16k_mono, live_audio_classification
from utils.preparation import prepare_data, split_data, prepare_data1

def main_binary_classification(data_path, class_names, num_classes, output_filename='predictions_binary.csv'):
    data = prepare_data(data_path, class_names, num_classes)
    train, test = split_data(data)
    
    
    model = create_binary_model()
    model.summary()
    hist = model.fit(train, epochs=3, validation_data=test)  
    
    
    save_predictions_to_csv(model, data_path, output_filename, binary=True)

def main_multiclass_classification(data_path, class_names, num_classes, output_filename='predictions_multiclass.csv'):
    
    #predict_with_saved_model(model_path = "../EchoWatch/models/pt500_model.h5", wav_file_path="../EchoWatch/data/PT500/C_1000_23.wav")
    #loaded_model = tf.keras.models.load_model("../EchoWatch/models/pt500_model.h5")
    #live_audio_classification(loaded_model)

    data = prepare_data(data_path, class_names, num_classes)
    train, val, test = split_data(data)
    
    
    model = piczak_modell() 
    model.summary()
    hist = model.fit(train, epochs=20, validation_data=val)  # Reduced epochs for testing
    
    loss, accuracy, recall, precision = model.evaluate(test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    loaded_model = tf.keras.models.load_model("../EchoWatch/models/pt500_model.h5")
    
    save_predictions_to_csv(model, data_path, output_filename, binary=False)

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



def predict_with_saved_model(model_path , wav_file_path):
    
    loaded_model = tf.keras.models.load_model(model_path)
    
    wav = load_wav_16k_mono(wav_file_path)
    processed_wav = preprocess_wav_for_model(wav)
    processed_wav_batched = tf.expand_dims(processed_wav, axis=0)
    
    predictions = loaded_model.predict(processed_wav_batched)
    
    # Return the class with the highest probability
    #predicted_class = tf.argmax(predictions[0]).numpy()

    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    return print(predicted_class)


data_path = os.path.join('data', 'PT500')
class_names=['A', 'B', 'C', 'D']
num_classes=4
main_multiclass_classification(data_path, class_names, num_classes)