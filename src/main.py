import os
import sys
sys.path.append("../EchoWatch")
import tensorflow as tf
from models.model import model_dropout, model_optimized, modell_piczak, simple_model
from utils.utilities import save_predictions_to_csv, preprocess_wav_for_model, load_wav_16k_mono, live_audio_classification, split_and_save_audio_chunks, predict_with_saved_model
from utils.preparation import prepare_data, split_data, prepare_data_for_cross_validation

# Funktion für binäre Klassifikation von Maschinengeräuschen
def main_binary_classification(data_path, class_names, num_classes, acti_func ='sigmoid', loss_func = 'BinaryCrossentropy', output_filename='predictions_binary.csv'):
    
    _ = tf.keras.utils.get_file('pump.zip',
                        'https://zenodo.org/record/3678171/files/dev_data_pump.zip?download=1',
                        cache_dir='./',
                        cache_subdir='data',
                        extract=True)
    
    
    #data = prepare_data(data_path, class_names, num_classes)
    #train, val, test = split_data(data)
    
    
    #model = model_optimized(num_classes, acti_func, loss_func) 
    #model.summary()
    
    for train_dataset, val_dataset, test_dataset in prepare_data_for_cross_validation(data_path, class_names, num_classes):
        model = model_optimized(num_classes, acti_func, loss_func) 
        model.summary()
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=1,
                                                restore_best_weights=True)
        # Training des Modells mit den Trainingsdaten
        hist = model.fit(train_dataset, epochs=3, validation_data=val_dataset, callbacks=callback)

        # Evaluieren der Modellperformance mit den Testdaten
        loss, accuracy, recall, precision = model.evaluate(test_dataset)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")
    
    with open('pump_model_metrics.txt', 'w') as f:
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test Accuracy: {accuracy}\n")
    model.save('./models/binary_pump_model.h5')
    save_predictions_to_csv(model, data_path, output_filename, binary=True)

# Funktion für multiklassen Klassifikation von Maschinengeräuschen
def main_multiclass_classification(data_path, class_names, num_classes, acti_func ='softmax', loss_func = 'CategoricalCrossentropy', output_filename='predictions_multiclass.csv'):
    
    #split_and_save_audio_chunks("../EchoWatch/data/F_1500.wav", "PT500")
    
    
    #predict_with_saved_model(model_path = "../EchoWatch/models/pt500_model.h5", wav_file_path="../EchoWatch/data/PT500/C_1000_23.wav")
    #loaded_model = tf.keras.models.load_model("../EchoWatch/models/pt500_model.h5")
    #live_audio_classification(loaded_model)
    
    for train_dataset, val_dataset, test_dataset in prepare_data_for_cross_validation(data_path, class_names, num_classes):
        model = model_optimized(num_classes, acti_func, loss_func) 
        model.summary()
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=1,
                                                restore_best_weights=True)
        # Training des Modells mit den Trainingsdaten
        hist = model.fit(train_dataset, epochs=3, validation_data=val_dataset, callbacks=callback)

        # Evaluieren der Modellperformance mit den Testdaten
        loss, accuracy, recall, precision = model.evaluate(test_dataset)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")
    
    with open('pt500_model_metrics.txt', 'w') as f:
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test Accuracy: {accuracy}\n")

    # Erstellen und Initialisieren des Klassifikationsmodells
    #loaded_model = tf.keras.models.load_model("../EchoWatch/models/pt500_model.h5")
    model.save('./models/multiclass_pt500_model.h5')

    save_predictions_to_csv(model, data_path, output_filename, binary=False)



data_path = os.path.join('data', 'pump', "test")
class_names=['normal', 'anomaly']
num_classes=2
main_binary_classification(data_path, class_names, num_classes)

data_path1 = os.path.join('data', 'PT500')
class_names1=['A', 'B', 'C', 'D', 'E', 'F']
num_classes1=6
main_multiclass_classification(data_path1, class_names1, num_classes1)


