import os
import sys
sys.path.append("../EchoWatch")
import tensorflow as tf
from models.model import model_dropout, model_optimized, modell_piczak, simple_model
from utils.utilities import save_predictions_to_csv, preprocess_wav_for_model, load_wav_16k_mono, live_audio_classification, split_and_save_audio_chunks, predict_with_saved_model
from utils.preparation import prepare_data, split_data


def main_binary_classification(data_path, class_names, num_classes, acti_func ='sigmoid', loss_func = 'BinaryCrossentropy', output_filename='predictions_binary.csv'):
    data = prepare_data(data_path, class_names, num_classes)
    train, test = split_data(data)
    
    
    model = simple_model(num_classes, acti_func, loss_func)
    model.summary()
    hist = model.fit(train, epochs=3, validation_data=test)  
    
    
    save_predictions_to_csv(model, data_path, output_filename, binary=True)

def main_multiclass_classification(data_path, class_names, num_classes, acti_func ='softmax', loss_func = 'CategoricalCrossentropy', output_filename='predictions_multiclass.csv'):
    
    #split_and_save_audio_chunks("../EchoWatch/data/F_1500.wav", "PT500")
    
    
    #predict_with_saved_model(model_path = "../EchoWatch/models/pt500_model.h5", wav_file_path="../EchoWatch/data/PT500/C_1000_23.wav")
    #loaded_model = tf.keras.models.load_model("../EchoWatch/models/pt500_model.h5")
    #live_audio_classification(loaded_model)
    
    data = prepare_data(data_path, class_names, num_classes)
    train, val, test = split_data(data)
    
    
    model = model_optimized(num_classes, acti_func, loss_func) 
    model.summary()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)
    
    hist = model.fit(train, epochs=100, validation_data=val)  # Reduced epochs for testing
    
    loss, accuracy, recall, precision = model.evaluate(test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    loaded_model = tf.keras.models.load_model("../EchoWatch/models/pt500_model.h5")
    
    save_predictions_to_csv(model, data_path, output_filename, binary=False)



data_path = os.path.join('data', 'PT500')
class_names=['A', 'B', 'C', 'D', 'E', 'F']
num_classes=6
main_multiclass_classification(data_path, class_names, num_classes)