import os
import sys
sys.path.append("../EchoWatch")
import tensorflow as tf
from models.model import model_dropout, model_optimized, model_piczak, simple_model
from utils.utilities import save_predictions_to_csv, plot_confusion_matrix, preprocess_wav_for_model, load_wav_16k_mono, live_audio_classification, split_and_save_audio_chunks, predict_with_saved_model
from utils.preparation import prepare_data, split_data

# Funktion, um Konfusionsmatrizen für alle Modelle zu erstellen.
def train_and_plot_all_models(data_path, class_names, num_classes, model_functions, acti_func='softmax', loss_func='CategoricalCrossentropy'):
    data = prepare_data(data_path, class_names, num_classes)
    train, val, test = split_data(data)
    
    for model_function in model_functions:
        print(f"Folgendes Modell wird traniert: {model_function.__name__}")
        
        model = model_function(num_classes, acti_func, loss_func)
        model.summary()
        
        callback_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        callback_acc = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        
        hist = model.fit(train, epochs=100, validation_data=val, callbacks=[callback_loss, callback_acc])
        
        plot_confusion_matrix(model, test, class_names, save_path=f"./models/matrices/{model_function.__name__}_matrix.png")
        print(f"Modell {model_function.__name__} traniert und Konfusionsmatrix abgespeichert.")

# Funktion für binäre Klassifikation von Maschinengeräuschen
def main_binary_classification(data_path, class_names, num_classes, acti_func ='sigmoid', loss_func = 'BinaryCrossentropy', output_filename='predictions_binary.csv'):
    
    # Daten werden geladen und unterteilt
    data = prepare_data(data_path, class_names, num_classes)
    train, val, test = split_data(data)
    
    # Modell wird geladen
    model = model_optimized(num_classes, acti_func, loss_func) 
    model.summary()
    
    # EarlyStopping-Callback, der das Training stoppt, wenn 'val_loss' und 'val_accuracy' sich für 10 Epochen nicht verbessert und die besten Gewichte wiederherstellt.
    callback_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callback_acc = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    # Training des Modells mit den Trainingsdaten
    hist = model.fit(train, epochs=100, validation_data=val, callbacks=[callback_loss, callback_acc])
    
    # Evaluieren der Modellperformance mit den Testdaten
    loss, accuracy, recall, precision = model.evaluate(test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    
    # Konfusionsmatrix wird erstellt und mit ausgewähltem Namen abgespeichert.
    plot_confusion_matrix(model, test, class_names, save_path='./models/matrices/model_optimized_binary.png')

    # Modell speichern
    model.save('../EchoWatch/models/cnns/binary_pump_model.h5')

    with open('pump_model_metrics.txt', 'w') as f:
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test Accuracy: {accuracy}\n")

    save_predictions_to_csv(model, data_path, output_filename, binary=True)

# Funktion für multiklassen Klassifikation von Maschinengeräuschen
def main_multiclass_classification(data_path, class_names, num_classes, acti_func ='softmax', loss_func = 'CategoricalCrossentropy', output_filename='predictions_multiclass.csv'):
    
    data = prepare_data(data_path, class_names, num_classes)
    train, val, test = split_data(data)

    model = model_optimized(num_classes, acti_func, loss_func) 
    model.summary()
   
    callback_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callback_acc = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    hist = model.fit(train, epochs=100, validation_data=val, callbacks=[callback_loss, callback_acc])
    
    loss, accuracy, recall, precision = model.evaluate(test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    
    plot_confusion_matrix(model, test, class_names, save_path='./models/matrices/model_optimized.png')

    model.save('./models/cnns/multiclass_pt500_model.h5')

    with open('pt500_model_metrics.txt', 'w') as f:
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test Accuracy: {accuracy}\n")
    
    save_predictions_to_csv(model, data_path, output_filename, binary=False)


# Hier wird das Modell geladen und eine Vorhersage wird getroffen!
# Für beide kommt ein Spektrogramm samt Klassifikation raus
loaded_model = tf.keras.models.load_model("./models/cnns/multiclass_pt500_model.h5")
predict_with_saved_model(loaded_model, ["./data/PT500/F_500_96.wav"])
live_audio_classification(loaded_model)




# Alles nachfolgend ist eher für das Training!
data_path = os.path.join('data', 'PT500')
class_names=['A', 'B', 'C', 'D', 'E', 'F']
num_classes=len(class_names)
#all_models = [simple_model, model_optimized]
#train_and_plot_all_models(data_path, class_names, num_classes, all_models)

main_multiclass_classification(data_path, class_names, num_classes)

data_path1 = os.path.join('data', 'pump', 'test')
class_names1 = ['normal', 'anomaly']
num_classes1 = len(class_names1)
main_binary_classification(data_path1, class_names1, num_classes1)
