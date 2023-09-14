import os
import sys
sys.path.append("../EchoWatch")
from itertools import product
import tensorflow as tf
import numpy as np
from utils.utilities import load_audio, preprocess_wav_for_model, predict_from_wav, load_wav_16k_mono, live_audio_classification
from utils.preparation import prepare_data, split_data, prepare_data1
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold


def revised_cnn_model_create(input_shape=(1491, 257, 1), num_classes=4, 
                             conv2d_filters=(32, 64, 128), kernel_size=(3, 3), 
                             max_pooling_size=(2, 2), dense_units=32, 
                             l2_reg=0.01, learning_rate=0.001):
    
    model = Sequential([
        Conv2D(filters=conv2d_filters[0], kernel_size=kernel_size, activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=max_pooling_size),
        
        Conv2D(filters=conv2d_filters[1], kernel_size=kernel_size, activation='relu'),
        MaxPooling2D(pool_size=max_pooling_size),
        
        Conv2D(filters=conv2d_filters[2], kernel_size=kernel_size, activation='relu'),
        MaxPooling2D(pool_size=max_pooling_size),
        
        Flatten(),
        Dense(units=dense_units, activation='relu', kernel_regularizer=l2(l2_reg)),
        
        Dense(units=num_classes, activation='softmax')
    ])


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])


    return model


def grid_search_hyperparameter_optimization(data, labels, param_grid, n_splits=5):
    best_val_accuracy = 0
    best_params = None
    best_val_loss = float('inf')
    
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]

    # Initialisierung des K-Fold-Cross-Validators
    kf = KFold(n_splits=n_splits)

    results = []

    for params in param_combinations:
        val_accuracies = []
        val_losses = []

        for train_index, val_index in kf.split(data):
            train_data, val_data = data[train_index], data[val_index]
            train_labels, val_labels = labels[train_index], labels[val_index]

            # Erstellen und Trainieren des Modells mit der aktuellen Kombination von Hyperparametern
            model = revised_cnn_model_create(**params)
            history = model.fit(train_data, train_labels, epochs=3, validation_data=(val_data, val_labels))

            # Bewertung des Modells auf den Validierungsdaten
            val_loss, val_accuracy = model.evaluate(val_data, val_labels)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

        # Berechnung der durchschnittlichen Validierungsgenauigkeit und -verlust über alle Folds
        mean_val_accuracy = np.mean(val_accuracies)
        mean_val_loss = np.mean(val_losses)

        results.append({'params': params, 'accuracy': mean_val_accuracy, 'loss': mean_val_loss})

    # Sortieren der Ergebnisse nach Genauigkeit und Verlust
    results = sorted(results, key=lambda x: (x['accuracy'], -x['loss']), reverse=True)

    # Ausgabe der besten 3 Parameterkombinationen
    best_params = results[:3]

    for i, res in enumerate(best_params):
        print(f"Beste Parameter {i+1}: {res['params']}, Genauigkeit: {res['accuracy']}, Verlust: {res['loss']}")

    return best_params


param_grid = {
    'conv2d_filters': [(16, 32, 64), (32, 64, 128), (64, 128, 256)],
    'kernel_size': [(3, 3), (5, 5)],
    'max_pooling_size': [(3, 3), (4, 4)],
    'dense_units': [16, 32, 64],
    'l2_reg': [0.001, 0.1],
    'learning_rate': [0.001, 0.0001, 0.00001]
}

def get_input_shape_from_data(dataset):
    # Nehmen Sie ein Beispiel aus dem Datensatz
    sample_data = next(iter(dataset))
    
    # Überprüfen Sie, ob das Beispiel ein TensorFlow Tensor ist und konvertieren Sie es ggf. in ein numpy Array
    if tf.is_tensor(sample_data):
        sample_data = sample_data.numpy()
    
    return sample_data.shape

data_path = os.path.join('data', 'PT500')
class_names=['A', 'B', 'C', 'D']
num_classes=4

data_np, labels_np = prepare_data1(data_path, class_names, num_classes)

#train, val, test = split_data(data)
#shapee = get_input_shape_from_data(data_np)
#print(shapee)

best_params = grid_search_hyperparameter_optimization(data_np, labels_np, param_grid, n_splits=5)
print("Beste Hyperparameter: ", best_params)

