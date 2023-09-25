import os
import sys
sys.path.append("../EchoWatch")
from itertools import product
import tensorflow as tf
from utils.utilities import load_audio, preprocess_wav_for_model, get_input_shape_from_data, predict_from_wav, load_wav_16k_mono, live_audio_classification
from utils.preparation import prepare_data, split_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2

def revised_cnn_model_create(input_shape=(1491, 257, 1), num_classes=4, 
                             conv2d_filters=(32, 64, 128), kernel_size=(3, 3), 
                             dense_units=32, l2_reg=0.01, 
                             learning_rate=0.001, pool_size=(2, 2)):
    """
    Erstellt ein CNN-Modell mit anpassbaren Hyperparametern.

    Args:
    input_shape (tuple): Die Form des Eingabe-Tensors.
    num_classes (int): Die Anzahl der Ausgangsklassen.
    conv2d_filters (tuple): Die Anzahl der Filter für jede Conv2D-Schicht.
    kernel_size (tuple): Die Größe des Kernels für die Conv2D-Schichten.
    dense_units (int): Die Anzahl der Einheiten in der Dense-Schicht.
    l2_reg (float): Der Regularisierungsfaktor für die L2-Regularisierung.
    learning_rate (float): Die Lernrate für den Adam-Optimierer.
    pool_size (tuple): Die Größe der Pooling-Fenster für die MaxPooling2D-Schichten.

    Returns:
    model: Das erstellte Modell.
    """
    model = Sequential([
        Conv2D(filters=conv2d_filters[0], kernel_size=kernel_size, activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=pool_size),
        
        Conv2D(filters=conv2d_filters[1], kernel_size=kernel_size, activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        
        Conv2D(filters=conv2d_filters[2], kernel_size=kernel_size, activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        
        Flatten(),
        Dense(units=dense_units, activation='relu', kernel_regularizer=l2(l2_reg)),
        
        Dense(units=num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model


def grid_search_hyperparameter_optimization(train_data, val_data, param_grid):
    best_val_accuracy = 0
    best_params = None

    # Erstellen Sie eine Liste aller möglichen Kombinationen von Hyperparametern
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]

    for params in param_combinations:
        # Erstellen Sie ein Modell mit der aktuellen Kombination von Hyperparametern
        model = revised_cnn_model_create(**params)
        
        # Trainieren Sie das Modell auf den Trainingsdaten
        model.fit(train_data, epochs=5, validation_data=val_data)
        
        # Bewerten Sie das Modell auf den Validierungsdaten
        val_loss, val_accuracy = model.evaluate(val_data)
        
        # Wenn das Modell eine bessere Genauigkeit auf den Validierungsdaten erzielt hat, speichern Sie die Hyperparameter
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = params

    return best_params

# Definition der Hyperparameter-Bereiche für die Grid Search
param_grid = {
    'conv2d_filters': [(16, 32, 64), (32, 64, 128), (64, 128, 256)],
    'kernel_size': [(2, 2), (3, 3), (4, 4)],
    'dense_units': [16, 32, 64, 128],
    'l2_reg': [0.01, 0.1, 0.5, 0.9],
    'pool_size': [(2, 2), (3, 3), (4,4)],
    'learning_rate': [0.001, 0.0001, 0.00001]
}


# Nachdem Sie die Daten vorbereitet haben, können Sie die Grid Search wie folgt aufrufen:

data_path = os.path.join('data', 'PT500')
class_names=['A', 'B', 'C', 'D']
num_classes=4

data = prepare_data(data_path, class_names, num_classes)

train, val, test = split_data(data)

best_params = grid_search_hyperparameter_optimization(train, val, param_grid)
print("Beste Hyperparameter: ", best_params)
