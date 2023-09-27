import os
import sys
sys.path.append("../EchoWatch")
from itertools import product
import tensorflow as tf
import numpy as np
from utils.preparation import prepare_data_optimization
from sklearn.model_selection import KFold

# Erstellung eines Modells mit gegebenen Hyperparametern
def create_model(input_shape=(1491, 257, 1), num_classes=2, 
                             conv2d_filters=(32, 64, 128), kernel_size=(3, 3), 
                             max_pooling_size=(2, 2), dense_units=32, 
                             l2_reg=0.01, learning_rate=0.001):
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=conv2d_filters[0], kernel_size=kernel_size, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=max_pooling_size),
        
        tf.keras.layers.Conv2D(filters=conv2d_filters[1], kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=max_pooling_size),
        
        tf.keras.layers.Conv2D(filters=conv2d_filters[2], kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=max_pooling_size),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        
        tf.keras.layers.Dense(units=num_classes, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='BinaryCrossentropy', 
                  metrics=['accuracy'])

    return model

# Funktion zur Durchführung einer Grid-Search zur Hyperparameter-Optimierung
def grid_search_hyperparameter_optimization(data, labels, param_grid, n_splits=5):

    best_params = None
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]
    kf = KFold(n_splits=n_splits)

    results = []

    for params in param_combinations:
        val_accuracies = []
        val_losses = []

        for train_index, val_index in kf.split(data):
            train_data, val_data = data[train_index], data[val_index]
            train_labels, val_labels = labels[train_index], labels[val_index]
            model = create_model(**params)
            hist = model.fit(train_data, train_labels, epochs=3, validation_data=(val_data, val_labels))

            val_loss, val_accuracy = model.evaluate(val_data, val_labels)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

        mean_val_accuracy = np.mean(val_accuracies)
        mean_val_loss = np.mean(val_losses)

        results.append({'params': params, 'accuracy': mean_val_accuracy, 'loss': mean_val_loss})

    results = sorted(results, key=lambda x: (x['accuracy'], -x['loss']), reverse=True)
    best_params = results[:3]

    for i, res in enumerate(best_params):
        print(f"Beste Parameter {i+1}: {res['params']}, Genauigkeit: {res['accuracy']}, Verlust: {res['loss']}")

    return best_params

# 'Grid' der Hyperparamter für Suche
param_grid = {
    'conv2d_filters': [(16, 32, 64), (32, 64, 128)],
    'kernel_size': [(3, 3), (5, 5)],
    'max_pooling_size': [(3, 3), (4, 4)],
    'dense_units': [16, 32, 64],
    'l2_reg': [0.001, 0.1],
    'learning_rate': [0.001, 0.0001]
    }

data_path = os.path.join('data', 'pump', 'test')
class_names=['normal', 'anomaly']
num_classes=2
data_np, labels_np = prepare_data_optimization(data_path, class_names, num_classes)

# Führt die Grid-Search zur Hyperparameter-Optimierung durch und gibt die besten Parameter aus
best_params = grid_search_hyperparameter_optimization(data_np, labels_np, param_grid, n_splits=5)
print("Beste Hyperparameter: ", best_params)

