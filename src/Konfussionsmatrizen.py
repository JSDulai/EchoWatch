import os
import sys
sys.path.append("../EchoWatch")
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.model import simple_model, model_optimized, model_dropout, modell_piczak
from utils.preparation import prepare_data, split_data

def generate_confusion_matrices(data_path, class_names, num_classes, save_path):
    # Daten vorbereiten
    data = prepare_data(data_path, class_names, num_classes)
    train, val, test = split_data(data)
    
    # Liste der Modelle
    models = [model_optimized, modell_piczak]
    model_names = ["model_optimized_1","modell_piczak_1"]
    
    confusion_matrices = {}
    
    for model_func, model_name in zip(models, model_names):
        # Modell initialisieren und trainieren
        model = model_func(num_classes, 'softmax', 'CategoricalCrossentropy')
        model.fit(train, epochs=20, validation_data=val)  # Sie können die Anzahl der Epochen anpassen
        
        # Vorhersagen für den Testdatensatz machen
        y_pred = model.predict(test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Tatsächliche Labels extrahieren
        y_true = np.concatenate([y for x, y in test], axis=0)
        y_true_classes = np.argmax(y_true, axis=1)
        
        # Konfusionsmatrix erstellen
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        confusion_matrices[model_name] = cm
        
        # Visualisierung der Konfusionsmatrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Vorhergesagte Labels')
        plt.ylabel('Tatsächliche Labels')
        plt.title(f'Konfusionsmatrix für {model_name}')
        
        # Speichern der Konfusionsmatrix als .png-Datei
        plt.savefig(f"{save_path}/{model_name}.png")
        plt.close()
        
    return confusion_matrices




# Aufruf der Funktion
data_path = os.path.join('data', 'PT500')
class_names=['A', 'B', 'C', 'D', 'E', 'F']
num_classes=len(class_names)
save_path = os.path.join('models', 'Matrizen')  # Bitte ersetzen Sie durch den tatsächlichen Pfad, wo Sie die Dateien speichern möchten
confusion_matrices = generate_confusion_matrices(data_path, class_names, num_classes, save_path)
