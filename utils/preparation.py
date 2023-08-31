import os
import tensorflow as tf
import numpy as np
from utils.utilities import load_wav_16k_mono, preprocess_wav_for_model

def prepare_data(dataset_path): #class_names als Parameter hinzufügen
    file_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path)]
    labels = [1 if 'normal' in filename else 0 for filename in os.listdir(dataset_path)]
    #labels = [get_label_from_filename(filename, class_names) for filename in os.listdir(dataset_path)]
    data = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    data = data.map(lambda filepath, label: (preprocess_wav_for_model(load_wav_16k_mono(filepath)), label))
    data = data.cache().shuffle(buffer_size=1000).batch(16).prefetch(8)
    return data

def split_data(data):
    train_size = int(len(data)*0.7)
    test_size = len(data) - train_size
    train = data.take(train_size)
    test = data.skip(train_size).take(test_size)
    return train, test

#Für die Gui später, um pro Klasse ein label hinzuzufügen.
def get_label_from_filename(filename, class_names):
    for index, class_name in enumerate(class_names, start=1):
        if class_name.lower() in filename.lower():
            return index
    return 0  #Sollte nicht passieren, wenn jeder Dateiname genau eine Klasse enthält



def prepare_data1(dataset_path): #class_names als Parameter hinzufügen
    file_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path)]
    labels = [tf.keras.utils.to_categorical(get_label_from_filename1(filename, ['A', 'B', 'C', 'D']), num_classes=4) for filename in os.listdir(dataset_path)]
    #labels = [get_label_from_filename(filename, class_names) for filename in os.listdir(dataset_path)]
    data = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    data = data.map(lambda filepath, label: (preprocess_wav_for_model(load_wav_16k_mono(filepath)), label))
    data = data.cache().shuffle(buffer_size=1000).batch(12).prefetch(6)
    return data

def split_data1(data):
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)
    # Der Rest geht in den Testsatz
    test_size = len(data) - train_size - val_size
    
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    
    return train, val, test


#Für die Gui später, um pro Klasse ein label hinzuzufügen.
def get_label_from_filename1(filename, class_names):
    for index, class_name in enumerate(class_names):
        if filename.lower().startswith(class_name.lower()):
            return index
    raise ValueError(f'Unmatched filename: {filename}')

