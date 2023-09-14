import os
import sys
sys.path.append("../EchoWatch")
import tensorflow as tf
import numpy as np
from utils.utilities import load_audio, preprocess_wav_for_model, load_wav_16k_mono

def prepare_data(dataset_path, class_names, num_classes, batch_size=8, buffer_size=1000):
    file_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path)]
    labels = [tf.keras.utils.to_categorical(get_label_from_filename(filename, class_names), num_classes=num_classes) for filename in os.listdir(dataset_path)]
    data = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    data = data.map(lambda filepath, label: (preprocess_wav_for_model(load_wav_16k_mono(filepath)), label))
    data = data.cache().shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(buffer_size//batch_size)
    return data

def prepare_data1(dataset_path, class_names, num_classes, batch_size=12, buffer_size=1000):
    file_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path)]
    labels = [tf.keras.utils.to_categorical(get_label_from_filename(filename, class_names), num_classes=num_classes) for filename in os.listdir(dataset_path)]
    data = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    data = data.map(lambda filepath, label: (preprocess_wav_for_model(load_wav_16k_mono(filepath)), label))
    data = data.cache().shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(buffer_size//batch_size)
    
    # Konvertieren Sie das Dataset in eine Liste von Arrays
    data_list = []
    labels_list = []
    for batch_data, batch_labels in data.as_numpy_iterator():
        data_list.append(batch_data)
        labels_list.append(batch_labels)
    
    # Konvertieren Sie die Listen in numpy Arrays
    data_array = np.concatenate(data_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    
    return data_array, labels_array



def split_data(data, train_ratio=0.7, val_ratio=0.15):
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    
    return train, val, test

def get_label_from_filename(filename, class_names):
    for index, class_name in enumerate(class_names):
        if filename.lower().startswith(class_name.lower()):
            return index
    raise ValueError(f'Unmatched filename: {filename}')