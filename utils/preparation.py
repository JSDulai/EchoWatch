import os
import tensorflow as tf
from utils.utilities import load_audio, preprocess_wav_for_model, load_wav_16k_mono

def prepare_data(dataset_path, class_names, num_classes, batch_size=12, buffer_size=1000):
    file_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path)]
    labels = [tf.keras.utils.to_categorical(get_label_from_filename(filename, class_names), num_classes=num_classes) for filename in os.listdir(dataset_path)]
    data = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    data = data.map(lambda filepath, label: (preprocess_wav_for_model(load_wav_16k_mono(filepath)), label))
    data = data.cache().shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(buffer_size//batch_size)
    return data

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