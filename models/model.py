import tensorflow as tf


def simple_model(num_classes, actiFunc, loss_func):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257, 1)))
    model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation = actiFunc))
    model.compile('Adam', loss=loss_func, metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    return model

def model_optimized(num_classes, actiFunc, loss_func):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(1491, 257, 1)))
    model.add(tf.keras.layers.MaxPooling2D((3, 3)))
    
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((3, 3)))

    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((3, 3)))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(num_classes, activation=actiFunc))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=loss_func, 
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    return model

# Model mit Dropout-Layer
def model_dropout(num_classes, actiFunc, loss_func): 
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1491, 257, 1)))
    model.add(tf.keras.layers.MaxPooling2D((4, 4)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((4, 4)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((4, 4)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(num_classes, activation=actiFunc))
    
    model.compile(optimizer='Adam', loss=loss_func, 
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    return model


def modell_piczak(num_classes, actiFunc, loss_func):
    model = tf.keras.Sequential()

    # First convolutional layer
    model.add(tf.keras.layers.Conv2D(80, (57, 6), strides=(1, 1), activation='relu', input_shape=(1491, 257, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))
    model.add(tf.keras.layers.Dropout(0.5)) 

    # Second convolutional layer
    model.add(tf.keras.layers.Conv2D(80, (1, 3), strides=(1, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))

    # Fully connected layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(5000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(5000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.5))

    # Softmax output layer
    model.add(tf.keras.layers.Dense(num_classes, activation=actiFunc))
    
    # Compile the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.002, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

    return model
