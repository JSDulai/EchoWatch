import tensorflow as tf

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257, 1)))
    model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu'))
    model.add(tf.keras.layers.Flatten()) #2D flatten auf 1D Daten
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))#Aktivierungsfunktion vielleicht auf softmax ändern.
    model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    #model.compile('Adam', loss='BinaryCrossentropy', metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]) #mit accuracy
    return model

#def create_optimized_model():
    model = tf.keras.models.Sequential()

    # Convolutional layers
    model.add(tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(1491, 257, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # Global Average Pooling
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    # Dense layers
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    # Compile with adjusted learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    return model

#def create_moodel_pt():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(1491, 257, 1)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])

    return model

#def create_moodel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(1491, 257, 1)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, 
              loss='BinaryCrossentropy', 
              metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    return model

#def model_create():

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(1491, 257, 1)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5)) 
    model.add(tf.keras.layers.Dense(4, activation='softmax'))#Aktivierungsfunktion vielleicht auf softmax ändern.\n    
    #model.compile('Adam', loss='CategoricalCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    model.compile('Adam', loss='CategoricalCrossentropy', metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]) #mit accuracy\n    
    return model

#def modified_model_create():
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Flatten(input_shape=(1491, 257, 1)))
    
    # L2 Regularisierung hinzugefügt
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    
    model.add(tf.keras.layers.BatchNormalization())
    
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    # Angepasste Lernrate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(optimizer, loss='CategoricalCrossentropy', metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    return model

#def cnn_model_create():
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1491, 257, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    
    model.compile(optimizer='Adam', loss='CategoricalCrossentropy', 
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    return model

def revised_cnn_model_create():
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
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='CategoricalCrossentropy', 
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    return model
# Model mit Dropout-Layer
def revised_cnn_model_create1(): 
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
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    
    model.compile(optimizer='Adam', loss='CategoricalCrossentropy', 
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    return model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def piczak_modell():
    model = Sequential()

# Anpassung der Größe der Filter und der Pooling-Schichten
    model.add(Conv2D(40, (149, 25), strides=(1, 1), activation='relu', input_shape=(1491, 257, 1)))
    model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))

    model.add(Conv2D(40, (1, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))

# Flattening der Daten für die Fully-Connected-Layer
    model.add(Flatten())

# Anpassung der Anzahl der Neuronen in den Fully-Connected-Layern
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))

# Output Layer mit Softmax-Aktivierung
    model.add(Dense(4, activation='softmax'))

# Kompilieren des Modells
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

