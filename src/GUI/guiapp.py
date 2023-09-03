from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QFileDialog, QProgressBar, 
                             QTextEdit, QComboBox)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys
sys.path.append("../EchoWatch")
import os
import tensorflow as tf

# Importing functions from the provided files
from utils.preparation import prepare_data1, split_data1, get_label_from_filename1
from models.model import revised_cnn_model_create1, revised_cnn_model_create
# ... and any other required imports

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_signal, num_epochs):
        super().__init__()
        self.progress_signal = progress_signal
        self.num_epochs = num_epochs

    def on_epoch_end(self, epoch, logs=None):
        # Calculate the training progress percentage and emit the signal
        progress_percentage = int((epoch + 1) / self.num_epochs * 100)
        self.progress_signal.emit(progress_percentage)


class TrainModelThread(QThread):
    progress_signal = pyqtSignal(int)  # Signal to update the progress bar

    def __init__(self, model, train_data, val_data, num_epochs=4):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.num_epochs = num_epochs

    def run(self):
        # Create the custom callback
        progress_callback = TrainingProgressCallback(self.progress_signal, self.num_epochs)

        # Train the model using the provided training function and include the callback
        history = self.model.fit(self.train_data, validation_data=self.val_data, 
                                 epochs=self.num_epochs, callbacks=[progress_callback])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Maschinengeräusch Klassifikation")
        self.setGeometry(100, 100, 1000, 700)

        # Main Layout
        self.main_layout = QVBoxLayout()

        # Data Loading
        self.data_loading_layout = QHBoxLayout()
        self.load_data_button = QPushButton("Daten Laden")
        self.load_data_button.clicked.connect(self.load_data)
        self.data_path_label = QLabel("Keine Daten ausgewählt.")
        self.data_loading_layout.addWidget(self.load_data_button)
        self.data_loading_layout.addWidget(self.data_path_label)
        self.main_layout.addLayout(self.data_loading_layout)

        # Model Training
        self.train_model_button = QPushButton("Modell Trainieren")
        self.train_progress = QProgressBar(self)
        self.train_progress.setVisible(False)
        self.train_model_button.clicked.connect(self.train_model)
        self.main_layout.addWidget(self.train_model_button)
        self.main_layout.addWidget(self.train_progress)
                # Prediction
        self.predict_layout = QHBoxLayout()
        self.select_file_combobox = QComboBox()
        self.predict_button = QPushButton("Vorhersage")
        self.predict_button.clicked.connect(self.make_prediction)
        self.predict_layout.addWidget(QLabel("Datei:"))
        self.predict_layout.addWidget(self.select_file_combobox)
        self.predict_layout.addWidget(self.predict_button)
        self.main_layout.addLayout(self.predict_layout)

        # Spectrogram Display
        self.spectrogram_label = QLabel("Spektrogramm wird hier angezeigt.")
        self.spectrogram_label.setAlignment(Qt.AlignCenter)
        self.spectrogram_label.setPixmap(QPixmap())  # Empty QPixmap
        self.main_layout.addWidget(self.spectrogram_label)

        # Save Results
        self.save_results_button = QPushButton("Ergebnisse Speichern")
        self.save_results_button.clicked.connect(self.save_results)
        self.main_layout.addWidget(self.save_results_button)

        # Status and Log Display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.main_layout.addWidget(self.log_display)

        # Set the central widget
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

    import os

    def load_data(self):
        data_path = QFileDialog.getExistingDirectory(self, "Datenverzeichnis auswählen")
        if data_path:
            self.data_path_label.setText(data_path)
        
            # List all files in the selected directory
            file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        
        # Update the ComboBox with the actual file paths
            for file_path in file_paths:
                self.select_file_combobox.addItem(file_path)
        
        # Prepare the data
            data = prepare_data1(data_path)
        
        # Split the data into train, val, and test sets
            self.train_data, self.val_data, self.test_data = split_data1(data)
        
        # Update log display
            self.log_display.append(f"Data loaded and prepared from {data_path}.")
            self.log_display.append(f"Training samples: {len(self.train_data)}")
            self.log_display.append(f"Validation samples: {len(self.val_data)}")
            self.log_display.append(f"Test samples: {len(self.test_data)}")


    def train_model(self):
        # Create the model (if it doesn't exist)
        if not hasattr(self, 'model'):
            self.model = revised_cnn_model_create()

        # Create a thread for model training
        self.train_thread = TrainModelThread(self.model, self.train_data, self.val_data)
        self.train_thread.progress_signal.connect(self.update_training_progress)
        self.train_thread.start()

    def update_training_progress(self, progress):
        self.train_progress.setValue(progress)
        if progress == 100:
            self.log_display.append("Model training completed!")

    def save_results(self):
        save_path = QFileDialog.getSaveFileName(self, "Ergebnisse speichern", "", "CSV Files (*.csv)")[0]
        if save_path:
            # TODO: Save the predictions to the selected CSV file
            pass

    def make_prediction(self):
        if not hasattr(self, 'model'):
            self.log_display.append("Error: Model not trained.")
            return

        # Get selected file from combobox
        selected_file = self.select_file_combobox.currentText()
        if not selected_file:
            self.log_display.append("Error: No file selected for prediction.")
            return

        # TODO: Use the model to make a prediction for the selected file
        prediction = "Class A"  # Placeholder prediction

        # Update log display with the prediction
        self.log_display.append(f"Prediction for {selected_file}: {prediction}")

        # Display the spectrogram for the selected file
        self.display_spectrogram(selected_file)

    def display_spectrogram(self, file_path):
        # TODO: Generate and display the spectrogram for the given file
        placeholder_image = QPixmap()  # Placeholder empty QPixmap
        self.spectrogram_label.setPixmap(placeholder_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
