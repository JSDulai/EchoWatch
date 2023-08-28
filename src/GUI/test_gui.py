import os
import csv
import sys
sys.path.append("../EchoWatch")
import tensorflow as tf
from models.model import create_model
from utils.utilities import load_mp3_16k_mono, preprocess_wav_for_model
from utils.preparation import prepare_data, split_data
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QWidget, QFileDialog, QLabel, QLineEdit, QComboBox, QGroupBox, QSlider, 
                             QStyle, QToolTip, QAction, QMenuBar, QMenu, QTextEdit, QProgressBar)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

class AudioKlassifikatorGUI(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Dein Audio-Klassifikator')
        self.setGeometry(300, 100, 800, 600)
        
        # Hauptmenü
        menubar = self.menuBar()
        fileMenu = QMenu('Datei', self)
        settingsMenu = QMenu('Einstellungen', self)
        helpMenu = QMenu('Hilfe', self)
        
        importAction = QAction('Audio-Datei-Import', self)
        importAction.triggered.connect(self.importAudio)
        fileMenu.addAction(importAction)

        exportAction = QAction('Ergebnisse exportieren', self)
        fileMenu.addAction(exportAction)
        
        themeAction = QAction('Theme-Umschaltung', self)
        settingsMenu.addAction(themeAction)
        
        helpAction = QAction('Anleitung', self)
        helpAction.triggered.connect(self.showHelp)
        helpMenu.addAction(helpAction)
        
        feedbackAction = QAction('Feedback geben', self)
        helpMenu.addAction(feedbackAction)
        
        menubar.addMenu(fileMenu)
        menubar.addMenu(settingsMenu)
        menubar.addMenu(helpMenu)

        # Hauptlayout
        main_layout = QVBoxLayout()

        # Audio-Datei-Import & Player
        audio_layout = QHBoxLayout()
        self.importButton = QPushButton('Audio-Datei-Import', self)
        self.importButton.clicked.connect(self.importAudio)
        audio_layout.addWidget(self.importButton)

        self.playButton = QPushButton('', self)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.playAudio)
        audio_layout.addWidget(self.playButton)
        main_layout.addLayout(audio_layout)
        
        # Spektrogramm-Anzeige
        self.spectrogramLabel = QLabel("Hier wird das Spektrogramm angezeigt", self)
        main_layout.addWidget(self.spectrogramLabel)
        
        # Klassifikations-Button & Ergebnis
        classify_layout = QHBoxLayout()
        self.classifyButton = QPushButton('Klassifikation starten', self)
        self.classifyButton.clicked.connect(self.classifyAudio)
        classify_layout.addWidget(self.classifyButton)

        self.resultLabel = QLabel("Ergebnis wird hier angezeigt", self)
        classify_layout.addWidget(self.resultLabel)
        main_layout.addLayout(classify_layout)

        # Fortschrittsanzeige
        self.progress = QProgressBar(self)
        main_layout.addWidget(self.progress)

        # Anwendungsbereich für andere Funktionen
        # z.B. Datenbankverbindung, Training-Modus, Einstellungen etc.

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        
        self.show()

    def importAudio(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Audio-Datei-Import")
        if filepath:
            # Audio-Verarbeitung hier...
            pass

    def playAudio(self):
        # Audio-Abspielfunktion hier...
        pass

    def classifyAudio(self):
        # Klassifikation des Audios hier...
        pass

    def showHelp(self):
        QToolTip.showText(self.mapToGlobal(self.rect().center()), 'Hier kann eine ausführliche Anleitung stehen...')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioKlassifikatorGUI()
    sys.exit(app.exec_())
