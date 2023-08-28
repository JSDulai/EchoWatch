import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QSpinBox, QFileDialog, QLineEdit

class ClassifyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel('Unter wie vielen Klassen soll klassifiziert werden?')
        layout.addWidget(self.label)

        self.spinBox = QSpinBox(self)
        self.spinBox.setMinimum(1)
        layout.addWidget(self.spinBox)

        self.generateButton = QPushButton('Generiere Eingabefelder', self)
        self.generateButton.clicked.connect(self.generate_inputs)
        layout.addWidget(self.generateButton)

        self.setLayout(layout)

    def generate_inputs(self):
        num_classes = self.spinBox.value()

        for i in reversed(range(self.layout().count())): 
            widget = self.layout().itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        for i in range(num_classes):
            layout = QVBoxLayout()
            label = QLabel(f'Klasse {i+1} Name:')
            layout.addWidget(label)

            textEdit = QLineEdit(self)
            layout.addWidget(textEdit)

            btn = QPushButton(f'Lade Audio für Klasse {i+1}')
            btn.clicked.connect(self.showDialog)
            layout.addWidget(btn)

            self.layout().addLayout(layout)

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        if fname[0]:
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ClassifyApp()
    ex.show()
    sys.exit(app.exec_())
