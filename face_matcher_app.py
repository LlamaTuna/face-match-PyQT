import sys
import os
import numpy as np
import hashlib
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageTk
import cv2
from PyQt5.QtWidgets import QAction

# Modify the system path and set the environment variables
pyqt5_path = '/home/vance_octane/projects/face-match_Pyqt/face-match-venv/lib/python3.9/site-packages/PyQt5'
if pyqt5_path in sys.path:
    sys.path.remove(pyqt5_path)
sys.path.append('/usr/lib/python3/dist-packages')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/vance_octane/projects/face-match_Pyqt/face-match-venv/lib/python3.9/site-packages'

# Import the remaining PyQt5 components
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QGridLayout, QMessageBox, QScrollArea, QFrame,QTableWidgetItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from face_matcher_app_class import FaceMatcherApp
from PyQt5.QtWidgets import QHBoxLayout, QStyle




current_directory = os.path.dirname(os.path.abspath(__file__))
dark_theme_path = os.path.join(current_directory, "styles", "dark_theme.qss")
print(dark_theme_path)
print(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if __name__ == '__main__':
    app = QApplication(sys.argv)
    face_matcher_app = FaceMatcherApp()
    face_matcher_app.show()
    sys.exit(app.exec_())
