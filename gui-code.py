import sys
import os
import numpy as np
import hashlib
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageTk
import cv2

pyqt5_path = '/home/vance_octane/projects/face-match_Pyqt/face-match-venv/lib/python3.9/site-packages/PyQt5'
if pyqt5_path in sys.path:
    sys.path.remove(pyqt5_path)

sys.path.append('/usr/lib/python3/dist-packages')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/vance_octane/projects/face-match_Pyqt/face-match-venv/lib/python3.9/site-packages'

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QGridLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


def save_faces_from_folder(folder_path, face_cascade, output_folder):
    face_data = {}
    valid_extensions = ['.png', '.jpeg', '.jpg']

    for image_name in tqdm(os.listdir(folder_path), desc="Processing images"):
        file_extension = os.path.splitext(image_name)[-1].lower()

        if file_extension not in valid_extensions:
            continue

        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            img_hash = hashlib.sha256(open(image_path, 'rb').read()).hexdigest()
            face_data[img_hash] = []
            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                face_data[img_hash].append(face_img)
                output_path = os.path.join(output_folder, f"{img_hash}_{len(face_data[img_hash])}.png")
                cv2.imwrite(output_path, face_img)
    return face_data


def find_matching_face(image_path, face_cascade, face_data, threshold=0.5):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    best_similarity = float('inf')
    best_img_hash = None
    best_stored_face = None
    best_original_image_name = None

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        for img_hash, stored_faces in face_data.items():
            for i, stored_face in enumerate(stored_faces):
                stored_face_resized = cv2.resize(stored_face, (100, 100))
                similarity = np.mean(np.abs(face_img.astype(np.float32) - stored_face_resized.astype(np.float32))) / 255.0

                if similarity < best_similarity:
                    best_similarity = similarity
                    best_img_hash = img_hash
                    best_stored_face = stored_face
                    best_original_image_name = f"{img_hash}_{i+1}.png"

    if best_similarity < threshold:
        return best_img_hash, best_stored_face, best_similarity, best_original_image_name

    return None, None, None, None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
input_folder = "./faceTests/"
output_folder = "./output/"
    
# GUI code
class FaceMatcherApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Face Matcher')

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QGridLayout(main_widget)

        # Input folder
        self.input_folder_edit = QLineEdit()
        input_folder_button = QPushButton('Browse')
        input_folder_button.clicked.connect(self.browse_input_folder)
        layout.addWidget(QLabel('Input folder:'), 0, 0)
        layout.addWidget(self.input_folder_edit, 0, 1)
        layout.addWidget(input_folder_button, 0, 2)

        # Output folder
        self.output_folder_edit = QLineEdit()
        output_folder_button = QPushButton('Browse')
        output_folder_button.clicked.connect(self.browse_output_folder)
        layout.addWidget(QLabel('Output folder:'), 1, 0)
        layout.addWidget(self.output_folder_edit, 1, 1)
        layout.addWidget(output_folder_button, 1, 2)

        # Image to search
        self.image_to_search_edit = QLineEdit()
        image_to_search_button = QPushButton('Browse')
        image_to_search_button.clicked.connect(self.browse_image_to_search)
        layout.addWidget(QLabel('Image to search:'), 2, 0)
        layout.addWidget(self.image_to_search_edit, 2, 1)
        layout.addWidget(image_to_search_button, 2, 2)

        # Image preview
        self.image_preview_label = QLabel()
        layout.addWidget(self.image_preview_label, 2, 3)

        # Matched face thumbnail
        self.matched_face_label = QLabel()
        layout.addWidget(self.matched_face_label, 3, 3)

        # Find match button
        find_match_button = QPushButton('Find match')
        find_match_button.clicked.connect(self.find_match)
        layout.addWidget(find_match_button, 3, 0, 1, 3)

        # Result label
        self.result_label = QLabel()
        layout.addWidget(self.result_label, 4, 0, 1, 3)

    def browse_input_folder(self):
        print("Browsing input folder")
        input_folder = QFileDialog.getExistingDirectory(self, 'Select Input Folder')
        if input_folder:
            self.input_folder_edit.setText(input_folder)
        print("Finished browsing input folder")

    def browse_output_folder(self):
        print("Browsing output folder")
        output_folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if output_folder:
            self.output_folder_edit.setText(output_folder)
        print("Finished browsing output folder")

    def browse_image_to_search(self):
        print("Browsing image to search")
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Image to Search', '', 'Image files (*.png *.jpeg *.jpg *.bmp *.tiff)')
        if file_path:
            self.image_to_search_edit.setText(file_path)
            self.load_image_thumbnail(file_path)
        print("Finished browsing image to search")

    def load_image_thumbnail(self, file_path):
        print("Loading image thumbnail")
        image = QImage(file_path)
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(100, 100, aspectRatioMode=Qt.KeepAspectRatio)
        self.image_preview_label.setPixmap(scaled_pixmap)
        print("Finished loading image thumbnail")
        
    def display_matched_face(self, matched_face):
        print("Displaying matched face")
        height, width, _ = matched_face.shape
        bytes_per_line = width * 3
        q_image = QImage(matched_face.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(100, 100, aspectRatioMode=Qt.KeepAspectRatio)
        self.matched_face_label.setPixmap(scaled_pixmap)
        print("Finished displaying matched face")

    def find_match(self):
        print("Starting find_match")
        input_folder = self.input_folder_edit.text()
        output_folder = self.output_folder_edit.text()
        image_to_search = self.image_to_search_edit.text()

        if not input_folder or not output_folder or not image_to_search:
            QMessageBox.critical(self, "Error", "Please select all required folders and files.")
            return

        face_data = save_faces_from_folder(input_folder, face_cascade, output_folder)
        img_hash, matched_face, similarity, original_image_name = find_matching_face(image_to_search, face_cascade, face_data)

        if matched_face is not None:
            # Get the hash of the input image
            input_image_hash = hashlib.sha256(open(image_to_search, 'rb').read()).hexdigest()
            
            # Display the input image hash, matched image hash, and the original filename of the matched image
            self.result_label.setText(f"Match found.\nInput image hash: {input_image_hash}\nInput image file: {os.path.basename(image_to_search)}\nOriginal image hash: {img_hash}\nOriginal image file: {original_image_name}")
            self.display_matched_face(matched_face)
        else:
            self.result_label.setText("No match found.")
        print("Finished find_match")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    face_matcher_app = FaceMatcherApp()
    face_matcher_app.show()
    sys.exit(app.exec_())
