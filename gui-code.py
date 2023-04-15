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
from PyQt5.QtWidgets import QProgressBar



def save_faces_from_folder(folder_path, face_cascade, output_folder, progress_callback=None):
    face_data = {}
    valid_extensions = ['.png', '.jpeg', '.jpg']

    image_names = os.listdir(folder_path)
    num_images = len(image_names)

    for idx, image_name in enumerate(image_names, start=1):
        file_extension = os.path.splitext(image_name)[-1].lower()

        if file_extension not in valid_extensions:
            continue

        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            img_hash = hashlib.sha256(open(image_path, 'rb').read()).hexdigest()
            face_data[img_hash] = {"file_name": image_name, "faces": []}  # Store the original image name
            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                face_data[img_hash]["faces"].append(face_img)
                output_path = os.path.join(output_folder, f"{img_hash}_{len(face_data[img_hash]['faces'])}.png")
                cv2.imwrite(output_path, face_img)

        if progress_callback:
            progress_callback(idx / num_images * 100)

    return face_data


def find_matching_face(image_path, face_cascade, face_data, threshold=0.5):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    matching_faces = []

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        for img_hash, stored_data in face_data.items():
            stored_faces = stored_data["faces"]  # Access the stored faces
            for i, stored_face in enumerate(stored_faces):
                stored_face_resized = cv2.resize(stored_face, (100, 100))
                similarity = np.mean(np.abs(face_img.astype(np.float32) - stored_face_resized.astype(np.float32))) / 255.0

                if similarity < threshold:
                    matching_faces.append((img_hash, stored_data["file_name"], stored_face, similarity, f"{img_hash}_{i+1}.png"))  # Add the original image name

    return matching_faces


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
input_folder = "./faceTests/"
output_folder = "./output/"
    
# GUI code
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QGridLayout, QMessageBox, QScrollArea, QFrame

class FaceMatcherApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Face Matcher')

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        self.progress_bar = QProgressBar()

        layout = QGridLayout(main_widget)
        layout.addWidget(QLabel('Progress:'), 5, 0)
        layout.addWidget(self.progress_bar, 5, 1, 1, 3)
        
        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area, 0, 0, 1, 4)

        # Create a container widget for the scroll area
        scroll_container = QWidget()
        scroll_area.setWidget(scroll_container)

        # Create a layout for the container widget
        scroll_layout = QVBoxLayout(scroll_container)

        # Input folder
        self.input_folder_edit = QLineEdit()
        input_folder_button = QPushButton('Browse')
        input_folder_button.clicked.connect(self.browse_input_folder)
        scroll_layout.addWidget(QLabel('Input folder:'))
        scroll_layout.addWidget(self.input_folder_edit)
        scroll_layout.addWidget(input_folder_button)

        # Output folder
        self.output_folder_edit = QLineEdit()
        output_folder_button = QPushButton('Browse')
        output_folder_button.clicked.connect(self.browse_output_folder)
        scroll_layout.addWidget(QLabel('Output folder:'))
        scroll_layout.addWidget(self.output_folder_edit)
        scroll_layout.addWidget(output_folder_button)

        # Image to search
        self.image_to_search_edit = QLineEdit()
        image_to_search_button = QPushButton('Browse')
        image_to_search_button.clicked.connect(self.browse_image_to_search)
        scroll_layout.addWidget(QLabel('Image to search for:'))
        scroll_layout.addWidget(self.image_to_search_edit)
        scroll_layout.addWidget(image_to_search_button)

        # Image preview
        self.image_preview_label = QLabel()
        scroll_layout.addWidget(self.image_preview_label)

        # Matched face thumbnail
        self.matched_face_label = QLabel()
        scroll_layout.addWidget(self.matched_face_label)

        # Find match button
        find_match_button = QPushButton('Find match')
        find_match_button.clicked.connect(self.find_match)
        
        # Progress label and progress bar
        # scroll_layout.addWidget(QLabel('Progress:'))
        # scroll_layout.addWidget(self.progress_bar)
        
        scroll_layout.addWidget(find_match_button)

        # Result label
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)  # Enable word wrap for better readability
        self.result_label.setFrameShape(QFrame.Box)  # Add a frame to the result label
        self.result_label.setFrameShadow(QFrame.Sunken)
        scroll_layout.addWidget(self.result_label)

        layout.setContentsMargins(10, 10, 10, 10)


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

        face_data = save_faces_from_folder(input_folder, face_cascade, output_folder, progress_callback=self.update_progress_bar)
        matching_faces = find_matching_face(image_to_search, face_cascade, face_data)

        if len(matching_faces) > 0:
            # Get the hash of the input image
            input_image_hash = hashlib.sha256(open(image_to_search, 'rb').read()).hexdigest()

            result_text = f"Match(es) found:\nInput image hash: {input_image_hash}\nInput image file: {os.path.basename(image_to_search)}\n"
            for i, (img_hash, original_image_name, matched_face, similarity, resized_image_name) in enumerate(matching_faces):
                result_text += f"\nMatch {i + 1}:\nOriginal image hash: {img_hash}\nOriginal image file: {original_image_name}\nResized image file: {resized_image_name}"

                if i == 0:
                    self.display_matched_face(matched_face)

            self.result_label.setText(result_text)
        else:
            self.result_label.setText("No match found.")
        print("Finished find_match")


    def update_progress_bar(self, progress):
        self.progress_bar.setValue(int(progress))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    face_matcher_app = FaceMatcherApp()
    face_matcher_app.show()
    sys.exit(app.exec_())
