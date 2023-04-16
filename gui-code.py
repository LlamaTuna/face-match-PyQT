import sys
import os
import numpy as np
import hashlib
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageTk
import cv2
from PyQt5.QtWidgets import QAction
pyqt5_path = '/home/vance_octane/projects/face-match_Pyqt/face-match-venv/lib/python3.9/site-packages/PyQt5'
if pyqt5_path in sys.path:
    sys.path.remove(pyqt5_path)
sys.path.append('/usr/lib/python3/dist-packages')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/vance_octane/projects/face-match_Pyqt/face-match-venv/lib/python3.9/site-packages'
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QGridLayout, QMessageBox, QScrollArea, QFrame,QTableWidgetItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem

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
                    matching_faces.append((img_hash, stored_data["file_name"], stored_face, similarity, f"{img_hash}_{i+1}.png"))  # Add the resized image name
    return matching_faces

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
input_folder = "./faceTests/"
output_folder = "./output/"
    
# GUI code
class NumericTableWidgetItem(QTableWidgetItem):
    def __init__(self, value):
        super().__init__(str(value))

    def __lt__(self, other):
        my_number = float(self.text().rstrip('%'))
        other_number = float(other.text().rstrip('%'))
        return my_number < other_number
  
class MatchTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        my_number = int(self.text().split()[-1])
        other_number = int(other.text().split()[-1])
        return my_number < other_number
    
class FaceMatcherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dark_theme_enabled = False
        self.initUI()

    def create_menu_bar(self):
        menubar = self.menuBar()

        # Create File menu
        file_menu = menubar.addMenu('File')

        # Create Exit action
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Create View menu
        view_menu = menubar.addMenu('View')

        # Create Toggle Dark Theme action
        toggle_dark_theme_action = QAction('Toggle Dark Theme', self)
        toggle_dark_theme_action.triggered.connect(self.toggle_dark_theme)
        view_menu.addAction(toggle_dark_theme_action)
    
    def toggle_dark_theme(self):
        if self.dark_theme_enabled:
            self.dark_theme_enabled = False
            self.setStyleSheet("")
        else:
            self.dark_theme_enabled = True
            current_directory = os.path.dirname(os.path.abspath(__file__))
            dark_theme_path = os.path.join(current_directory, "styles", "dark_theme.qss")
            self.setStyleSheet(load_stylesheet(dark_theme_path))

    def initUI(self):
        self.setWindowTitle('Face Matcher')
        self.create_menu_bar()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        self.progress_bar = QProgressBar()

        # self.exit_button = QPushButton("Exit", self)
        # self.exit_button.setFixedSize(50, 20)  # Set the size of the exit button
        # self.exit_button.clicked.connect(self.close)
        layout = QGridLayout(main_widget)
        layout.addWidget(QLabel('Progress:'), 5, 0)
        layout.addWidget(self.progress_bar, 5, 1, 1, 2)  # Update the column span from 3 to 2
        # layout.addWidget(self.exit_button, 5, 3, alignment=Qt.AlignRight)  # Set the alignment to AlignRight

        
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
     
        scroll_layout.addWidget(find_match_button)

        # Result label
        self.result_table = QTableWidget(self)
        self.result_table.setSortingEnabled(True)

        layout.addWidget(self.result_table, 7, 0, 1, 3)

        layout.setContentsMargins(10, 10, 10, 10)

    def browse_input_folder(self):
        print("Browsing input folder")
        input_folder = QFileDialog.getExistingDirectory(self, 'Select Input Folder', options=QFileDialog.DontUseNativeDialog)
        if input_folder:
            self.input_folder_edit.setText(input_folder)
        print("Finished browsing input folder")

    def browse_output_folder(self):
        print("Browsing output folder")
        output_folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder', options=QFileDialog.DontUseNativeDialog)
        if output_folder:
            self.output_folder_edit.setText(output_folder)
        print("Finished browsing output folder")

    def browse_image_to_search(self):
        print("Browsing image to search")
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Image to Search', '', 'Image files (*.png *.jpeg *.jpg *.bmp *.tiff)', options=QFileDialog.DontUseNativeDialog)
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
            self.result_table.setColumnCount(5)
            self.result_table.setHorizontalHeaderLabels(['Match', 'Similarity', 'Original Image File', 'Original Image Hash', 'Resized Image File'])
            self.result_table.setRowCount(len(matching_faces))

            for i, (img_hash, original_image_name, matched_face, similarity, resized_image_name) in enumerate(matching_faces):
                self.result_table.setItem(i, 0, MatchTableWidgetItem(f"Match {i + 1}"))
                self.result_table.setItem(i, 1, NumericTableWidgetItem(f"{similarity * 100:.2f}%"))
                self.result_table.setItem(i, 2, QTableWidgetItem(original_image_name))
                self.result_table.setItem(i, 3, QTableWidgetItem(img_hash))
                self.result_table.setItem(i, 4, QTableWidgetItem(resized_image_name))

                if i == 0:
                    self.display_matched_face(matched_face)

            self.result_table.resizeColumnsToContents()
        else:
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)

        print("Finished find_match")

    def update_progress_bar(self, progress):
        self.progress_bar.setValue(int(progress))
    
def load_stylesheet(file_path):
    with open(file_path, "r") as file:
        return file.read()


current_directory = os.path.dirname(os.path.abspath(__file__))
dark_theme_path = os.path.join(current_directory, "styles", "dark_theme.qss")
print(dark_theme_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    face_matcher_app = FaceMatcherApp()
    face_matcher_app.show()
    sys.exit(app.exec_())
