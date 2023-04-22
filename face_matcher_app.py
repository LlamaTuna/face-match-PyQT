import sys
import os
from PyQt5.QtWidgets import QApplication 
from face_matcher_app_class import FaceMatcherApp

pyqt5_path = '/home/vance_octane/projects/face-match_Pyqt/face-match-venv/lib/python3.9/site-packages/PyQt5'
if pyqt5_path in sys.path:
    sys.path.remove(pyqt5_path)
sys.path.append('/usr/lib/python3/dist-packages')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/vance_octane/projects/face-match_Pyqt/face-match-venv/lib/python3.9/site-packages'

current_directory = os.path.dirname(os.path.abspath(__file__))
dark_theme_path = os.path.join(current_directory, "styles", "dark_theme.qss")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    face_matcher_app = FaceMatcherApp()
    face_matcher_app.show()
    sys.exit(app.exec_())
