import sys
import os
from PyQt5.QtWidgets import QApplication
from face_matcher_app_class import FaceMatcherApp

if getattr(sys, 'frozen', False):
    # If the application is run as a bundled executable
    app_path = sys._MEIPASS
    pyqt5_path = os.path.join(app_path, 'PyQt5')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(pyqt5_path, 'Qt', 'plugins')
else:
    # If the application is run from a script
    app_path = os.path.dirname(os.path.abspath(__file__))
    pyqt5_path = '/home/vance_octane/projects/face-match_Pyqt/face-match-venv/lib/python3.9/site-packages/PyQt5'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/vance_octane/projects/face-match_Pyqt/face-match-venv/lib/python3.9/site-packages'

if pyqt5_path in sys.path:
    sys.path.remove(pyqt5_path)
sys.path.append('/usr/lib/python3/dist-packages')

current_directory = os.path.dirname(os.path.abspath(__file__))
dark_theme_path = os.path.join(getattr(sys, '_MEIPASS', current_directory), "styles", "dark_theme.qss")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    face_matcher_app = FaceMatcherApp()
    face_matcher_app.show()
    sys.exit(app.exec_())
