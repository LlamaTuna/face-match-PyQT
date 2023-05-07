import os
import cv2
import hashlib
import numpy as np
import dlib

input_folder = "./faceTests/"
output_folder = "./output/"
import sys
import os

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        # Running as a bundled executable
        base_path = sys._MEIPASS
    else:
        # Running as a normal script
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

cnn_face_detector_path = resource_path("tests/mmod_human_face_detector.dat")
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_path)


def convert_image_format(image_path, output_format='png'):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image from {image_path}")
    output_path = os.path.splitext(image_path)[0] + '.' + output_format
    cv2.imwrite(output_path, img)
    return output_path

def save_faces_from_folder(folder_path, face_cascade, output_folder, progress_callback=None):
    face_data = {}
    valid_extensions = ['.png', '.jpeg', '.jpg', '.bmp']

    image_names = os.listdir(folder_path)
    num_images = len(image_names)

    for idx, image_name in enumerate(image_names, start=1):
        file_extension = os.path.splitext(image_name)[-1].lower()

        if file_extension not in valid_extensions:
            continue

        image_path = os.path.join(folder_path, image_name)

        try:
            converted_image_path = convert_image_format(image_path, output_format='png')
            img = cv2.imread(converted_image_path)
            assert img is not None, f"Image at {converted_image_path} is None"

            detected_faces = cnn_face_detector(img, 1)
            faces = [(face.rect.top(), face.rect.right(), face.rect.bottom(), face.rect.left()) for face in detected_faces]


            if len(faces) > 0:
                img_hash = hashlib.sha256(open(converted_image_path, 'rb').read()).hexdigest()
                face_data[img_hash] = {"file_name": image_name, "faces": []}
                for (top, right, bottom, left) in faces:
                    face_img = img[top:bottom, left:right]
                    face_data[img_hash]["faces"].append(face_img)
                    output_path = os.path.join(output_folder, f"{img_hash}_{len(face_data[img_hash]['faces'])}.png")
                    cv2.imwrite(output_path, face_img)

            if progress_callback:
                progress_callback(idx / num_images * 100)

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")

    return face_data

def find_matching_face(image_path, face_data, threshold=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image from {image_path}")

    detected_faces = cnn_face_detector(img, 1)
    matching_faces = []

    for rect in detected_faces:
        left, top, right, bottom = rect.rect.left(), rect.rect.top(), rect.rect.right(), rect.rect.bottom()
        face_img = img[top:bottom, left:right]
        face_img = cv2.resize(face_img, (100, 100))
        for img_hash, stored_data in face_data.items():
            stored_faces = stored_data["faces"]
            for i, stored_face in enumerate(stored_faces):
                if stored_face.size == 0:  # Add this check for empty images
                    continue
                stored_face_resized = cv2.resize(stored_face, (100, 100))
                similarity = np.mean(np.abs(face_img.astype(np.float32) - stored_face_resized.astype(np.float32))) / 255.0

                if similarity < threshold:
                    matching_faces.append((img_hash, stored_data["file_name"], stored_face, similarity, f"{img_hash}_{i+1}.png"))

    return matching_faces
