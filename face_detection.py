import os
import cv2
import hashlib
import numpy as np
import face_recognition

input_folder = "./faceTests/"
output_folder = "./output/"

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
        faces = face_recognition.face_locations(img)

        if len(faces) > 0:
            img_hash = hashlib.sha256(open(image_path, 'rb').read()).hexdigest()
            face_data[img_hash] = {"file_name": image_name, "faces": []}  # Store the original image name
            for (top, right, bottom, left) in faces:
                face_img = img[top:bottom, left:right]
                face_data[img_hash]["faces"].append(face_img)
                output_path = os.path.join(output_folder, f"{img_hash}_{len(face_data[img_hash]['faces'])}.png")
                cv2.imwrite(output_path, face_img)

        if progress_callback:
            progress_callback(idx / num_images * 100)

    return face_data

def find_matching_face(image_path, face_data, threshold=0.5):

    img = cv2.imread(image_path)
    faces = face_recognition.face_locations(img)

    matching_faces = []

    for (top, right, bottom, left) in faces:
        face_img = img[top:bottom, left:right]
        face_img = cv2.resize(face_img, (100, 100))
        for img_hash, stored_data in face_data.items():
            stored_faces = stored_data["faces"]  # Access the stored faces
            for i, stored_face in enumerate(stored_faces):
                stored_face_resized = cv2.resize(stored_face, (100, 100))
                similarity = np.mean(np.abs(face_img.astype(np.float32) - stored_face_resized.astype(np.float32))) / 255.0

                if similarity < threshold:
                    matching_faces.append((img_hash, stored_data["file_name"], stored_face, similarity, f"{img_hash}_{i+1}.png"))  # Add the resized image name
    return matching_faces