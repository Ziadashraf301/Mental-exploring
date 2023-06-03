# Import necessary modules
import cv2           # OpenCV for image processing
import numpy as np   # NumPy for numerical operations
import pandas as pd  # pandas for data processing and analysis
import tensorflow as tf  # TensorFlow for deep learning

from retinaface import RetinaFace  # RetinaFace for face detection
import matplotlib.pyplot as plt

#emotion recognition model
emotion_recognition_model = tf.keras.models.load_model('CNN_emotion_detection.h5')

def faces_detection(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the input image using RetinaFace
    faces = RetinaFace.detect_faces(image_rgb)

    # Loop over the detected faces and highlight the facial areas
    for face in faces.keys():
        entity = faces[face]
        facial_area = entity["facial_area"]
 
        # Highlight the facial area with a white rectangle
        cv2.rectangle(image, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)

    # Display the original image with the highlighted facial areas
    plt.figure(figsize=(20,20))
    plt.imshow(image[:,:,::-1])
    plt.axis('off')
    plt.show()


def face_emotion(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract the face regions from the input image using RetinaFace
    faces = RetinaFace.extract_faces(image_rgb, align=True)

    # Preprocess each face region and make emotion predictions
    for face in faces:
        # Resize and preprocess the face region for input to the emotion recognition model
        resized_face_region = cv2.resize(face, (48, 48))
        preprocessed_face_region = np.expand_dims(resized_face_region, axis=0) / 255.0

        # Convert the face region from RGB to BGR format
        new_img = cv2.cvtColor(resized_face_region, cv2.COLOR_RGB2BGR)
        plt.imshow(new_img)
        plt.axis('off')
        plt.show()

        # Make emotion prediction using the pre-trained model and print the result
        prob_sad = emotion_recognition_model.predict(preprocessed_face_region)[0][0]
        print('Probability of being sad:', prob_sad)


faces_detection('istockphoto-1141421597-612x612.jpg')
face_emotion('istockphoto-1141421597-612x612.jpg')