from flask import Flask, request, render_template
import cv2
import dlib
import numpy as np
from keras.models import load_model

FACE_DETECTOR_WEIGHTS = '../dlib_data/mmod_human_face_detector.dat'
EMOTION_DETECTOR_WEIGHTS = '../model_weights/emotion_model1.h5py'
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
NO_OF_CHANNELS = 1
emotions = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}


IMAGES_PATH = '../data/user_images/'
def predict_emotion(image):
    '''
    function to predict emotion from face/faces detected in image
    :param image:
    :return:emotion
    '''
    cnn_face_detector = dlib.cnn_face_detection_model_v1(FACE_DETECTOR_WEIGHTS)
    faceRects = cnn_face_detector(image, 1)
    emotion_detector = load_model(EMOTION_DETECTOR_WEIGHTS)
    for face in faceRects:
        cv2.rectangle(image,
                      (face.rect.left(),
                       face.rect.top()),
                      (face.rect.right(),
                       face.rect.bottom()),
                      (0, 255, 0),
                      3)
        face_part = image[face.rect.top():face.rect.bottom(), face.rect.left():face.rect.right()]
        # graysale
        face_part = cv2.cvtColor(face_part, cv2.COLOR_BGR2GRAY)
        # resize
        face_part = cv2.resize(face_part, (IMAGE_HEIGHT, IMAGE_WIDTH))

        face_part = np.reshape(face_part, (1, IMAGE_HEIGHT, IMAGE_WIDTH, NO_OF_CHANNELS))
        face_part = face_part / 255.0
        emotion = emotions[np.argmax(emotion_detector.predict(face_part))]
        cv2.putText(image, emotion,
                    (face.rect.left() - 20, face.rect.top() - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,
                    (255, 255, 255), 2,
                    cv2.LINE_AA)
        # cv2.imwrite()
        return emotion

app = Flask(__name__,
            template_folder='templates')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save image
        f.save(IMAGES_PATH)
        image = cv2.imread(IMAGES_PATH)
        emotion = predict_emotion(image)
        return emotion

    return None


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
