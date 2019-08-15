import cv2
import dlib
import numpy as np
from keras.models import load_model

lmks_weights = "dlib-data/shape_predictor_68_face_landmarks.dat"
fd_weights = "dlib-data/mmod_human_face_detector.dat"
predictor = dlib.shape_predictor(lmks_weights)
detector = dlib.cnn_face_detection_model_v1(fd_weights)
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
model = load_model('emotion_model.h5')
im = cv2.imread('happy3.jpg')
im = cv2.resize(im, (80, 80))


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


face_parts = ['jaw', 'left_eyebrow', 'right_eyebrow', 'nose', 'left_eye', 'right_eye', 'mouth']


def map_coords_to_face_part(key, landmarks):
    '''
    maps list of coordinates /(x,y)points on human face
    :param key: face part
    :param landmarks: landmarks points
    :return:list of points for specefic parts
    refer to this for more info :https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
    '''
    landmarks_coords = {
        face_parts[0]: landmarks[:17],
        face_parts[1]: landmarks[18:22],
        face_parts[2]: landmarks[23:27],
        face_parts[3]: landmarks[28:35],
        face_parts[4]: landmarks[37:41],
        face_parts[5]: landmarks[43:48],
        face_parts[6]: landmarks[49:68]
    }
    return landmarks_coords[key]


faces = detector(im)
input_vector = []
for face in faces:
    # lmks = shape_to_np(predictor(df['pixels'].iloc[i], face.rect))
    lmks = shape_to_np(predictor(im, face.rect))
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for part in face_parts:
        for idx, (x, y) in enumerate(map_coords_to_face_part(part, lmks)):
            # dftrainp2[part + '_' + 'point_' + str(idx + 1) + '_x'].iloc[i] = x
            # dftrainp2[part + '_' + 'point_' + str(idx + 1) + '_y'].iloc[i] = y
            input_vector.append(x)
            input_vector.append(y)
'''print(input_vector)
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
input_vector=sc.fit_transform(np.array(input_vector).reshape(-1,1))
x=input_vector[:,0]
x=x.reshape(1,-1)
x.shape
print(np.argmax(model.predict(x)))'''
