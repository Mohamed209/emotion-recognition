import time
from multiprocessing import cpu_count

import cv2
import numpy as np
import pandas as pd

nCores = cpu_count()

start = time.time()

TextFileReader = pd.read_csv('fer2013.csv', chunksize=1000)
end = time.time()
print(end - start)

s = time.time()
dflist = []
for df in TextFileReader:
    dflist.append(df)
df = pd.concat(dflist, sort=False)
e = time.time()
print(e - s)


# todo parallel processing str_to_image function
def str_to_image(row):
    image = np.array([i for i in row.split(' ')], dtype=np.uint8).reshape(48, 48)
    image = cv2.resize(image, (80, 80))
    return image


df['pixels'] = df['pixels'].apply(lambda x: str_to_image(x))

columns = []
for i in range(17):
    columns.append('jaw_point_' + str(i + 1) + '_x')
    columns.append('jaw_point_' + str(i + 1) + '_y')
for i in range(5):
    columns.append('left_eyebrow_point_' + str(i + 1) + '_x')
    columns.append('left_eyebrow_point_' + str(i + 1) + '_y')
    columns.append('right_eyebrow_point_' + str(i + 1) + '_x')
    columns.append('right_eyebrow_point_' + str(i + 1) + '_y')
for i in range(9):
    columns.append('nose_point_' + str(i + 1) + '_x')
    columns.append('nose_point_' + str(i + 1) + '_y')
for i in range(6):
    columns.append('left_eye_point_' + str(i + 1) + '_x')
    columns.append('left_eye_point_' + str(i + 1) + '_y')
    columns.append('right_eye_point_' + str(i + 1) + '_x')
    columns.append('right_eye_point_' + str(i + 1) + '_y')
for i in range(20):
    columns.append('mouth_point_' + str(i + 1) + '_x')
    columns.append('mouth_point_' + str(i + 1) + '_y')
features_dict = {key: [] for key in columns}
import dlib

lmks_weights = "dlib-data/shape_predictor_68_face_landmarks.dat"
fd_weights = "dlib-data/mmod_human_face_detector.dat"
predictor = dlib.shape_predictor(lmks_weights)
detector = dlib.cnn_face_detection_model_v1(fd_weights)


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


# dftrainp1 = pd.DataFrame(df['emotion'])
pd.DataFrame(df['emotion']).to_pickle('labels.pkl')


# dftrainp2 = pd.DataFrame(index=range(len(df['pixels'])), columns=columns)


def convert_dataset(row):
    # faces = detector(df['pixels'].iloc[i])
    faces = detector(row)
    for face in faces:
        # lmks = shape_to_np(predictor(df['pixels'].iloc[i], face.rect))
        lmks = shape_to_np(predictor(row, face.rect))
        vector = []
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for part in face_parts:
            for idx, (x, y) in enumerate(map_coords_to_face_part(part, lmks)):
                # dftrainp2[part + '_' + 'point_' + str(idx + 1) + '_x'].iloc[i] = x
                # dftrainp2[part + '_' + 'point_' + str(idx + 1) + '_y'].iloc[i] = y
                features_dict[part + '_' + 'point_' + str(idx + 1) + '_x'].append(x)
                features_dict[part + '_' + 'point_' + str(idx + 1) + '_y'].append(y)


df['pixels'].apply(lambda x: convert_dataset(x))
import pickle

pickle.dump(features_dict, open("features.pkl", "wb"))
'''import multiprocessing as mp

with mp.Pool(4) as pool:
    result = pool.imap(convert_dataset(), df.itertuples(name=False), chunksize=10)
    output = [round(x, 2) for x in result]

#dftrain = pd.concat([dftrainp1, dftrainp2], axis=1)
#print(dftrain)
#dftrain.to_csv('fer_lmks.csv')'''
print("finished :)")
