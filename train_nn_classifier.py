from sknn.mlp import Regressor, Layer, Classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import glob
import gabor_filters
import parse_jaffe_data
import re
import dlib
import cv2
import numpy as np
import sys
import os
import logging
import pickle
import rotate_image

logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG,
    stream=sys.stdout)

predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

gf = gabor_filters.build_filters()

def generate_feature_vector(img, face):
    feature_vector = []
    shape = predictor(img, face)
    points = [(p.x, p.y) for p in shape.parts()]
    points = points[17:]
    points = points[0:10] + points[14:]
    #points = points[17:]
    for filter in gf:
        img_filtered = cv2.filter2D(img, cv2.CV_8UC1, filter)
        for (x,y) in points:
            #x = max(0,x)
            #x = min(255,x)
            #y = max(0,y)
            #y = min(255,y)
            #feature_vector.append(img_filtered[x][y])
            feature_vector.append(img_filtered.item(y,x))
    return feature_vector

def get_face(img):
    face = detector(img, 1)[0]
    return face
    
def stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

training_files = glob.glob('/home/wendy/Desktop/EmotionDetector/FaceGrabberNew/*')
training_files = sorted(training_files, key = stringSplitByNumbers)

#print(training_files)
#print(len(training_files))
training_input = []
for num, file in enumerate(training_files):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face = get_face(img)
    face_area = rotate_image.cut_face(img, face)
    face_area = rotate_image.rotate_image(face_area)
    dim = face_area.shape[0]
    face2 = dlib.rectangle(0, 0, dim, dim)

    training_input.append(generate_feature_vector(face_area, face2))
    print("Processed training file ", num)

train_x = np.array(training_input)

#pickle.dump(train_x, open('fg_train_x.pk1', 'wb'))

training_output = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

train_y = np.array(training_output)
   
# pipeline = Pipeline([
    # ('min/max scaler', MinMaxScaler(feature_range=(0.0,1.0))),
    # ('neural network', Regressor(
        # layers = [
            # #Layer("Linear", units=500),
            # Layer("Sigmoid", units=600),
            # Layer("Linear")],
        # learning_rate = 0.01,
        # n_iter = 200))])   
 
pipeline = Pipeline([
    ('min/max scaler', MinMaxScaler(feature_range=(0.0,1.0))),
    ('neural network', Classifier(
        layers = [
            #Layer("Linear", units=500),
            Layer("Sigmoid", units=400),
            Layer("Softmax")],
        learning_rate = 0.01,
        n_iter = 200))])   
 
pipeline.fit(train_x, train_y)

pickle.dump(pipeline, open('nnclf2.pk1', 'wb'))
#test_img = cv2.imread('sad_woman1.jpg')
#test_img = cv2.imread('jaffe/KM.HA2.5.tiff')
#test_face = get_face(test_img)
#test_feature_vector = generate_feature_vector(test_img, test_face)
#test_x = np.array([test_feature_vector])

#test_y = pipeline.predict(test_x)
#print(test_y)

print("done")