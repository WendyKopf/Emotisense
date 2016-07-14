from sknn.mlp import Regressor, Layer, Classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import glob
import gabor_filters
import parse_jaffe_data
import dlib
import cv2
import numpy as np
import sys
import logging
import pickle
import rotate_image

logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG,
    stream=sys.stdout)

pipeline = pickle.load(open('nnclf2.pk1', 'rb'))


predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

gf = gabor_filters.build_filters()

def generate_feature_vector(img, face):
    feature_vector = []
    shape = predictor(img, face)
    points = [(p.x, p.y) for p in shape.parts()]
    #points_copy = points[:]
    points = points[17:]
    points = points[0:10] + points[14:]
    #img = img[face.top():face.bottom(), face.left():face.right()]
    #img = cv2.resize(img, (155, 155))
    #points = [(x-face.left(), y-face.top()) for (x,y) in points]
    #points = points[17:]
    #print(points)
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
    
def get_faces(img):
    faces = detector(img, 1)
    return faces

def get_emotions(img):
    #face = detector(img, 1)[0]
    emotions = {}
    emotions[0] = [0]
    emotions[1] = [0]
    emotions[2] = [0]
    emotions[3] = [0]
    emotions[4] = [0]
    
    faces = get_faces(img)
    
    if faces is None:
        return emotions
    
    for face in faces:
        face_area = rotate_image.cut(image, face)
        face_area = rotate_image.rotate_face(face_area)
        dim = face_area.shape[0]
        face2 = dlib.rectangle(0, 0, dim, dim)
        face_vector = generate_feature_vector(face_area, face2)
        test_x = np.array([test_feature_vector])
        test_y = pipeline.predict(test_x)
        #array.append(test_y)
        emotions[test_y] = 1
    
    return emotions
    
    #happy: emotions[3]

if __name__ == '__main__':    
    
    #pipeline = pickle.load(open('nnclf2.pk1', 'rb'))

    img_fn = sys.argv[1]

    test_img = cv2.imread(img_fn)

    #test_img = cv2.imread('sad_woman1.jpg')
    #test_img = cv2.imread('neutral_man1.jpg')
    #test_img = cv2.imread('surprised_woman0.jpg')
    #test_img = cv2.imread('scared_woman3.jpg')
    #test_img = cv2.imread('happy_woman2.jpg')
    #test_img = cv2.imread('woman_happy0.jpg')
    #test_img = cv2.imread('angry_man0.jpg')
    #test_img = cv2.imread('jaffe/KM.HA2.5.tiff')
    #test_img = cv2.imread('jaffe/KA.NE2.27.tiff')
    #h,w = test_img.shape[:2]
    #sf = (h*w)/(256*256)
    #test_img = cv2.resize(test_img, (w//sf, h//sf))

    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    
    #test_img = cv2.resize(test_img, (494, 424))
    test_face = get_face(test_img)
    
    face_area = rotate_image.cut_face(test_img, test_face)
    face_area = rotate_image.rotate_image(face_area)
    dim = face_area.shape[0]
    face2 = dlib.rectangle(0, 0, dim, dim)
    
    #sf = 100.0 / float(test_face.right()-test_face.left())
    #test_img = cv2.resize(test_img, None, fx=sf, fy=sf)
    #test_face = get_face(test_img)
    test_feature_vector = generate_feature_vector(face_area, face2)
    test_x = np.array([test_feature_vector])

    test_y = pipeline.predict(test_x)
    print(test_y)

print("done")