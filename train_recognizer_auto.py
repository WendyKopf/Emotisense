import cv2, os
import numpy as np
from PIL import Image
from cv2.cv import * 
from cv2 import * 

import sys

NamedWindow("w1", CV_WINDOW_AUTOSIZE)

def get_frames(n_times):
    results = []
    capture = cv2.VideoCapture(0)
    for i in xrange(n_times):
        ret, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        results.append(frame)
        ShowImage("w1", cv2.cv.fromarray(frame))
        WaitKey(10)
    return results

def run_training(training_data, identity, frames):
    # For face detection we will use the Haar Cascade provided by OpenCV.
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # For face recognition we will the the LBPH Face Recognizer 
    recognizer = cv2.createLBPHFaceRecognizer()
    try:
        recognizer.load(training_data)
    except Exception:
        pass

    images, labels = [], []
    for frame in frames:
        # Convert the image format into numpy array
        image = np.array(frame, 'uint8')

        for (x, y, w, h) in faceCascade.detectMultiScale(image):
            images.append(image[y: y + h, x: x + w])
            labels.append(int(identity))
            cv2.waitKey(1)

    print len(images)
    cv2.destroyAllWindows()


    # Perform the training
    recognizer.update(images, np.array(labels));
    recognizer.save(training_data)

if __name__ == "__main__":
    run_training(sys.argv[1], sys.argv[2] ,get_frames(500))
