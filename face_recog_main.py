#actual recognition of face 
import cv2, os
import numpy as np
from PIL import Image

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('recognizer_training')


if __name__ == "__main__":

    path = './yalefaces'

    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
    for image_path in image_paths:
        predict_image_pil = Image.open(image_path).convert('L')
        predict_image = np.array(predict_image_pil, 'uint8')
        faces = faceCascade.detectMultiScale(predict_image)
        for (x, y, w, h) in faces:
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            #nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        
            if conf > 40:
                print "{} is recognized with confidence {}".format(nbr_actual, conf)
	    else:
	        print "{} is not recognized ".format(nbr_actual)
	
	
	
	    cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
            cv2.waitKey(1000)