import cv2
import dlib
import numpy

global video_source


def set_cam_brightness(val):
    '''Sets camera brightness to input value (0 to 255)'''
    video_source.set(10,val)
    
def set_capture_interval(val):
    '''Sets webcam capture interval to input value (in ms)'''
    capture_interval = val

def detect_faces(gray, sf = 1.3, minS = (50,50), maxS = (200,200)):
    '''Returns bounding boxes of faces for input image using scale factor and size constraints'''
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor = sf, minSize = minS, maxSize = maxS)
    return faces
    
def detect_profiles(gray, sf = 1.3, minS = (50,50), maxS = (200,200)):
    '''Returns bounding boxes of profiles for input image using scale factor and size constraints'''
    
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor = sf, minSize = minS, maxSize = maxS)
    return profiles
    
def draw_faces_old(img, faces):
    '''Draws bounding boxes of faces on input image'''
    
    for (x,y,w,h) in faces:
        img2 = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

def draw_face(img, left, top, right, bot):
    '''Draws bounding box of face on input image'''
    
    cv2.rectangle(img, (left, top), (right, bot), (255, 0, 0), 2)
        
def draw_points(img, points):
    '''Draws list of points on input image'''
    for p in points:
        cv2.circle(img, p, 2, color=(0,0,255))
        
def draw_points_numbered(img, points):
    '''Draws numbered list of points on input image'''
    font = cv2.FONT_HERSHEY_SIMPLEX
    for n, p in enumerate(points):
        cv2.putText(img, str(n), p, font, 0.25, (0,0,255))        
        
def detect_eyes(gray, (x,y,w,h)):
    '''Returns bounding boxes of eyes for input image'''
    
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    return eyes
    
def draw_eyes(img, (x,y,w,h), eyes):
    '''Draws bounding boxes of eyes on input image'''
    
    roi_color = img[y:y+h, x:x+w]
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
        
if __name__ == '__main__':
    import sys
    
    img_fn = sys.argv[1]
    
    img = cv2.imread(img_fn)
    #h,w = img.shape[:2]
    #sf = (h*w)/(256*256)
    #img = cv2.resize(img, (w/sf, h/sf))
    #img = cv2.resize(img, (450, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    faces = detector(img, 1)
    face_region = None
    
    for face in faces:
        shape = predictor(img, face)
        # face_coords = ((face.left(), face.top()), (face.right(), face.bottom()))
        print(face.right()-face.left(), face.bottom()-face.top())
        draw_face(img, face.left(), face.top(), face.right(), face.bottom())
        points = [(p.x, p.y) for p in shape.parts()]
        #print(points[17:])
        draw_points_numbered(img, points)
        #print(img.item(167,407))
        #face_region = img[face.top():face.bottom(), face.left():face.right()]  
        #print(face_region)
        #print(img)
    cv2.imshow('features', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()