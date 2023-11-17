#!wget -nd https://github.com/JeffTrain/selfie/raw/master/shape_predictor_68_face_landmarks.dat
#pip install numpy opencv-python dlib scikit-learn keras tensorflow mediapipe

import numpy as np
import cv2
import dlib
import json
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
from keras.optimizers import Adam
import mediapipe as mp

variable_dict = {
    "left_var": 0,
    "right_var": 0,
    "top_var": 0,
    "bottom_var": 0,
    "center_var": 0
}


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

#------------------------------------------------
#LOAD DATA
#--------------------------------------------------

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

"""function"""
def setup():
    # Specify the file paths for the model architecture and weights
    model_architecture_path = 'head_orientation_model/model_architecture.json'
    model_weights_path = 'head_orientation_model/model_weights.h5'
    model_scale_path = 'head_orientation_model/scaler_params.json'

    #-------------- load data from file -----------------

    # Load the scaler parameters from the JSON file
    with open(model_scale_path, 'r') as json_file:
        scaler_params = json.load(json_file)

    # Create a new StandardScaler object and set the mean and scale based on the loaded parameters
    scaler = StandardScaler()
    scaler.mean_ = scaler_params["mean"]
    scaler.scale_ = scaler_params["scale"]


    # Load the model architecture from the JSON file
    with open(model_architecture_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)


    # Load the model weights from the HDF5 file
    loaded_model.load_weights(model_weights_path)

    # Compile the loaded model (make sure to compile it with the same optimizer, loss, and metrics as during training)
    opt=Adam(learning_rate=0.0001)
    loaded_model.compile(optimizer=opt, loss='mean_squared_error')
    return loaded_model,scaler

#------------------------------------------------
#PROCESS DATA
#--------------------------------------------------


def normalize_landmarks(landmarks,w,h):
    landmarks_array = np.array(landmarks)
    centroid = (w/2,h/2)
    average_distance = np.mean(np.linalg.norm(landmarks_array - centroid, axis=1))
    scale_factor = 1.0 / average_distance
    # Normalize landmarks
    normalized_landmarks = (landmarks_array - centroid) * scale_factor
    return normalized_landmarks.tolist()



def find_landmarks(gray):
    faces = face_detector(gray)
    landmarks_list = []

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        # Iterate through the 68 facial landmarks and store (x, y) coordinates
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            landmarks_list.append((x, y))
    return landmarks_list


def check_data(image_data, meta_data):
    return len(image_data) == len(meta_data) > 0
    



def extract_data_v1(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame)
    
    image_data = []
    meta_data = []
    faces = len(results.detections)
    
    if faces <= 0:
        print("[Info] No face detected")
        return None, None

    print(f"[Info] {faces} face(s) detected")
    
    # Check if faces are detected
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face_image = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            face_image = np.array(face_image)
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            landmarks = find_landmarks(face_image) #input gray scale
            print('Landmarks found:', len(landmarks))
            face_image = np.stack((face_image,) * 3, axis=-1)  # convert to 3 channel
            face_image = cv2.resize(face_image, (224, 224))/255.0
            
            if len(landmarks) == 68:
                image_data.append(face_image)
                meta_data.append(normalize_landmarks(landmarks, iw, ih))
        
    
    if check_data(image_data, meta_data): #checking if is equal and greater then 0
        return image_data, meta_data
    elif faces > 0:
        return -1, None
    else:
        return None, None



def check_with_media(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame)
    return len(results.detections)



def extract_data_v3(frame):
    image_data = []
    meta_data = []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    

    if not faces:
        print("[Info] No face detected")
        return None, None

    print(f"[Info] {len(faces)} face(s) detected")

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = frame[y:y+h, x:x+w]
        face_image = np.array(face_image)
        # Resize the face image to 224x224 and normalize pixel values
        landmarks = find_landmarks(face_image)
        print('Landmarks found:', len(landmarks))
        
        face_image = cv2.resize(face_image, (224, 224)) / 255.0

        if len(landmarks) == 68:
            image_data.append(face_image)
            meta_data.append(normalize_landmarks(landmarks, w, h))
        
    
    if check_data(image_data, meta_data):
        return image_data, meta_data
    elif check_with_media(frame) > 0:
        return -1, None
    else:
        return None, None



def extract_data(frame):
    image_data = []
    meta_data = []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    

    if not faces:
        print("[Info] No face detected")
        return None, None

    print(f"[Info] {len(faces)} face(s) detected")

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = frame[y:y+h, x:x+w]
        face_image = np.array(face_image)
        # Resize the face image to 224x224 and normalize pixel values
        landmarks = find_landmarks(face_image)
        print('Landmarks found:', len(landmarks))
        
        face_image = cv2.resize(face_image, (224, 224)) / 255.0

        if len(landmarks) == 68:
            image_data.append(face_image)
            meta_data.append(normalize_landmarks(landmarks, w, h))
        
    
    if check_data(image_data, meta_data):
        return image_data, meta_data
    else:
        return None, None





#------------------------------------------------
# MAP DATA
#--------------------------------------------------

    
def map(x,y,t):
  #t > -5 and t<5
  if y > -x and y < x  :
    return t[1]
  elif -x > y :
    return t[0]
  else :
    return t[2]



def result_gen(pitch,yaw,roll):
    temp=''
    if int(abs(pitch)/2) > abs(roll) :
        temp=map(10,pitch,['top','center','bottom'])
    else:
        temp=map(5,roll,['left','center','right'])
    return temp
    
def fill_dict(temp):
  global variable_dict
  if temp == 'center':
      variable_dict['center_var']+=1
  elif temp == 'top':
      variable_dict['top_var']+=1
  elif temp == 'bottom':
      variable_dict['bottom_var']+=1
  elif temp == 'left':
      variable_dict['left_var']+=1
  else:
      variable_dict['right_var']+=1
  
    
    
    

def map_to_direction(pitch, yaw, roll):
  direction=[]
  direction.append(map(10,pitch,['top','center','bottom']))
  direction.append(map(5,yaw,['forward','center','backward']))
  direction.append(map(5,roll,['left','center','right']))
  return direction


def listtostring(x):
  z=''
  for i in x:
    z+=i+'|'
  return z


def cout(text,img,x=10,y=10):
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.5
  font_color = (0,0,0)  # White color in BGR
  thickness = 2
  cv2.putText(img, str(text), (x, y), font, font_scale, font_color, thickness)
  return img


#------------------------------------------------
# WORK ON DATA
#--------------------------------------------------

store = ""
old_state = ""


temp_data=[]

def run(frame):
  global store 
  global old_state

  loaded_model,scaler=setup()

  image_data,meta_data=extract_data_v3(frame)

  if image_data == -1 :
    print("No change")
    return cout(store,frame) # original frame with old data
  elif image_data == None:
    print("extract data issue")
    return frame # original frame
  x_test=np.array(image_data)
  m_test=np.array(meta_data)
  #print(np.shape(image_data),np.shape(meta_data))
  y_pred=loaded_model.predict((x_test,m_test))
  #print('img',np.shape(image_data),'meta',np.shape(meta_data))
  scaled_y_pred = scaler.inverse_transform(y_pred)

  for p,y,r in scaled_y_pred:
    print(map_to_direction(p,y,r))
    print(p,y,r)
    temp_data.append([p,y,r])
    store = listtostring(map_to_direction(p,y,r))
    frame = cout(store,frame) # do for 1 person
    new_state=result_gen(p,y,r)
    print("old:",old_state,"new:",new_state)
    if new_state != old_state:
      fill_dict(new_state)
      print("update",variable_dict)
      old_state=new_state
  return frame


#-------------------------------------------------
"""_Main"""
#-------------------------------------------------

def head_orientation_model(video_filename):
    try:
        # Open the video file
        #video_filename='video.mp4'                
        
        cap = cv2.VideoCapture(video_filename)  # Replace 'video.mp4' with the video file path
        print("[info] file_name:",video_filename)
        # Get video frame properties (e.g., frame width, height, and frames per second)
        frame_width = int(cap.get(3) if cap.isOpened() else 0)  # Width of frames in the video
        frame_height = int(cap.get(4) if cap.isOpened() else 0)  # Height of frames in the video
        fps = int(cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0)  # Frames per second
        # Get the frame count (number of frames) and frame rate (frames per second)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        # Calculate the video duration in seconds
        video_duration = frame_count / frame_rate

        # Print the video duration
        print(f"[info] Video Duration: {video_duration:.2f} seconds")

        if frame_width == 0 or frame_height == 0 or fps == 0:
            raise ValueError("Video properties not available or invalid.")

        # Define the codec and create a VideoWriter object to write the output video
        output_video_path = 'output_video.mp4'  # Replace with your desired output video file path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video format
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            raise ValueError("Output video writer could not be opened.")

        # Get the frames per second (FPS) of the video
        fps = int(cap.get(1))
        # Set the desired interval to capture one frame per second
        frame_interval = fps  # Capture one frame per second
        # Initialize a frame counter
        frame_counter = 0

        while True:
            ret, frame = cap.read()  # Read a frame from the video

            # Check if the frame was read successfully
            if not ret:
                break  # Break the loop if the video has ended

            # Process the frame using your 'run' function
            out.write(run(frame))
            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and writer
        cap.release()
        out.release()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    print("[Info] video save as output_video.mp4")
    print(variable_dict)
    print("[Info] program terminated")
