#Importing Libraries
import cv2
import time
import mediapipe as mp


# Import drawing utilities from MediaPipe
mp_drawing = mp.solutions.drawing_utils


#Holistic Model from MediaPipe
#Initizialing the model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    static_image_mode = False,
    model_complexity = 1,
    smooth_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
    )

#Video Capture
capture = cv2.VideoCapture(0)

#current time and precious time for FPS
previousTime = 0
currentTime = 0

while capture.isOpened():
    #capture frame by frame
    ret, frame = capture.read()
    
    #Resize the frame for perfect view
    frame = cv2.resize(frame, (800 , 600))
    
    #Converting from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    #Make predicitions using holistic model
    #To improve performance
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    
    #Converting back the RGB to BGR
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    #Drawing the facial landmarks
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(
            color=(255,0,255),
            thickness = 1,
            circle_radius = 1
            ),
        mp_drawing.DrawingSpec(
            color=(255,0,255),
            thickness = 1,
            circle_radius = 1
            )
        )
    
    #drawing Right hand landmarks
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
        )
    
    #drawing Left hand landmarks
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
        )
    
    #Calculating the the FPS
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    
    #Display the FPS
    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    
    #Diplay the resulting image
    cv2.imshow("Face and Hand Landmarks", image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
    