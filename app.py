from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

app = Flask(__name__)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def draw_styled_landmarks(image, results):
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
      
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

actions = np.array(['Indian', 'Sign', 'Language', 'Man', 'Woman', 'Namaste', 'Deaf', 'Sorry', 'Happy', 'Sad', 'Understand', 'Bye', 'Please', 'Food', 'Water'])

model = load_model('actions_95%_0.1_1000.h5')

def sign_lang():

    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:

        while cap.isOpened():
            # Capture frame-by-frame
            success, frame = cap.read()  # read the camera frame

            image, results = mediapipe_detection(frame, holistic)

            draw_styled_landmarks(image, results)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
            
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold:
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                # accuracy.append(str(round(int(res[np.argmax(res)]*100),2)))
                        else:
                            sentence.append(actions[np.argmax(res)])
                            # accuracy.append(str(round(int(res[np.argmax(res)]*100),2)))

            if len(sentence) > 1: 
                sentence = sentence[-1:]
                # accuracy = accuracy[-1:]
            
            string = str(sentence);  
            count = 0;  
            
            for i in range(0, len(string)):  
                if(string[i] != ' '):  
                    count = count + 1
            
            if (count==7):
                count = count + 1
            elif (count==14):
                count = count - 2   

                
            cv2.rectangle(image, (0,0), (count*15, 40), (245, 117, 16), -1)
            cv2.putText(image,''.join(sentence), (15,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            

            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # 

            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():


    #Video streaming route. Put this in the src attribute of an img tag
    return Response(sign_lang(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)