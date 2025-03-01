import cv2
import tkinter as tk
from tkinter import filedialog

# Function to build the rectangular shape around detected faces
def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bbox.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bbox

# Paths to the models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Load the models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-13)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60+)']
genderList = ['Male', 'Female']

# Create a Tkinter root window and hide it
root = tk.Tk()
root.withdraw()

# Open a file dialog to choose the image
imagePath = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
)

# Check if a file was selected
if imagePath:
    # Load the image
    frame = cv2.imread(imagePath)
    
    if frame is None:
        print("Error loading image")
    else:
        # Detect faces and predict age and gender
        frame, bbox = faceBox(faceNet, frame)
        for bb in bbox:
            x1, y1, x2, y2 = bb
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                print("Invalid bounding box coordinates")
                continue
            
            face = frame[y1:y2, x1:x2]
            
            if face.size == 0:
                print("Empty face region")
                continue
            
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Predict gender
            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]
            
            # Predict age
            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]

            # Draw label and rectangle around the face
            label = "{},{}".format(gender, age)
            cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the output
        cv2.imshow("Age-Gender Prediction", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("No file selected")
