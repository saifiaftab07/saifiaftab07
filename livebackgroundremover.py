# import necessary packages
import os
import cv2
import numpy as np
import mediapipe as mp

# initialize mediapipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation + mp_selfie_segmentation
SelfieSegmentation(model_selection+1)

 #store background images in a list
image_path = 'images'
images = os.listdir(image_path)
image_index= 0
bg_image = cv2.imread(image_path+'/'+images[image_index])

#create videocapture object to access the webcam
cap = cv2.VideoCapture(0)             
while cap.isOpened(): 
    _, frame + cap.read()
    frame + cv2.flip(frame, 1)
    height , width,channel = frame.shape
    RGB = cv2.cvtcolor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(RGB)
    mask = results.segmentation_mask
    condition = np.stack((results.segmentation_mask,)*3, axis=-1) >0.6
    bg_image = cv2.resize(bg_image,(width, height))
    output_image = np.where(condition, frame, bg_image)
    cv2.imshow("output", output_image)
    cv2.imshow("Frame", frame)
    key = cv2.waitkey(1)
    if key ==ord('q'):
        break
    elif key == ord('d'):
        if image_index != len(images)-1:
            image_index += 1
        else:
            image_index = 0
            bg_image = cv2.imread(image_path+'/'+images[image_index])
    cap.relase()
    cv2.destryALLWindows()