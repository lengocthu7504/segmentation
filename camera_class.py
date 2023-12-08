from flask import Flask, render_template, Response
import cv2

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm

from torchsummary import summary
# import segmentation_models_pytorch as smp

device = torch.device("cpu")


import segmentation_models_pytorch as smp


class VideoStreamer:
    def __init__(self):
        
        self.setup()
    

    def setup(self):
        self.model = smp.Unet('efficientnet-b3', encoder_weights='imagenet', classes=14, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16], decoder_attention_type="scse")
        self.model = torch.load('./scse_concate.pt', map_location=torch.device('cpu'))
        self.class_index = \
   {
        0: [(64, 128, 64),  'Bed'],
         1: [(192, 0, 64),  'Books'],
         2: [(0, 128, 192),  'Ceiling'],
         3: [(0, 128, 128),   'Chair'],
         4: [(128, 64, 0),    'Floor'],
         5: [(64, 64, 128),   'Furniture'],
         6: [(64, 64, 192),   'Objects'],
         7: [(192, 128, 64), 'Picture'],
         8: [(192, 192, 128),'Sofa'],
         9: [(128, 192, 128),  'Table'],
        10: [(128, 0, 192),  'TV'],
        11: [(192, 0, 192),   'Wall'],
        12: [(128, 128, 64), 'Window'],
        13: [(0, 0, 0),  'Unlabeled'],

    }

        self.class_labels = \
    [
        'bed',
        'books',
        'ceiling',
        'chair',
        'floor',
        'furniture',
        'objects',
        'picture',
        'sofa',
        'table',
        'tv',
        'wall',
        'window',
          'unlabled'
          ]
        self.pipeline = "libcamerasrc ! video/x-raw, width=640, height=480, framerate=15/1 ! videoconvert ! videoscale ! video/x-raw, width=640, height=480 ! appsink"

        self.cap =  cv2.VideoCapture(-1)

        self.transform = A.Resize(480, 640, interpolation=cv2.INTER_NEAREST)




    def predict_image(self, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.model.eval()
        t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        image = t(image)
        # model.to(device); image=image.to(device)
        # mask = mask.to(device)
        with torch.no_grad():

            image = image.unsqueeze(0)
            # mask = mask.unsqueeze(0)

            output = self.model(image)
            # score = mIoU(output, mask)
            masked = torch.argmax(output, dim=1)
            masked = masked.cpu().squeeze(0)
        return masked


# Function to convert a single channel mask representation to an RGB mask.
    def class_to_rgb(self, mask_class, class_index):

    # Create RGB channels
        r_map = np.zeros_like(mask_class).astype(np.uint8)
        g_map = np.zeros_like(mask_class).astype(np.uint8)
        b_map = np.zeros_like(mask_class).astype(np.uint8)

        # Populate RGB color channels based on the color assigned to each class.
        for class_id in range(len(class_index)):
            index = mask_class == class_id
            r_map[index] = class_index[class_id][0][0]
            g_map[index] = class_index[class_id][0][1]
            b_map[index] = class_index[class_id][0][2]

        seg_map_rgb = np.stack([r_map, g_map, b_map], axis=2)

        return seg_map_rgb

    def overlay_image(self, frame, masked_frame):
        masked = np.array(self.class_to_rgb(masked_frame, self.class_index))
        img_overlay = frame.copy()
        cv2.addWeighted(masked, 0.6, frame, 0.4, 0, img_overlay)

        return img_overlay

    def label_segmented_areas(self,segmented_image, labels):
        # Create an output image to draw on
        # output_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)

        for i in range(0, 13):  # Class labels are from 1 to 13
            # Find pixels belonging to the current class
            mask = segmented_image == i
            if np.any(mask):
                # Find the contours of the class region
                contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    # Compute the centroid of the contour
                    M = cv2.moments(cnt)
                    if M['m00'] <= 1000:
                        continue
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                    # Place a label at the centroid
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(segmented_image, labels[i], (cx, cy), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        return segmented_image




    def generate_frames(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Camera not found")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=frame)
            frame = aug['image']
            masked_frame = self.predict_image(frame)
            labeled_image = self.label_segmented_areas(np.ascontiguousarray(masked_frame, dtype=np.uint8), self.class_labels)
            frame_overlay = self.overlay_image(frame, labeled_image)

            _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame_overlay, cv2.COLOR_RGB2BGR))
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
