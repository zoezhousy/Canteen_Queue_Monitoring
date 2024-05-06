import os
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

# Set the path to the YOLOv5 repository
yolov5_path = "./content/yolov5/"

# Set the path to the image folder
image_folder = "../Dataset/images/Original/Communal1"

# Load the YOLOv5 model
model = torch.hub.load(yolov5_path, "custom", path_or_model="yolov5s.pt")

# Set the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Initialize the counter
total_people = 0

# Iterate over the images in the folder
people_frame = []

image_files = Path(image_folder).glob("*.jpg")  # Adjust the file extension based on your image format
for image_file in tqdm(image_files):
    image = cv2.imread(str(image_file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image, size=640)  # Adjust the size if needed
    number = 0
    for obj in results.xyxy[0]:
        if obj[-1] == 0:  # Class ID for person detection is 0 in YOLOv5
            total_people += 1
            number +=1

    people_frame.append(number)
            

# Print the total number of people
print("Total people detected:", total_people)
print(people_frame)