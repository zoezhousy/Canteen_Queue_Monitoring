# %% [markdown]
# # Preprocessing

# %% [markdown]
# ## Preprocessing Videos

# %%
# Get video list, and convert to picture
import os
import matplotlib.pyplot as plt

folder_path = '../Dataset/video'
file_names = [files for files in os.listdir(folder_path) if files.endswith(".mp4")]

print("Video List:", file_names)

path_list = []
for name in file_names:
    path_list.append(os.path.join(folder_path, name))

print("Video Paths:", path_list)


# %% [markdown]
# ## Extract Frames

# %%
import cv2
import os

for video in path_list: 
    video_name = video
    print("Processing video: " + file_names[path_list.index(video)])

    cap = cv2.VideoCapture(video_name)

    time_skips = float(1000) #skip every 1 seconds. modify if need

    count = 0
    success,image = cap.read()
    while success:
        # save image
        img_name =  f'../Dataset/images/{file_names[path_list.index(video)][:-4]}/frame{count}.jpg'
        cv2.imwrite(img_name, image)

        cap.set(cv2.CAP_PROP_POS_MSEC, (count*time_skips))

        # move the time
        success,image = cap.read()
        count += 1

    # release after reading    
    cap.release()

print("Finish Processing all videos.")

# %% [markdown]
# ## Preprocessing label with image-frames

# %%
# Processing data and label in every frame
import jsonlines # read annotation from .jsonl file
import os
import fnmatch

# load images
root = "../Dataset/label"
label_files = [files for files in os.listdir(root) if files.endswith(".jsonl")]

# print("Video List:")
# print(file_names)

customerNum = []
for file in label_files: 
    customer = []
    # print(file)
    file = os.path.join(root, file)
    if os.path.exists(file):
        # print('File exists')
        with jsonlines.open(file) as reader:
            # save annotation and file name into list
            for line in reader:
                num = line['customernum']
                customer.append(num)
    else:
        raise ValueError('Invalid label file path [%s]'%file)
    customerNum.append(customer)

print(customerNum)

# Get number of frames in every video 
# for checking
folder_dict = {}
for path, dirs, files in os.walk('../Dataset/images/Original'):
    folder_name = os.path.basename(path)
    file_count = len(fnmatch.filter(files, '*.jpg'))
    folder_dict[folder_name] = file_count

print(folder_dict) # number of frames



# %% [markdown]
# # Algorithm for people detection

# %% [markdown]
# ## HOG Algorithm

# %% [markdown]
# ### detecting people Algorithm - HOG

# %%
import cv2
import matplotlib.pyplot as plt

# Initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

pred_customer_num = []
for video in path_list: 
    print("Processing video: " + file_names[path_list.index(video)])

    # Open video
    cap = cv2.VideoCapture(video)

    time_skips = float(1000) #skip every 1 seconds. modify if need

    count = 0
    success,image = cap.read()
    pred_num_people = []
    img_count = 0
    while success:
            
        # Detect people in the image
        boxes, weights = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # Draw rectangle around each person
        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Display the image using matplotlib
        img_name =  file_names[path_list.index(video)][:-4] + "/frame%d.jpg" % count
        if count % 50 == 0:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(img_name)
            plt.show()
            
        # Count the number of people detected
        num_people = len(boxes)

        pred_num_people.append(num_people)
        cap.set(cv2.CAP_PROP_POS_MSEC, (count*time_skips))

        # move the time
        success,image = cap.read()
        count += 1

    pred_customer_num.append(pred_num_people)
    # release after reading    
    cap.release()

print("Finish Processing all videos.")


# %% [markdown]
# ### Match Rate for all algorithm

# %%
# Match Rate
def match_rate(original, pred):
    match_rate = []
    for i in range(len(original)):
        match = 0
        for j in range(len(original[i])):
            if original[i][j] == pred[i][j]:
                match += 1
        rate = match / len(pred[i])
        match_rate.append(rate)
    return match_rate

# %% [markdown]
# ### Accuracy for HOG + SVM

# %%
print("Match Rate for 6 videos for HOG:")
print(match_rate(customerNum, pred_customer_num))

# %% [markdown]
# ## Improved parameter in HOG Algorithm

# %%
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

# Initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

people_list_HOG_2=[]
for video in path_list: 
    print("Processing video: " + file_names[path_list.index(video)])
    cap = cv2.VideoCapture(video)

    time_skips = float(1000) #skip every 1 seconds. modify if need

    count = 0
    success,image = cap.read()
    pred_num_people = []
    while success:
            
        # Detect people in the image
        boxes, weights = hog.detectMultiScale(image, winStride=(6, 6), padding=(7, 7), scale=1.05)

    # Apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        pick = non_max_suppression(boxes, probs=None, overlapThresh=0.65)
        # Draw rectangle around each person
        #for (x, y, w, h) in boxes:
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
        # Draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 0, 255), 2)
        
        # Display the image using matplotlib
        img_name =  file_names[path_list.index(video)][:-4] + "/frame%d.jpg" % count
        if count % 50 == 0:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(img_name)
            plt.show()
        
        # save image
        # cv2.imwrite("/Users/tracyhanwenyu/Desktop/images/frame%d.jpg" % count, image)
        
        # Count the number of people detected
        num_people = len(boxes)
        # print('Number of people detected:', num_people)
        pred_num_people.append(num_people)
        cap.set(cv2.CAP_PROP_POS_MSEC, (count*time_skips))
        # print('New frame captured: ' + str(count))

        # move the time
        success,image = cap.read()
        count += 1

    people_list_HOG_2.append(pred_num_people)
    # release after reading    
    cap.release()
print("Finish Processing all videos.")

# %%
print("Match Rate for 6 videos for HOG improved:")
print(match_rate(customerNum, people_list_HOG_2))

# %% [markdown]
# ## YOLOv5
# Using YOLOv5 model to detect people in picture

# %% [markdown]
# ### YOLOv5 Algorithm

# %%
# convert into img path list and then sort by number in file name
import os

frame_path = '../Dataset/images/Original'
img_path_list = []

for root, dirs, files in os.walk(frame_path):
    temp_list = []
    for dir_name in dirs:
        # Create the full path to the directory
        dir_path = os.path.join(root, dir_name)
        # Get the names of all .jpg files in the directory
        for file in os.listdir(dir_path):
            if file.endswith(".jpg"):
                temp_list.append(os.path.join(dir_path, file))
        
        # Add the full paths to the .jpg files to temp_list
        # img_path_list.append(temp_list)
        # print(temp_list)
        temp_list.sort(key=lambda x: int(x[len(dir_path)+6:-4]))
        img_path_list.append(temp_list)
        temp_list = []


# %%
# %cd content
# !git clone https://github.com/ultralytics/yolov5
# %cd yolov5
# %pip install -r requirements.txt

# from yolov5 import utils
# display = utils.notebook_init()

# !python detect.py --weights yolov5s.pt --img 256 --conf 0.25 --source ../../Dataset/images/Original/Communal1

# reference: https://github.com/ultralytics/yolov5/issues/36

import torch

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = model.to(device)  # Move the model to GPU
model.eval()

people_list = []
for img_path in img_path_list:
    temp_list = []
    for img in img_path:
        # # Image
        # im = '../Dataset/images/Original/Communal1/frame0.jpg'

        # Inference
        results = model(img)

        # print(results.pandas().xyxy[0])
        # results.pandas().xyxy[0]
        #      xmin    ymin    xmax   ymax  confidence  class    name
        # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
        # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
        # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
        # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

        # Filter the DataFrame to only include rows where 'name' is 'person'
        people = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'person']
        # confidence = people['confidence']
        # # Get the number of people
        num_people = len(people)
        # # Calculate the average confidence
        # average_confidence = confidence.mean()
        # print('Number of people detected:', num_people)
        temp_list.append(num_people)

        # # print('Confidence:', confidence)
        # print('Average confidence:', average_confidence)
    # print(temp_list)
    people_list.append(temp_list)

# print(people_list)
print("finish processing all videos.")


# %% [markdown]
# ### Detection Accuracy for YOLOv5 Model

# %%
print("Match rate for 6 videos for YOLOv5:")
print(match_rate(customerNum, people_list))

# %% [markdown]
# ## Faster R-CNN

# %% [markdown]
# ### R-CNN Model Algorithm

# %%
import torchvision
import cv2
import numpy as np
import torch
import torchvision.transforms as T

# Load the pre-trained Faster R-CNN model

########################### cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)  # Move the model to GPU
model.eval()

people_list_RCNN = []
# load video path for capture
for video in path_list:
    temp_list =[]
    print("Processing video:", video)
    # Load the video
    cap = cv2.VideoCapture(video)

    # Initialize frame counter
    #frame_counter = 0
    #seconds = 0
    count = 0
    time_skips = float(1000) #skip every 1 seconds. modify if need

    while True:
    #fps = cap.get(cv2.CAP_PROP_FPS)

    #while(cap.isOpened()):
        # Set the position in the video to the current second
        cap.set(cv2.CAP_PROP_POS_MSEC, count * time_skips)
        # Read the next frame
        ret, frame = cap.read()

        if not ret:
            break

        frame_model = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_model = frame_model / 255.0
        frame_model = np.transpose(frame_model, (2, 0, 1))
        frame_model = torch.from_numpy(frame_model).float().to(device) ########################### cuda
        # frame_model = torch.from_numpy(frame_model).float()

        # Apply the Faster R-CNN model to the frame
        output = model([frame_model])
        ########################### cuda
        output = [{k: v.to('cpu') for k, v in t.items()} for t in output]

        # Apply non-maximum suppression
        nms_indices = torchvision.ops.nms(output[0]['boxes'], output[0]['scores'], 0.3)
        output[0]['boxes'] = output[0]['boxes'][nms_indices]
        output[0]['labels'] = output[0]['labels'][nms_indices]

        # Filter out the detections with low confidence scores
        high_conf_indices = [i for i, score in enumerate(output[0]['scores']) if score > 0.7]
        output[0]['boxes'] = output[0]['boxes'][high_conf_indices]
        output[0]['labels'] = output[0]['labels'][high_conf_indices]

        # Count the number of people detected in the frame
        num_people = sum(1 for box, label in zip(output[0]['boxes'], output[0]['labels']) if label == 1)

        # Draw bounding boxes around the detected people
        for box, label in zip(output[0]['boxes'], output[0]['labels']):
                if label == 1:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)   

        # Count the number of people detected in the frame
        num_people = sum(1 for box, label in zip(output[0]['boxes'], output[0]['labels']) if label == 1)
            
        # Print the number of people detected in the frame along with the frame number
        print(f'Frame {count}: Number of people: {num_people}')
        temp_list.append(num_people)

        count += 1
    # Release the video capture
    cap.release()
    print(video, temp_list)
    people_list_RCNN.append(temp_list)

# Close all OpenCV windows
cv2.destroyAllWindows()

# %% [markdown]
# ### Acuuracy for faster RCNN algorithm

# %%
# print(people_list_RCNN)
print("Match Rate for 6 videos for faster RCNN:")
print(match_rate(customerNum, people_list_RCNN))

# %% [markdown]
# # Serving Time Calculation

# %% [markdown]
# ## Preprocessing data

# %%
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# Read the CSV file
df = pd.read_csv('../Dataset/label/servingTime.csv')

df_va = df[df['window'].str.startswith('VA')]
df_communal = df[df['window'].str.startswith('Communal')]
df_va = df_va.reset_index(drop=True)
# print(df_va)

# %%
# Print the DataFrame of 2 canteens
# print(df)
print(f'There are {df.isnull().any().sum()} columns in the dataset with missing values.')
print(f'The dataset of VA canteen has {df_va.shape[0]} rows and {df_va.shape[1]} columns.')
print(f'The dataset of Communal canteen has {df_communal.shape[0]} rows and {df_communal.shape[1]} columns.')
df_va.head()

# %%
df_communal.head()

# %% [markdown]
# ## Waiting time prediction for VA

# %%
df_va.describe()

# %%
# data to be plotted
mu = df_va["waitingTime"].mean()  # mean of distribution
sigma = df_va["waitingTime"].std()  # standard deviation of distribution
x = df_va["waitingTime"]
num_bins = 33
fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
   np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Customer waiting time (seconds)')
ax.set_ylabel('Probability density')
print(mu, sigma)

# %%
# mean encoding for regression output
def mean_encoder_regression(input_vector, output_vector):
    assert len(input_vector) == len(output_vector)
    numberOfRows = len(input_vector)

    temp = pd.concat([input_vector, output_vector], axis=1)
    # Compute target mean
    averages = temp.groupby(by=input_vector.name)[output_vector.name].agg(["mean", "count"])

    print(averages)
    return_vector = pd.DataFrame(0, index=np.arange(numberOfRows), columns={'feature'})


    for i in range(numberOfRows):
        return_vector.iloc[i] = averages['mean'][input_vector.iloc[i]]

    return return_vector

# %% [markdown]
# ### Prediction

# %%
workingCopyVA = df_va
workingCopyVA.drop(['serviceTime'], axis=1)
encoded_input_vector_window_va = mean_encoder_regression(workingCopyVA['window'], workingCopyVA['waitingTime'])
encoded_input_vector_window_va.columns = ['window']

# %%
X = pd.concat([pd.DataFrame(workingCopyVA['waitingPeople']), encoded_input_vector_window_va['window']], axis=1)
y = workingCopyVA['waitingTime']

X.describe()
# print(X)
# print(df_va)

# %%
print(X.shape)
print(y.shape)

# %%
from sklearn.model_selection import train_test_split

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.15, random_state=42)
print(trainX.shape, trainy.shape)
print(testX.shape, testy.shape)

# %%
def scale_input(X, means, stds):
    return (X - means) / stds

def descale_input(X, means, stds):
    return (X * stds) + means

# %%
meansX = trainX.mean(axis=0)
stdsX = trainX.std(axis=0) + 1e-10

# %%
trainX_scaled = scale_input(trainX, meansX, stdsX)
testX_scaled = scale_input(testX, meansX, stdsX)

# %%
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_absolute_error
NN = MLPRegressor(max_iter=500, activation = "relu", hidden_layer_sizes=(100,100))

NN.fit(trainX,trainy)

NN_pred = NN.predict(testX)

print("MAE for VA Canteen:   ", mean_absolute_error(testy,NN_pred))

# %%
testy_pred = NN.predict(testX)
myLength = len(testy_pred)
plt.plot(range(myLength), testy)
plt.plot(range(myLength), testy_pred)
plt.ylabel('Customer waiting time (mins)')
plt.xlabel('Client')
plt.legend(['Real', 'Predicted'], loc='upper left')

# %%
myMae = mean_absolute_error(testy, testy_pred)
print(f'The mean absolute error for VA with the neural network is {myMae} seconds.')

# %% [markdown]
# ## Waiting time prediction for Communal

# %%
df_communal.describe()

# %%
# data to be plotted
mu = df_communal["waitingTime"].mean()  # mean of distribution
sigma = df_communal["waitingTime"].std()  # standard deviation of distribution
x = df_communal["waitingTime"]
num_bins = 33
fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
   np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Customer waiting time (mins)')
ax.set_ylabel('Probability density')

# %% [markdown]
# ### Prediction

# %%
workingCopyC = df_communal
workingCopyC.drop(['serviceTime'], axis=1)
encoded_input_vector_window_c = mean_encoder_regression(workingCopyC['window'], workingCopyC['waitingTime'])
encoded_input_vector_window_c.columns = ['window']

# %%
X = pd.concat([pd.DataFrame(workingCopyC['waitingPeople']), encoded_input_vector_window_c['window']], axis=1)
y = workingCopyC['waitingTime']

X.describe()

# %%
from sklearn.model_selection import train_test_split

# print(X.shape)
# print(y.shape)
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.15, random_state=42)
print(trainX.shape, trainy.shape)
print(testX.shape, testy.shape)

# %%
meansX = trainX.mean(axis=0)
stdsX = trainX.std(axis=0) + 1e-10
trainX_scaled = scale_input(trainX, meansX, stdsX)
testX_scaled = scale_input(testX, meansX, stdsX)

# %%
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_absolute_error
NN = MLPRegressor(max_iter=500, activation = "relu", hidden_layer_sizes=(100,100))

NN.fit(trainX,trainy)

NN_pred = NN.predict(testX)

print("MAE for Communal Canteen    ", mean_absolute_error(testy,NN_pred))
testy_pred = NN.predict(testX)
myLength = len(testy_pred)
plt.plot(range(myLength), testy)
plt.plot(range(myLength), testy_pred)
plt.ylabel('Customer waiting time (mins)')
plt.xlabel('Client')
plt.legend(['Real', 'Predicted'], loc='upper left')

# %%
myMae = mean_absolute_error(testy, testy_pred)
print(f'The mean absolute error for VA with the neural network is {myMae} seconds.')


