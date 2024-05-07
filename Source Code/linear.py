# import json
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from math import sqrt

# # Load the data from the JSONL file
# data = []
# with open('../Dataset/label/Communal1.jsonl', 'r') as f:
#     for line in f:
#         data.append(json.loads(line))

# # Prepare data for regression
# X = []
# y = []
# for i, frame in enumerate(data):
#     X.append([i, frame['customernum']])  # features: frame number, customer number
#     y.append(i)  # target: frame number (as a proxy for waiting time)

# X = np.array(X)
# y = np.array(y)

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train a linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Calculate the mean squared error of the predictions
# mse = mean_squared_error(y_test, y_pred)
# print('Mean Squared Error:', mse)

# print("Intercept:", model.intercept_)
# print("Coefficient:", model.coef_)


# # Assume y_test are the true values and y_pred are the predicted values
# # y_test = ...
# # y_pred = model.predict(X_test)

# # Calculate metrics
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = sqrt(mse)  # or mse**(0.5)  
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Absolute Error (MAE): {mae}')
# print(f'Mean Squared Error (MSE): {mse}')
# print(f'Root Mean Squared Error (RMSE): {rmse}')
# print(f'R-squared (R²): {r2}')




# # with the order of name of the image(frame1 to frame100)

# import os

# frame_path = '../Dataset/images/Original'
# img_path_list = []

# for root, dirs, files in os.walk(frame_path):
#     temp_list = []
#     for dir_name in dirs:
#         # Create the full path to the directory
#         dir_path = os.path.join(root, dir_name)
#         # Get the names of all .jpg files in the directory
#         jpg_files = [file for file in os.listdir(dir_path) if file.endswith(".jpg")]
#         # Sort the file names based on the numeric part (assuming the format "frameX.jpg")
#         jpg_files.sort(key=lambda x: int(x[5:-4]))
#         # Add the full paths to the .jpg files to temp_list
#         temp_list.extend([os.path.join(dir_path, file) for file in jpg_files])
        
#     img_path_list.append(temp_list)

# print(img_path_list[4])