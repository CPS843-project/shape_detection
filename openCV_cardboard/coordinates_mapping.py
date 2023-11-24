import cv2
import numpy as np
import pandas as pd
# Load the image
image = cv2.imread("dataset/cardboard_1.jpg")  # Load your image

# Define your list of coordinate points
SobelXDataFrame = pd.read_csv('cardboard_sobel.csv')

# Define the BGR color for green
green_color = (0, 255, 0)

# Loop through the coordinates and change pixel values to green
for coordinate in SobelXDataFrame:
    print (coordinate)
    #image[1, 0] = green_color  # Set the pixel at the coordinate to green

# Display or save the resulting image
# cv2.imshow("Image with Green Pixels", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # To save the result
# cv2.imwrite("result.jpg", image)