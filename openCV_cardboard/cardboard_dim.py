import time
import numpy as np
import pandas as pd
import cv2
from shapeDetector import ShapeDetector
import imutils
from matplotlib import pyplot as plt

# Harris Crner Detection


# Canny edge detection
# img = cv2.imread('dataset/cardboard_1.jpg', cv2.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# img = cv2.GaussianBlur(img, (1, 1), 0)
# edges = cv2.Canny(img,100,200)
# row_indexes, col_indexes = np.nonzero(edges)
# df = pd.DataFrame({'row_indexes':row_indexes, 'col_indexes':col_indexes})
# df.to_csv('canny_line_coordinates.csv')
# print(row_indexes)
# print(type(row_indexes))
# print(len(row_indexes))
# print(col_indexes)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()


#  sobel edge detection
colored_img = cv2.imread('dataset/cardboardhires.jpg')
original_img = cv2.imread('dataset/cardboardhires.jpg', cv2.IMREAD_GRAYSCALE)
gaussian_img = cv2.GaussianBlur(original_img, (5, 5), 0)
# cv2.imshow('gaussian_img', gaussian_img)
cv2.imwrite('gaussian_img.png',gaussian_img)

canny_img = cv2.Canny(gaussian_img,100,200)
# cv2.imshow('canny_img', canny_img)
cv2.imwrite('canny_img.png',canny_img)

laplacian_img = cv2.Laplacian(canny_img,cv2.CV_64F)
# cv2.imshow('laplacian_img', laplacian_img)
cv2.imwrite('laplacian_img.png',laplacian_img)

sobelx = cv2.Sobel(canny_img, -1,1,0,ksize=3)
sobely = cv2.Sobel(canny_img, -1,0,1,ksize=3)
# plt.subplot(1,2,1),plt.imshow(sobelx, cmap="hot")
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,2,2),plt.imshow(sobely, cmap="hot")
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.show()

edge_coordinates = np.argwhere(sobelx > 250)
print (len(edge_coordinates))
# for x in edge_coordinates:
#     print (x)

sobleXDataFrame = pd.DataFrame(sobelx)

sobleXDataFrame.to_csv('cardboard_sobelx.csv')

sobleXDataFrame = pd.DataFrame(edge_coordinates)

sobleXDataFrame.to_csv('cardboard_sobel.csv')

# output_image = cv2.cvtColor(sobelx, cv2.COLOR_GRAY2BGR)  # Convert the Sobel output to BGR

# Define the BGR color for green
green_color = (0, 255, 0)
# Loop through the coordinates and change pixel values to green
for coords in edge_coordinates:
    cv2.circle(colored_img, [coords[1],coords[0]], 1, green_color, -1)

# cv2.imshow("Sobel Edges with Coordinate Points", img)
# plt.figure(figsize=(10,10))
plt.imshow(colored_img)
# cv2.imshow('orig image', colored_img)
# plt.show()
# cv2.waitKey(0)

# Apply HoughLinesP method to 
# to directly obtain line end points
lines_list =[]
lines = cv2.HoughLinesP(
            sobelx+sobely, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=150, # Min number of votes for valid line
            minLineLength=25, # Min allowed length of line
            maxLineGap=250 # Max allowed gap between line for joining them
            )
lined_img = cv2.imread('dataset/cardboardhires.jpg')
print (lines)
print(len(lines)) 
# Iterate over points
x = 0
for points in lines:
      # Extracted points nested in the list
    x1,y1,x2,y2=points[0]
    # Draw the lines joing the points
    # On the original image
    if (x%2 == 0): 
        cv2.line(lined_img,(x1,y1),(x2,y2),(0,0,255),5)
    else:
        cv2.line(lined_img,(x1,y1),(x2,y2),(0,255,0),5)
    # Maintain a simples lookup list for points
    lines_list.append([(x1,y1),(x2,y2)])
    x+=1
print(lined_img.shape)

lined_image = np.zeros(lined_img.shape, dtype = np.uint8)
# lined_image.fill(255)
for points in lines:
      # Extracted points nested in the list
    x1,y1,x2,y2=points[0]
    # Draw the lines joing the points
    # On the original image
    if (x%2 == 0): 
        cv2.line(lined_image,(x1,y1),(x2,y2),(255,255,255),5)
    else:
        cv2.line(lined_image,(x1,y1),(x2,y2),(255,255,255),5)
    # Maintain a simples lookup list for points
    lines_list.append([(x1,y1),(x2,y2)])
    x+=1
# Save the result image
plt.imshow(lined_image)
plt.show()
cv2.imwrite('lined_image.png',lined_image)


# identify and complete rectangular shapes

# img = cv2.imread('lined_image.png')
# kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(img,kernel,iterations = 10)
# dilate = cv2.dilate(erosion,kernel,iterations = 10)

# cv2.bitwise_not ( dilate, dilate )

# image = dilate
# resized = imutils.resize(image, width=300)
# ratio = image.shape[0] / float(resized.shape[0])

# # convert the resized image to grayscale, blur it slightly,
# # and threshold it
# gray = cv2.cvtColor(lined_image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (1, 1), 0)
# thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)

#thresh = dilate
# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()
print("cnts")
print(cnts)
# loop over the contours
for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    print(c)
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
exit()

















# Apply contour detection
contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Measure the dimensions of detecpngted objects
for contour in contours:
    # Calculate the bounding box of the object
    x, y, w, h = cv2.boundingRect(contour)
    
    # Width and height of the object
    width = w
    height = h
    
    # You can process or display the width and height as needed
    print(f"Object Width: {width}, Height: {height}")

# Display the image with bounding boxes (optional)
result_image = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Object Dimensions', result_image)
cv2.imwrite('result_image.png',result_image)


# path to input image specified and  
# image is loaded with imread command 
image = cv2.imread('dataset/cardboardhires.jpg') 
image = cv2.imread('dataset/cardboard_1.jpg') 
  
# convert the input image into 
# grayscale color space 
operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# modify the data type 
# setting to 32-bit floating point 
operatedImage = np.float32(operatedImage) 
  
# apply the cv2.cornerHarris method 
# to detect the corners with appropriate 
# values as input parameters 
dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07) 
  
# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
  
# Reverting back to the original image, 
# with optimal threshold value 
image[dest > 0.01 * dest.max()]=[0, 0, 255] 
  
# the window showing output image with corners 
cv2.imshow('Image with Borders', image) 
cv2.imwrite('harris_corner.png',image)

  
# De-allocate any associated memory usage  
if cv2.waitKey(0) & 0xff == 27: 
    cv2.destroyAllWindows() 
