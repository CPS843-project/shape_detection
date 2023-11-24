import cv2

# Read the image
image = cv2.imread('dataset/canny_img.png', cv2.IMREAD_GRAYSCALE)

# Apply contour detection
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
result_image = cv2.cvtColor(image)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Object Dimensions', result_image)
cv2.imwrite('result_image.png',result_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()