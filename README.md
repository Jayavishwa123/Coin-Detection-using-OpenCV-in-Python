# Coin-Detection-using-OpenCV-in-Python
# Name:Jaya Vishwa S
# Reg:212224230105
# AIM
To detect and count the number of coins in an image using OpenCV by applying grayscale conversion, thresholding, morphological operations, blob detection, and contour detection.

# ALGORITHM
Import the required OpenCV, NumPy, and Matplotlib libraries.

Read and display the input coin image.

Convert the input image from BGR to grayscale.

Apply thresholding to separate the coins from the background.

Perform morphological Opening to remove small noise.

Perform morphological Closing to fill small gaps and improve the coin regions.

Apply Blob Detection to detect coin-like objects and count them.

Apply Contour Detection to identify the boundaries of the coins and count them.

Display the detected coins and their boundaries.

Compare the results obtained using Blob Detection and Contour Detection.

# PROGRAM
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("coin.jpg")

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(8, 6))
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(8, 6))
plt.imshow(thresh, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
plt.show()


kernel = np.ones((5, 5), np.uint8)

# Opening - removes small noise
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Closing - fills small holes
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(opening, cmap="gray")
plt.title("After Opening")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(closing, cmap="gray")
plt.title("After Closing")
plt.axis("off")

plt.show()



params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 500
params.maxArea = 50000

params.filterByCircularity = True
params.minCircularity = 0.7

params.filterByConvexity = True
params.minConvexity = 0.8

params.filterByInertia = True
params.minInertiaRatio = 0.5

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(closing)

blob_image = cv2.drawKeypoints(
    cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR),
    keypoints,
    None,
    (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(blob_image, cv2.COLOR_BGR2RGB))
plt.title("Blob Detection")
plt.axis("off")
plt.show()

print("Number of coins detected using Blob Detection:", len(keypoints))


contours, hierarchy = cv2.findContours(
    closing,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Filter small contours
coin_contours = []

for contour in contours:
    area = cv2.contourArea(contour)
    
    if area > 500:
        coin_contours.append(contour)

# Draw contours
contour_image = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)

cv2.drawContours(
    contour_image,
    coin_contours,
    -1,
    (0, 255, 0),
    2
)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title("Contour Detection")
plt.axis("off")
plt.show()

print("Number of coins detected using Contours:", len(coin_contours))



print("===== RESULTS =====")
print("Coins detected using Blob Detection    :", len(keypoints))
print("Coins detected using Contour Detection :", len(coin_contours))
```
<img width="712" height="632" alt="645435509-1f1884d5-1c02-4aed-a79a-e03717e4d99b" src="https://github.com/user-attachments/assets/3fa3be72-f865-4115-81f6-ffb5924c1d20" />

<img width="822" height="607" alt="645436172-13ae3c5a-154d-4e38-b264-d9390ee0d97d" src="https://github.com/user-attachments/assets/3ef1094a-5cf4-4d17-939b-62095b7617d6" />

<img width="931" height="622" alt="645436397-95ee44bf-7abe-4e02-83e6-114b9a4fcc9a" src="https://github.com/user-attachments/assets/bc5b39f6-cf64-42e2-88c6-2430ef78e545" />

<img width="1228" height="543" alt="645436591-a985d819-a22f-478c-ac98-0e0fd6b68ea2" src="https://github.com/user-attachments/assets/771d5824-38ce-449a-8006-4555a3801532" />

<img width="832" height="655" alt="645436713-f7bf27c7-195d-421c-88d2-49c3db0ca26b" src="https://github.com/user-attachments/assets/97061009-bff5-4670-8c09-4bb2ba4fc4d4" />

<img width="666" height="642" alt="645437088-32489e3a-0353-4236-a88d-034a9ecb7e5e" src="https://github.com/user-attachments/assets/494030b3-2c7d-477e-a146-41194122f23d" />

# RESULT
The coins in the input image were successfully detected and counted using both Blob Detection and Contour Detection. The thresholding and morphological operations helped to improve the image quality and separate the coins from the background.

The number of coins detected using Blob Detection was obtained from the detected keypoints, while the number of coins detected using Contour Detection was obtained from the filtered contours.
