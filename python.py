import cv2 
import matplotlib.pyplot as plt

print(cv2.__version__)

img = cv2.imread('photo.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()