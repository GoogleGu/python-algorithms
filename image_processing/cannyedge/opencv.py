import cv2
from matplotlib import pyplot as plt

from util import get_data_file_path

pic_file = get_data_file_path('soccer.jpg')
img = cv2.imread(pic_file, 0)

"""
Canny()方法可以接受多个参数，常用的为：
threshold1: 高阈值
threshold2: 低阈值
apertureSize: Sobel算子的大小
"""
edges = cv2.Canny(img, 100, 200)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.xticks([]), plt.yticks([])
plt.show()
