from pathlib import Path
from copy import deepcopy

from cv2 import imread, imshow, imwrite
from numpy import reshape
from sklearn.cluster import KMeans

pic_path = Path('kmeans.jpg')
img = imread(pic_path.as_posix())

pixel = reshape(img, (img.shape[0] * img.shape[1], 3))
print(pixel.shape)
pixel_new = deepcopy(pixel)

print(img.shape)

model = KMeans(n_clusters=5)
# 注意，KMeans的fit_predict方法入参必须是两个dimension，d1是各个样本，d2是每个样本的features, 返回值是每个样本分到第几类的列表
labels = model.fit_predict(pixel)
# cluster_centers_返回各类中心点的列表
palette = model.cluster_centers_

print(labels)

for i in range(len(pixel)):
    pixel_new[i, :] = palette[labels[i]]

imwrite('zipped.jpg', reshape(pixel_new, (img.shape[0], img.shape[1], 3)))
