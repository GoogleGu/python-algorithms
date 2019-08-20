import cv2

from util import get_data_file_path


pic_file = get_data_file_path('soccer.jpg')


"""
MSER_create()方法可以接受多个参数：
_delta: Any = None,
_min_area: 最小可接受区域面积,
_max_area: 最大可接受区域面积,
_max_variation: 设置此值将与子区域面积相似的母区域移除,
_min_diversity: ,
_max_evolution: Any = None,
_area_threshold: Any = None,
_min_margin: Any = None,
_edge_blur_size: Any = Non
"""
detector = cv2.MSER_create(_max_variation=2, _min_area=500)
img = cv2.imread(pic_file, 0)

"""
detectRegions()方法返回两个对象，
第一个是探测出来的区域点集组成的列表，列表中每一个元素是一个roi区域，
第二个是标记roi区域的矩形框列表，列表中每个元素代表一个roi区域矩形框。
"""
regions, _ = detector.detectRegions(img)
boxes = [cv2.boundingRect(points) for points in regions]

# 可视化
for x, y, w, h in boxes:
    cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=1)
cv2.imwrite('after_mser.jpg', img)

"""
可以看到有很多的重叠区域，如果要对产生的区域进行优化选择去重叠的话，一般要通过Non Maximum Suppression（NMS）来选出重叠区域中最优的区域。
"""