import cv2
import numpy as np

img = cv2.imread("../dataset/flower.jpg")
cv2.imshow("flower", img)

# ***************************** img ==> R,G,B *****************************
[B, G, R] = cv2.split(img)
hstack_img = np.hstack([B, G, R])  # 横向堆叠显示
hstack_img = cv2.resize(hstack_img, (0, 0), None, fx=0.5, fy=0.5)  # 缩小显示
cv2.imshow("B-G-R channel", hstack_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ***************************** R,G,B ==> gray *****************************
# 方法一：以灰度图形式读入
gray1 = cv2.imread("../dataset/flower.jpg", 0)
cv2.imshow("gray1", gray1)


# 方法二：API转换
img = cv2.imread("../dataset/flower.jpg")
gray2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray2", gray2)


# 方法二：心理学公式
gray3 = R * 0.299 + G * 0.587 + B * 0.114
gray3 = gray3.astype(np.uint8)  # 一定要转成整形
cv2.imshow("gray3", gray3)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ***************************** R,G,B ==> h,s,v *****************************
img = cv2.imread("../dataset/flower.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
[H, S, V] = cv2.split(img_hsv)
hstack_hsv = np.hstack([H, S, V])  # 横向堆叠显示
hstack_hsv = cv2.resize(hstack_hsv, (0, 0), None, fx=0.5, fy=0.5)  # 缩小显示
cv2.imshow("h-s-v channel", hstack_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ***************************** R,G,B ==> l,a,b *****************************
img = cv2.imread("../dataset/flower.jpg")
img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
[l, a, b] = cv2.split(img_lab)
hstack_lab = np.hstack([l, a, b])  # 横向堆叠显示
hstack_lab = cv2.resize(hstack_lab, (0, 0), None, fx=0.5, fy=0.5)  # 缩小显示
cv2.imshow("l-a-b channel", hstack_lab)
cv2.waitKey(0)
cv2.destroyAllWindows()






