# coding=utf-8
import cv2
import numpy as np
from PIL import Image


# rotate(): rotate image
# return: rotated image object
def rotate(
        img,  # image matrix
        angle  # angle of rotation
):
    height = img.shape[0]
    width = img.shape[1]

    if angle % 180 == 0:
        scale = 1
    elif angle % 90 == 0:
        scale = float(max(height, width)) / min(height, width)
    else:
        scale = np.math.sqrt(pow(height, 2) + pow(width, 2)) / min(height, width)

    # print 'scale %f\n' %scale

    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    rotateImg = cv2.warpAffine(img, rotateMat, (width, height))
    # cv2.imshow('rotateImg',rotateImg)
    # cv2.waitKey(0)

    return rotateImg  # rotated image


filename = 'grabcut_output.png'

img = cv2.imread(filename)  # 载入图像
origin = cv2.imread(filename)  # 载入图像
originh, originw = img.shape[:2]  # 获取图像的高和宽
print("图片宽度和高度分别是{}", originh, originw)
# cv2.imshow("Origin", img)  # 显示原始图像

blured = cv2.blur(img, (5, 5))  # 进行滤波去掉噪声
# cv2.imshow("Blur", blured)  # 显示低通滤波后的图像

mask = np.zeros((originh + 2, originw + 2), np.uint8)  # 掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘
# 进行泛洪填充
cv2.floodFill(blured, mask, (originw - 1, originh - 1), (255, 255, 255), (2, 2, 2), (3, 3, 3), 8)
# cv2.imshow("floodfill", blured)

# 得到灰度图
gray = cv2.cvtColor(blured, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray)

# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
# 开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("closed", closed)

# 求二值图
ret, binary = cv2.threshold(opened, 250, 255, cv2.THRESH_BINARY)
# cv2.imshow("binary", binary)

# 找到轮廓
_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓

cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
# 绘制结果
# cv2.imshow("result", img)

for i in range(0, len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    cv2.rectangle(origin, (x, y), (x + w, y + h), (153, 153, 0), 5)
    # newimage = origin[y + 10:y + h - 5, x + 10:x + w - 5]  # 先用y确定高，再用x确定宽
    newimage = origin[y:y + h, x+10:x-10 + w]  # 先用y确定高，再用x确定宽
    # newimage = rotate(newimage, 1)
    nh, nw = newimage.shape[:2]  # 获取图像的高和宽
    if nw < originw:
        # nrootdir = "E:/color/pic/_resource/"
        nrootdir = "./_resource/"
        resultrootdir = "./segment/"
        if not cv2.os.path.isdir(nrootdir):
            cv2.os.makedirs(nrootdir)
        if not cv2.os.path.isdir(resultrootdir):
            cv2.os.makedirs(resultrootdir)
        cv2.imwrite(nrootdir + "_first" + str(0) + ".jpg", newimage)
        print(i)
        cv2.imshow("newimage", newimage)
        print("图片宽度和高度分别是{}", nh, nw)

        # im = Image.open("E:/color/pic/_resource/_first0.jpg")
        im = Image.open("./_resource/_first0.jpg")
        xx = 33
        yy = 3
        x = nw // xx
        y = nh // yy
        for j in range(yy):
            for i in range(xx):
                left = i * x
                up = y * j
                right = left + x
                low = up + y
                region = im.crop((left, up, right, low))
                # print((left, up, right, low))
                # if i % 3 == 1 and j == 1:
                if j == 1:
                    # if i % 3 == 1:
                    temp = str(i) + str(j)
                    # region.save("E:/color/segment/" + temp + ".png")
                    region.save("./segment/" + temp + ".png")

cv2.waitKey(0)
cv2.destroyAllWindows()
