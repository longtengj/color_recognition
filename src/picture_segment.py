# coding=utf-8
# 切割图片成正方形，并保存在相应文件夹下
import math
import os

import cv2
from PIL import Image

# 导入相关的库
from PIL import Image

# 打开一张图
img = Image.open('E:\color\pic\_resource\_first1.jpg')
# 图片尺寸
img_size = img.size
h = img_size[1]  # 图片高度
w = img_size[0]  # 图片宽度

x = w
y = h
w = w
h = h

# 开始截取
region = img.crop((x, y, x + w, y + h))
# 保存图片
region.save("test.jpg")


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
        scale = math.sqrt(pow(height, 2) + pow(width, 2)) / min(height, width)

    # print 'scale %f\n' %scale

    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    rotateImg = cv2.warpAffine(img, rotateMat, (width, height))
    # cv2.imshow('rotateImg',rotateImg)
    # cv2.waitKey(0)

    return rotateImg  # rotated image


# 切割图片的函数
def splitimage(src, rownum, colnum, dstpath):
    img = Image.open(src)
    w, h = img.size
    if rownum <= h and colnum <= w:
        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('开始处理图片切割, 请稍候...')

        s = os.path.split(src)
        if dstpath == '':
            dstpath = s[0]
        fn = s[1].split('.')
        basename = fn[0]
        ext = fn[-1]

        num = 0
        rowheight = h // rownum
        colwidth = w // colnum
        for r in range(rownum):
            for c in range(colnum):
                box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
                img.crop(box).save(os.path.join(dstpath, basename + '_' + str(num) + '.' + ext), ext)
                num = num + 1

        print('图片切割完毕，共生成 %s 张小图片。' % num)
    else:
        print('不合法的行列切割参数！')


# 创建文件夹的函数
def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False


folder = 'E:\color\pic\_resource\blues_1'  # 存放图片的文件夹
path = os.listdir(folder)
print(path)

for each_bmp in path:  # 遍历，来进行批量操作
    first_name, second_name = os.path.splitext(each_bmp)
    each_bmp = os.path.join(folder, each_bmp)
    src = each_bmp
    print(src)
    print(first_name)
    # 定义要创建的目录
    mkpath = "/Users/hjy/Desktop/img_file1/" + first_name
    # 调用函数
    mkdir(mkpath)
    if os.path.isfile(src):
        dstpath = mkpath
        if (dstpath == '') or os.path.exists(dstpath):
            row = int(1)  # 切割行数
            col = int(10)  # 切割列数
            if row > 0 and col > 0:
                splitimage(src, row, col, dstpath)
            else:
                print('无效的行列切割参数！')
        else:
            print('图片输出目录 %s 不存在！' % dstpath)
    else:
        print('图片文件 %s 不存在！' % src)
