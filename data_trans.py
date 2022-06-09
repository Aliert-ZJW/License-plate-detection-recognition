import cv2
import os
import json


# path = "/public/ccpd_green/test/"
path = "/data/zjw/data/yoluv5_porject_test/json/"

for filename in os.listdir(path):
    with open(path+filename, 'r', encoding='utf-8')as fp:
        json_data = json.load(fp)
        shape = json_data['shapes']
        points = shape[0]['points']
        lx, ly, rx, ry = points[0][0], points[0][1], points[1][0], points[1][1]
        # list1 = filename.split("-", 3)  # 第一次分割，以减号'-'做分割
        # subname = list1[2]
        # lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
        # lx, ly = lt.split("&", 1)
        # rx, ry = rb.split("&", 1)
        print(lx, ly, rx, ry)
        file, name = os.path.splitext(filename)
        width = int(rx) - int(lx)
        height = int(ry) - int(ly)  # bounding box的宽和高
        cx = float(lx) + width / 2
        cy = float(ly) + height / 2  # bounding box中心点
        print(path + file + '.jpg')
        img = cv2.imread('/data/zjw/data/yoluv5_porject_test/new_car/' + file + '.jpg')
        width = width / img.shape[1]
        height = height / img.shape[0]
        cx = cx / img.shape[1]
        cy = cy / img.shape[0]

        txtname = filename.split(".", 1)
        txtfile = '/data/zjw/data/yoluv5_porject_test/lable/' + txtname[0] + ".txt"
        # 绿牌是第0类，蓝牌是第1类
        with open(txtfile, "w") as f:
            f.write(str(0) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))

        print(filename)

# import os
# path_car = "/data/zjw_plus/data/real_json/images/train/"
# path_json = '/data/zjw_plus/data/real_json/images/label/'
# car_names = os.listdir(path_car)
# json_names = os.listdir(path_json)
# for car in car_names:
#     car_, txt = os.path.splitext(car)
#     car_ = car_ + '.json'
#     # print(car_)
#     if car_ in json_names:
#         continue
#     else:
#         print(car_)

# import os
#
# path = '/data/zjw_plus/data/new_car/'
#
# # 获取该目录下所有文件，存入列表中
# fileList = os.listdir(path)
#
# n = 824
# for i in fileList:
#     # 设置旧文件名（就是路径+文件名）
#     oldname = path + os.sep + i  # os.sep添加系统分隔符
#
#     # 设置新文件名
#     newname = path + os.sep + str(n) + '.jpg'
#
#     os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
#     print(oldname, '======>', newname)
#
#     n += 1
