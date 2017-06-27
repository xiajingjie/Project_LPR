# -*- coding: utf-8 -*-
import cv2                                #加载opencv模块
import numpy as np                        #加载numpy模块
from matplotlib import pyplot as plt      #加载matplotlib模块
from LPR import myopencv                  #加载自定义图像处理模块
from LPR import BPnn                      #加载BP神经网络类
from LPR import data
import time                               #加载计时模块
img = cv2.imread("picture/001.bmp") #读入图像
rows,cols=img.shape[:2]                                                  #获取行数列数
start = time.clock()                                                     #标记识别开始时间
img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                          #转换为灰度图
imgstretch = myopencv.stretch(100,150,rows,cols,img2gray)                #灰度拉伸
blur = cv2.GaussianBlur(imgstretch,(5,5),0)                              #高斯滤波
edges = cv2.Canny(blur, 150, 200)                                        #canny算子边缘检测
posit_x=list(myopencv.position_x(edges,20,15,15,10))                     #车牌上下边界确定
posit_y=myopencv.position_y(edges,posit_x,5,15)                          #车牌左右边界确定
image,plate = myopencv.adjust(img,posit_x,posit_y)                       #车牌位置确定 输出车牌原图、二值化图像
img1,img2,img3,img4,img5,img6,img7 = myopencv.split(plate,img,posit_x,posit_y) #车牌字符分割
###################################################识别部分##############
nn = BPnn.BPNeuralNetwork()
nn.setup(128,15,5)                                                      #初始化神经网络：24个输入神经元，15个隐含层，5个输出神经元
###################################################训练部分##############
# L = []                                                                   #盛放样本数据
# for i in range(1,121):                                                  #训练样本读取
#     str1 ="charSamples/"
#     imageName =  str1+str(i)+'.png'
#     img = cv2.imread(imageName)  # 读入图像
#     L.append(myopencv.feature_ext(img))                                 #车牌字符特征提取
# nn.train(L, data.L, 15000, 0.05, 0.1)
# print(nn.input_weights)                                                 #训练完成后打印权值，保存在data.py
# print(nn.output_weights)
############################################################################
nn.input_weights = data.L1                                              #载入训练好的权值
nn.output_weights = data.L2
data.judge(nn.predict(myopencv.feature_ext(img2)))                       #识别车牌字符
data.judge(nn.predict(myopencv.feature_ext(img3)))
data.judge(nn.predict(myopencv.feature_ext(img4)))
data.judge(nn.predict(myopencv.feature_ext(img5)))
data.judge(nn.predict(myopencv.feature_ext(img6)))
data.judge(nn.predict(myopencv.feature_ext(img7)))
end = time.clock()                                                      #标记识别结束时间
print('\n'+'总用时:'+str(end-start)+'秒')
###################################################绘制车牌图像##############
cv2.namedWindow("Image")                                               #命名图像
while(1):
    cv2.imshow("Image",image)  #显示图像
    k=cv2.waitKey(0)             #等待键盘输入
    if k== 27:
        break
cv2.destroyAllWindows()    #按Esc关闭窗口
