# -*- coding: utf-8 -*-
import cv2                                #加载opencv模块
import numpy as np                        #加载numpy模块
from matplotlib import pyplot as plt      #加载matplotlib模块

def stretch(a,b,rows,cols,image):                                 # 灰度拉伸
    for i in range(rows):
        for j in range(cols):
            if image[i, j] < a:                                   #灰度小于a的置零
                image[i, j] = 0
            elif image[i, j] > b:
                image[i, j] = 255                                 #灰度大于a的置255
            else:
                image[i, j] = 255 / (b - a) * (image[i, j] - a)   #其余部分线性化
    return image

def position_x(image,nx,dx,ny,dy):                                   #车牌定位 求取上下边缘
    ##############################第一部分：筛选跳变数合适的行#################################
    for n in range(20):                                              #第一部分：筛选跳变数合适的行
        L = []                                                       #扫描每幅时重置全图跳变点信息
        for i in range(int(image.shape[0]*4/9),image.shape[0]):     #由于车牌一般位于车辆中下部，所以从图像4/9部分开始扫描
            x1 = 0  # 跳变起始点                                    #扫描每行时重置跳变位置信息
            x2 = 0  # 跳变结束点
            s1 = 0  # 连续跳变个数
            for j in range(image.shape[1]-1):
                if image[i, j+1] != image[i, j]:
                    if s1==0:                                 #如果之前没有跳变点，从头开始
                        x1 = j
                        x2 = j
                        s1+=1
                    elif s1!=0:                               #如果之前存在跳变点，判断这次的跳变点与之前的间距
                        if j - x2 <=dx:                       #小于最大允许跳变间距dx，记录
                            x2 = j
                            s1+=1
                        elif j - x2 >dx and s1 < nx:         #大于最大允许跳变间距且长度不足，舍弃之前的跳变点
                            x1 = j
                            x2 = j
                            s1 = 1
                        elif j - x2 >dx and s1 > nx:        #大于每行跳变间距且长度足够，记录
                            L.append((i,x1,x2))
                            x1 = 0
                            x2 = 0
                            s1 = 0

        if len(L) >10 or nx<3:                                #阈值过高则减小1并重新扫描
            break
        else:
            nx-=1
    for n in range(20):
        l1 = 0   #连续行起始行
        l2 = 0   #连续行结束行
        s2 = 0   #连续行个数
        L2 = []
##############第二部分：选出可能是车牌位置的连续行位置，存入L2数组###########################333
        for i in range(len(L)-1):
            if i<len(L)-2:
                if L[i+1][0] - L[i][0] <= dy:                   #查看规定最大允许间隔下的连续行
                    if s2 == 0:                                 #如果之前没有跳变点，从头开始
                        l1 = i
                        l2 = i+1
                        s2 += 1
                    elif s2 != 0:                               #如果之前存在跳变点，记录
                        l2 = i+1
                        s2 += 1
                elif L[i+1][0] - L[i][0] > 3 and s2 >= ny:      #大于规定最大允许间隔，说明进入新的连续部分，于是记录之前的位置
                    L2.append((l1,l2))
                    s2 = 0
                else:
                    s2 = 0
            else:
                if L[i+1][0] - L[i][0] <= dy and s2 >= ny:      #程序最后一次循环结束时记录连续行信息
                    l2 = i + 1
                    L2.append((l1,l2))
                elif L[i+1][0] - L[i][0] > dy and s2 >= ny:
                    L2.append((l1,l2))
        if len(L2)>0 or ny<5:
            break
        else:
            ny-=1                                               #如果没有符合条件的连续行，则减小阈值重新搜索

###########################在连续行里记录粗略的左右边界######################################
    L3 = []                                                     #记录行筛选结果
    L4 = []                                                     #记录最终的车牌图像位置
    for i in range(len(L2)):
        bL = [] #储存车牌左边界
        bR = [] #储存车牌右边界
        for j in range(L2[i][0],L2[i][1]+1):
            bL.append(L[j][1])
            bR.append(L[j][2])
            L3.append((L[j][0],L[j][1],L[j][2]))
    L4=(L3[0][0],L3[-1][0],min(bL),max(bR))    #记录最终的车牌图像位置,取所有连续行中最宽的位置
    return L4

def position_y(image,Lp,thr,dx):                                #垂直投影计算左右边界位置
    #############################第一部分：求车牌大致位置的垂直投影#####################
    L = []
    sumy = 0
    img = image[Lp[0]:Lp[1],Lp[2]:Lp[3]]
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            if img[i,j]==255:                                    #将一列中的白点个数累加
                sumy += 1
        L.append(sumy)
        sumy = 0
    # plt.figure("垂直投影")
    # plt.bar(np.array(range(Lp[2],Lp[3])),np.array(L),alpha = 1, color = 'r')
    # plt.show()
    for i in range(len(L)):                                     #将每列白点数大于规定阈值的置一
        if L[i] >= thr:
            L[i] = 1
        else:                                                   #将每列白点数小于规定阈值的置零
            L[i] = 0
    # plt.figure("垂直投影")
    # plt.bar(np.array(range(Lp[2],Lp[3])), np.array(L), alpha=1, color='r')
    # plt.show()
    #############################第二部分：求车牌宽度#####################
    for n in range(20):
        s1 = 0
        x1 = 0
        x2 = 0
        L1 = []
        L2 = []
        for i in range(len(L)):
            if i<len(L)-1:
                if L[i] == 1:                               #寻找置一的位置，记为跳变点
                    if s1 == 0:                             #之前没有跳变点从头开始
                        x1 = i
                        x2 = i
                        s1 += 1
                    elif s1 != 0:                           #之前有跳变点
                        if i - x2 <= dx:                    #两跳变点间距小于最大允许跳变间距dx，更新x2长度
                            x2 = i
                            s1 += 1
                        elif i - x2 > dx:                   #两跳变点间距大于最大允许跳变间距，说明跳变距离结束，记录
                            L1.append((x1+Lp[2], x2+Lp[2]))
                            x1 = 0
                            x2 = 0
                            s1 = 0
            else:                                           #判断最后一次循环
                if i - x2 <= dx:                            # 小于最大允许跳变间距dx
                    x2 = i
                    L1.append((x1+Lp[2], x2+Lp[2]))
                elif i - x2 > dx:                           # 大于最大允许跳变间距
                    L1.append((x1+Lp[2], x2+Lp[2]))

        if len(L1) ==1:                                     #如果结果只有一组宽度，直接存为左右边界
            L2=list(L1[0])
        elif len(L1) > 1:                                   #如果结果只有一组宽度，取最大值，存为左右边界
            temp = 0
            for j in range(len(L1)):
                if L1[j][1]-L1[j][0]>L1[temp][1] - L1[temp][0]:
                    temp = j
            L2=list(L1[temp])
#############################第三部分：根据车牌长宽比判断结果#####################
        r = (L2[1]-L2[0])/(Lp[1]-Lp[0])
        if 4.0<=r<=6.5 or n>15:                             #长宽比满足要求，结束循环
            break
        elif r > 6.5 and n<=15:                             #长宽比大于阈值，将限定的最大允许跳变间距缩小，进入下次循环
            dx-=1
        elif r < 6.5 and n<=15:                             #长宽比小于阈值，将限定的最大允许跳变间距增大，进入下次循环
            dx += 1

    return L2

def adjust(image,posit_x,posit_y):                                              #车牌调整和二值化
    posit_x[0] = posit_x[0] - 2                                                 #边界拓展
    posit_x[1] = posit_x[1] + 2
    posit_y[0] = posit_y[0] - 2
    posit_y[1] = posit_y[1] + 2
    img = image[posit_x[0]:posit_x[1],posit_y[0]:posit_y[1]]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                #取灰度图
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,2)   #自适应阈值的二值化
    median = cv2.medianBlur(thresh, 3)                                          #中值滤波去除噪声
    return img,median

def split(image,img,posit_x,posit_y):                                                   #连通域法分割字符
    res=[]
    flag = np.zeros((image.shape[0],image.shape[1]),np.uint8)                           #连通域编号数组
    N = 0                                                                               #连通域编号
    overlap = []                                                                        #重复区域编号数组
    for i in range(image.shape[0]):
        if i == 0:                                                                     #第一行
            for j in range(image.shape[1]):
                if j == 0:                                                             #第一个元素
                    if image[i, j] == 255 and flag[i, j] == 0:                         #白点且未标记
                        if flag[i+1,j]==0 and flag[i+1,j+1]==0 and flag[i,j+1]==0:     #如果邻域都未标记，则为起始点
                            N+=1                                                        #生成新区域标号
                            flag[i, j] = N
                            flag[i+1,j] = flag[i, j] if image[i+1,j]==255 else 0                   #邻域是白点则划为同一区域
                            flag[i+1,j+1] = flag[i, j] if image[i+1,j+1] == 255 else 0
                            flag[i,j+1] = flag[i, j] if image[i,j+1] == 255 else 0
                elif j == image.shape[1]-1:                                                         #第一行最后一个元素
                    if image[i, j] == 255 and flag[i, j] == 0:                                      #白点且未标记
                        if flag[i,j-1] == 0 and flag[i+1,j-1] == 0 and flag[i+1,j] == 0:            #如果邻域都未标记，则为起始点
                            N += 1                                                                   #生成新区域标号
                            flag[i, j] = N
                            flag[i,j-1] = flag[i, j] if image[i,j-1] == 255 else 0                  #邻域是白点则划为同一区域
                            flag[i+1,j-1] = flag[i, j] if image[i+1,j-1] == 255 else 0
                            flag[i+1,j] = flag[i, j] if image[i+1,j] == 255 else 0
                    elif image[i, j] == 255 and flag[i, j] != 0:                                    #白点且已标记
                        flag[i, j - 1] = flag[i, j] if image[i, j - 1] == 255 else 0                #邻域是白点则划为同一区域
                        flag[i + 1, j - 1] = flag[i, j] if image[i + 1, j - 1] == 255 else 0
                        flag[i + 1, j] = flag[i, j] if image[i + 1, j] == 255 else 0
                else:                                                                               #第一行剩余元素
                    if image[i, j] == 255 and flag[i, j] == 0:                                      #白点且未标记
                        if flag[i,j-1]==0 and flag[i+1,j-1]==0 and flag[i+1,j]==0 and flag[i+1,j+1]==0 and flag[i,j+1]==0:
                            N += 1                                                                   #生成新区域标号
                            flag[i, j] = N
                            flag[i,j-1] = flag[i, j] if image[i,j-1] == 255 else 0                  #邻域是白点则划为同一区域
                            flag[i+1,j-1] = flag[i, j] if image[i+1,j-1] == 255 else 0
                            flag[i+1,j] = flag[i, j] if image[i+1,j] == 255 else 0
                            flag[i+1,j+1] = flag[i, j] if image[i+1,j+1] == 255 else 0
                            flag[i,j+1] = flag[i, j] if image[i,j+1] == 255 else 0
                    elif image[i, j] == 255 and flag[i, j] != 0:                                    #白点且已标记
                        flag[i, j - 1] = flag[i, j] if image[i, j - 1] == 255 else 0                #邻域是白点则划为同一区域
                        flag[i + 1, j - 1] = flag[i, j] if image[i + 1, j - 1] == 255 else 0
                        flag[i + 1, j] = flag[i, j] if image[i + 1, j] == 255 else 0
                        flag[i + 1, j + 1] = flag[i, j] if image[i + 1, j + 1] == 255 else 0
                        flag[i, j + 1] = flag[i, j] if image[i, j + 1] == 255 else 0
        elif i==image.shape[0]-1:                                                                      #最后一行
            for j in range(image.shape[1]):
                if j == 0:                                                                              #第一个元素
                    if image[i, j] == 255 and flag[i, j] == 0:
                        if flag[i,j+1]==0 and flag[i-1,j+1]==0 and flag[i-1,j]==0:
                            N+=1
                            flag[i, j] = N
                            flag[i,j+1] = flag[i, j] if image[i,j+1]==255 else 0
                            flag[i-1,j+1] = flag[i, j] if image[i-1,j+1] == 255 else 0
                            flag[i-1,j] = flag[i, j] if image[i-1,j] == 255 else 0
                    elif image[i, j] == 255 and flag[i, j] != 0:
                        flag[i, j + 1] = flag[i, j] if image[i, j + 1] == 255 else 0
                        flag[i - 1, j + 1] = flag[i, j] if image[i - 1, j + 1] == 255 else 0
                        flag[i - 1, j] = flag[i, j] if image[i - 1, j] == 255 else 0
                elif j == image.shape[1]-1:                                                            #最后一行最后一个元素
                    if image[i, j] == 255 and flag[i, j] == 0:
                        if flag[i,j-1] == 0 and flag[i-1,j-1] == 0 and flag[i-1,j] == 0:
                            N += 1
                            flag[i, j] = N
                            flag[i,j-1] = flag[i, j] if image[i,j-1] == 255 else 0
                            flag[i-1,j-1] = flag[i, j] if image[i-1,j-1] == 255 else 0
                            flag[i-1,j] = flag[i, j] if image[i-1,j] == 255 else 0
                else:                                                                                   #最后一行剩余元素
                    if image[i, j] == 255 and flag[i, j] == 0:
                        if flag[i,j-1]==0 and flag[i-1,j-1]==0 and flag[i-1,j]==0 and flag[i-1,j+1]==0 and flag[i,j+1]==0:
                            N += 1
                            flag[i, j] = N
                            flag[i,j-1] = flag[i, j] if image[i,j-1] == 255 else 0
                            flag[i-1,j-1] = flag[i, j] if image[i-1,j-1] == 255 else 0
                            flag[i-1,j] = flag[i, j] if image[i-1,j] == 255 else 0
                            flag[i-1,j+1] = flag[i, j] if image[i-1,j+1] == 255 else 0
                            flag[i,j+1] = flag[i, j] if image[i,j+1] == 255 else 0
                    elif image[i, j] == 255 and flag[i, j] != 0:
                        flag[i, j - 1] = flag[i, j] if image[i, j - 1] == 255 else 0
                        flag[i - 1, j - 1] = flag[i, j] if image[i - 1, j - 1] == 255 else 0
                        flag[i - 1, j] = flag[i, j] if image[i - 1, j] == 255 else 0
                        flag[i - 1, j + 1] = flag[i, j] if image[i - 1, j + 1] == 255 else 0
                        flag[i, j + 1] = flag[i, j] if image[i, j + 1] == 255 else 0
        else:                                                                                   #其余行
            for j in range(image.shape[1]):
                if j == 0:                                                             #第一个元素
                    if image[i, j] == 255 and flag[i, j] == 0:                         #白点且未标记
                        if flag[i+1,j]==0 and flag[i+1,j+1]==0 and flag[i,j+1]==0 and flag[i-1,j+1]==0 and flag[i-1,j]==0:     #如果邻域都未标记，则为起始点
                            N+=1                                                        #生成新区域标号
                            flag[i, j] = N
                            flag[i+1,j] = flag[i, j] if image[i+1,j]==255 else 0                   #邻域是白点则划为同一区域
                            flag[i+1,j+1] = flag[i, j] if image[i+1,j+1] == 255 else 0
                            flag[i,j+1] = flag[i, j] if image[i,j+1] == 255 else 0
                            flag[i-1,j+1] = flag[i, j] if image[i-1,j+1] == 255 else 0
                            flag[i-1,j] = flag[i, j] if image[i-1,j] == 255 else 0
                    elif image[i, j] == 255 and flag[i, j] != 0:
                        flag[i + 1, j] = flag[i, j] if image[i + 1, j] == 255 else 0  # 邻域是白点则划为同一区域
                        flag[i + 1, j + 1] = flag[i, j] if image[i + 1, j + 1] == 255 else 0
                        flag[i, j + 1] = flag[i, j] if image[i, j + 1] == 255 else 0
                        flag[i - 1, j + 1] = flag[i, j] if image[i - 1, j + 1] == 255 else 0
                        flag[i - 1, j] = flag[i, j] if image[i - 1, j] == 255 else 0
                elif j == image.shape[1]-1:                                                         #最后一个元素
                    if image[i, j] == 255 and flag[i, j] == 0:                                      #白点且未标记
                        if flag[i-1,j] == 0 and flag[i-1,j-1] == 0 and flag[i,j-1] == 0 and flag[i+1,j-1] == 0 and flag[i+1,j] == 0:            #如果邻域都未标记，则为起始点
                            N += 1                                                                   #生成新区域标号
                            flag[i, j] = N
                            flag[i-1,j] = flag[i, j] if image[i-1,j] == 255 else 0                  #邻域是白点则划为同一区域
                            flag[i-1,j-1] = flag[i, j] if image[i-1,j-1] == 255 else 0
                            flag[i,j-1] = flag[i, j] if image[i,j-1] == 255 else 0
                            flag[i+1,j-1] = flag[i, j] if image[i+1,j-1] == 255 else 0
                            flag[i+1,j] = flag[i, j] if image[i+1,j] == 255 else 0
                    elif image[i, j] == 255 and flag[i, j] != 0:                                    #白点且已标记
                        flag[i - 1, j] = flag[i, j] if image[i - 1, j] == 255 else 0                # 邻域是白点则划为同一区域
                        flag[i - 1, j - 1] = flag[i, j] if image[i - 1, j - 1] == 255 else 0
                        flag[i, j - 1] = flag[i, j] if image[i, j - 1] == 255 else 0
                        flag[i + 1, j - 1] = flag[i, j] if image[i + 1, j - 1] == 255 else 0
                        flag[i + 1, j] = flag[i, j] if image[i + 1, j] == 255 else 0
                else:                                                                               #剩余元素(可能产生重复区域)
                    if image[i, j] == 255 and flag[i, j] == 0:
                        if flag[i,j-1]==0 and flag[i+1,j-1]==0 and flag[i+1,j]==0 and flag[i+1,j+1]==0 and flag[i,j+1]==0 and flag[i-1,j+1]==0 and flag[i-1,j]==0 and flag[i-1,j-1]==0:
                            N += 1
                            flag[i, j] = N
                            for m in [0, 1, -1]:                                                    #遍历八邻域
                                for n in [0, 1, -1]:
                                    if m == 0 and n == 0:
                                        pass
                                    else:
                                        flag[i+m,j+n] = flag[i, j] if image[i+m,j+n] == 255 else 0
                    elif image[i, j] == 255 and flag[i, j] != 0:
                        for m in [0,1,-1]:                                                          #遍历八邻域
                            for n in [0, 1, -1]:
                                if m==0 and n==0:
                                    pass
                                else:
                                    if image[i+m, j+n] == 255 and flag[i+m, j+n] == 0:
                                        flag[i+m, j+n] = flag[i, j]
                                    elif image[i+m, j+n] == 255 and flag[i+m, j+n] != 0 and flag[i+m, j+n]!=flag[i, j]:
                                        overlap.append(sorted((flag[i+m, j+n], flag[i, j])))        #产生了重复区域



    if len(overlap)!=0:                                                                             #如果有重复命名区域
        L2 = []                                                                                     #首先对编号排序，归类
        for i in overlap:
            if not i in L2:
                L2.append(i)
        for i in range(len(L2)):
            for j in range(len(L2)):
                if L2[j][0]==L2[i][1]:
                    L2[j][0]=L2[i][0]
        for n in range(len(L2)):
            for i in range(flag.shape[0]):
                for j in range(flag.shape[1]):
                    if flag[i][j]==L2[n][1]:
                        flag[i][j]=L2[n][0]                                                          #合并为编号较小的区域
    L1 = sorted([(np.sum(flag == i), i) for i in set(flag.flat)])                                   #统计各连通域大小
    rangen = []                                                                                     #确定各个字符最终范围
    avewidth = 0                                                                                     #计算平均宽度
    L5 = []
    L6 = []
    for n in [i for i in set(flag.flat) if i!=0]:
        Ln=[]   #左右边界
        Lm=[]   #上下边界
        for i in range(flag.shape[0]):
            for j in range(flag.shape[1]):
                if flag[i][j]==n:
                    Ln.append(j)
                    Lm.append(i)
        rangen.append([min(Ln),max(Ln),min(Lm),max(Lm)])
    for i in range(len(rangen)):                                                        #该部分计算平均宽度来分割汉字位置
        if (rangen[i][3]-rangen[i][2])>=0.8*image.shape[0]:                                 #过滤高度不足的噪声
            if rangen[i][1]-rangen[i][0]>image.shape[1]*0.14:                               #如果字符黏连，从中间分开
                temp = rangen[i][0]+int((rangen[i][1]-rangen[i][0])*0.5)
                L5.append([rangen[i][0],temp,rangen[i][2],rangen[i][3]])
                L5.append([temp,rangen[i][1], rangen[i][2], rangen[i][3]])
            elif rangen[i][1]-rangen[i][0]<image.shape[1]*0.07:                             #去掉数字1来计算平均宽度
                pass
            else:
                L5.append(rangen[i])
    for i in range(len(L5)):
        avewidth += (L5[i][1]-L5[i][0])/len(L5)
    for i in range(len(rangen)):
        if (rangen[i][3]-rangen[i][2])>=0.8*image.shape[0]:                                 #过滤高度不足的噪声
            if rangen[i][1]-rangen[i][0]>image.shape[1]*0.14:                               #如果字符黏连，从中间分开
                temp = rangen[i][0]+int((rangen[i][1]-rangen[i][0])*0.5)
                L5.append([rangen[i][0],temp,rangen[i][2],rangen[i][3]])
                L5.append([temp,rangen[i][1], rangen[i][2], rangen[i][3]])
            elif rangen[i][1]-rangen[i][0]<image.shape[1]*0.07 and (rangen[i][1]>0.95*image.shape[1] or rangen[i][0]<0.1*image.shape[1]):  #排除宽度过窄的噪声(仅位于边缘)
                pass
            elif rangen[i][1]-rangen[i][0]<image.shape[1]*0.07:
                left = int(rangen[i][0]+(rangen[i][1]-rangen[i][0])*0.5-0.5*avewidth)
                right = int(rangen[i][0]+(rangen[i][1]-rangen[i][0])*0.5+0.5*avewidth)
                L5.append([left,right,rangen[i][2],rangen[i][3]])
    res =sorted(L5,key=lambda item: item[0])
######由于汉字连通域可能不止一个，所以不使用上述方法分割，而是使用第一个数字的位置利用上面求的平均宽度估计汉字位置#####
    if len(res)==7:                                                                         #如果汉字被识别了，将它替换掉
        left = res[1][0]-int(avewidth)-4
        right = res[1][1]-int(avewidth)-3
        res[0]=[left,right,res[1][2],res[1][3]]
    elif len(res)==6:                                                  #汉字未能识别，在第一个索引位置加入估计的汉字坐标
        left = res[0][0] - int(avewidth) - 4
        right = res[0][1] - int(avewidth) - 3
        res.insert(0,[left,right,res[1][2],res[1][3]])
    imaged = img[posit_x[0]:posit_x[1], posit_y[0]:posit_y[1]]
    fin = []
    for i in range(len(res)):                                          #将字符分为七个小字符输出
        fin.append(imaged[res[i][2]:res[i][3],res[i][0]:res[i][1]])
    return fin

def feature_ext(image):                                                      #字符特征提取
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                           #灰度化
    norm = res=cv2.resize(gray,(8,16),interpolation=cv2.INTER_CUBIC)         #归一化到16*8大小
    ret, img = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #OTSU二值化
    Lx = []
    sumx = 0
    Ly = []
    sumy = 0
    for j in range(img.shape[1]):                                    #将每一列中的白点个数累加，作为特征向量 8 个
        for i in range(img.shape[0]):
            if img[i,j]==255:
                Lx.append(1)
            else:
                Lx.append(0)
    return Lx

































