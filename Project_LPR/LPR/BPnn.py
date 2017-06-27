# -*- coding: utf-8 -*-
import math
import random

random.seed(0)                           #每次生成的随机数相同


def rand(a, b):                         #生成随机数
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):        #创建指定大小矩阵
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):                         #定义sigmod函数和它的导数:
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class BPNeuralNetwork:                  #定义BP神经网络类
    def __init__(self):
        self.input_n = 0                #输入层神经元个数
        self.hidden_n = 0               #隐含层神经元个数
        self.output_n = 0               #输出层神经元个数
        self.input_cells = []           #输入层神经元
        self.hidden_cells = []          #隐含层神经元
        self.output_cells = []          #输出层神经元
        self.input_weights = []         #输入层到隐含层权值
        self.output_weights = []        #隐含层到输出层权值
        self.input_correction = []      #输入矫正矩阵
        self.output_correction = []     #输出矫正矩阵
##################初始化神经网络#################
    def setup(self, ni, nh, no):
        self.input_n = ni + 1           #输入层额外增加一个偏置神经元，提供一个可控的输入修正
        self.hidden_n = nh
        self.output_n = no
        # 输入神经元值
        self.input_cells = [1.0] * self.input_n         #神经元输入值为列表形式
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # 输入权值
        self.input_weights = make_matrix(self.input_n, self.hidden_n)   #权值储存在矩阵里
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # 权值初值为随机数
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-2.0, 2.0)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # 矫正矩阵
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)
###################预测部分#######################
    def predict(self, inputs):
        # 将待预测值输入神经元
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # 各个隐含层值为输入层各个神经元加权和
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)       #加权和送入sigmoid函数得到最终结果
        # 各个输出层值为隐含层各个神经元加权和
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)      #加权和送入sigmoid函数得到最终结果
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # 反向传播
        self.predict(case)                             #一次前馈，得到结果
        # 得到输出误差
        output_deltas = [0.0] * self.output_n          #初始化误差矩阵
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]    #误差为期望值label与实际输出值作差
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error #输出层误差的最终形式
        # 得到隐含层误差
        hidden_deltas = [0.0] * self.hidden_n          #初始化误差矩阵
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):            #隐含层输出不存在参考值， 使用输出层误差的加权和代替
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # 输出权值更新
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]            #E*O
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o] #learn为学习率,correct为矫正率
                self.output_correction[h][o] = change                      #更新矫正矩阵
        # 输入权值更新
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # 损失函数选用输出层各节点的方差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):   #定义train方法控制迭代
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):                                     #对所有样本进行训练
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
            if j % 100 == 0:
                print('error %-.5f' % error)
            if error<0.1:                                                   #训练终止条件，当误差和达到要求停止
                break
    def test(self):                                                        #test方法，使用神经网络学习异或逻辑
        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        labels = [[0], [1], [1], [0]]
        self.setup(2, 5, 1)                                                 #输入两个特征，输出一个特征
        self.train(cases, labels, 10000, 0.05, 0.1)
        for case in cases:                                                 #四个样本，每个样本有两个特征
            print(self.predict(case))

if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()
