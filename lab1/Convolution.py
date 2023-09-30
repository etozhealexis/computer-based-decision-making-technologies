import math
import matplotlib.pyplot as plt
import numpy as np
import random as rand

from ConvolutionData import ConvolutionData


class Convolution:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def solveLab(self):
        additiveConvolutionData = self.additiveConvolution()
        variablesConvolutionData = self.variablesConvolution()
        multiplicativeConvolutionData = self.multiplicativeConvolution

        self.buildView(additiveConvolutionData)
        self.buildView(variablesConvolutionData)
        self.buildView(multiplicativeConvolutionData, np.arange(2, 6.5, 0.05))

        self.findBestConvolution(additiveConvolutionData, variablesConvolutionData, multiplicativeConvolutionData)

    def additiveConvolution(self):
        A = 0
        B = 0
        C = 0
        D = 0
        for i in range(self.len):
            A += self.x[i] ** 2
            B += self.x[i]
            C += self.x[i] * self.y[i]
            D += self.y[i]
        k = (C * self.len - B * D) / (A * self.len - B ** 2)
        b = (A * D - B * C) / (A * self.len - B ** 2)
        k1 = abs(k) / (abs(k) + 1)
        k2 = 1 / (abs(k) + 1)
        MSE = 0
        for i in range(self.len):
            MSE += (self.y[i] - (k * self.x[i] + b)) ** 2
        MSE *= 1 / self.len

        return ConvolutionData("additive", A, B, C, D, k, b, k1, k2, MSE)

    def variablesConvolution(self):
        A = 0
        B = 0
        C = 0
        D = 0
        for i in range(self.len):
            A += self.x[i] ** 4
            B += self.x[i] ** 2
            C += self.y[i] * self.x[i] ** 2
            D += self.y[i]
        a = (C * self.len - B * D) / (B ** 2 - A * self.len)
        b = (B * C - A * D) / (B ** 2 - A * self.len)
        k1 = a / (a + 1)
        k2 = 1 / (a + 1)
        MSE = 0
        for i in range(self.len):
            MSE += (self.y[i] - ((-a) * (self.x[i] ** 2) + b)) ** 2
        MSE *= 1 / self.len

        return ConvolutionData("variables", A, B, C, D, a, b, k1, k2, MSE)

    @property
    def multiplicativeConvolution(self):
        A = 0
        B = 0
        C = 0
        D = 0
        for i in range(self.len):
            A += math.log(self.x[i]) ** 2
            B += math.log(self.x[i])
            C += math.log(self.x[i]) * math.log(self.y[i])
            D += math.log(self.y[i])
        k = (C * self.len - B * D) / (A * self.len - B ** 2)
        b = (A * D - B * C) / (A * self.len - B ** 2)
        k1 = abs(k) / (abs(k) + 1)
        k2 = 1 / (abs(k) + 1)
        MSE = 0
        for i in range(self.len):
            MSE += (self.y[i] - (math.exp(k * math.log(self.x[i]) + b))) ** 2
        MSE *= 1 / self.len

        return ConvolutionData("multiplicative", A, B, C, D, k, b, k1, k2, MSE)

    def findBestConvolution(self, convolutionData1: ConvolutionData, convolutionData2: ConvolutionData, convolutionData3: ConvolutionData):
        minMSE = np.min([convolutionData1.MSE, convolutionData2.MSE, convolutionData3.MSE])

        print("Из трех рассмотренных сверток в данном примере следует выбрать")

        if minMSE == convolutionData1.MSE:
            print("аддитивную свертку:")
            self.alternatives(convolutionData1.weightK1, convolutionData1.weightK2)

        if minMSE == convolutionData2.MSE:
            print("свертку c неравными степенями переменных:")
            self.alternatives(convolutionData2.weightK1, convolutionData1.weightK2)

        if minMSE == convolutionData3.MSE:
            print("мультипликативную свертку:")
            self.alternatives(convolutionData3.weightK1, convolutionData1.weightK2)

    def alternatives(self, k1, k2):
        print(f"k1 = {round(k1, 2)}, k2 = {round(k2, 2)}")

        minX = np.min(self.x)
        maxX = np.max(self.x)
        minY = np.min(self.y)
        maxY = np.max(self.y)

        resA = {}
        for i in range(6):
            a = (rand.randint(minX, maxX), rand.randint(minY, maxY))
            u = a[0] ** k1 * a[1] ** k2
            print(f"a_{i} = <{a[0]}, {a[1]}>, u_{i} = {round(u, 1)}")
            resA[i] = u

        def addA(index):
            return f'a_{index}'

        resSorted = dict(sorted(resA.items(), key=lambda item: item[1], reverse=True))
        print(' > '.join(map(addA, map(str, resSorted.keys()))))

    def buildView(self, convolutionData, x_axis=None):
        if convolutionData.type == "additive":
            print("Аддитивная свертка")
        if convolutionData.type == "variables":
            print("Свертка с неравными степенями переменных")
        if convolutionData.type == "multiplicative":
            print("Мультипликативная свертка")
        self.buildKoeffsView(convolutionData)
        self.buildApproximationView(convolutionData, x_axis)

    def buildKoeffsView(self, convolutionData: ConvolutionData):
        print(f'A = {convolutionData.a}\nB = {convolutionData.b}\nC = {convolutionData.c}\nD = {convolutionData.d}')
        print(f'k = {convolutionData.aproxK1}\nb = {convolutionData.aproxK2}')
        print(f'k_1 = {convolutionData.weightK1}\nk_2 = {convolutionData.weightK2}')
        print(f'MSE: {convolutionData.MSE}')

    def buildApproximationView(self, convolutionData: ConvolutionData, xAxis):
        if convolutionData.type == "additive":
            aY = lambda aX: convolutionData.aproxK1 * aX + convolutionData.aproxK2
        if convolutionData.type == "variables":
            aY = lambda aX: convolutionData.aproxK2 - convolutionData.aproxK1 * (aX ** 2)
        if convolutionData.type == "multiplicative":
            aY = lambda aX: np.exp(convolutionData.aproxK1 * np.log(aX) + convolutionData.aproxK2)

        aX = np.linspace(0, 10, 100)
        if xAxis is not None:
            aX = xAxis

        plt.plot(aX, aY(aX))
        plt.scatter(self.x, self.y)
        plt.grid()
        plt.show()
