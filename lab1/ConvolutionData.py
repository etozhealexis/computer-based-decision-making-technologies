class ConvolutionData:
    def __init__(self, type, a, b, c, d, aproxK1, aproxK2, weightK1, weightK2, MSE):
        self.type = type
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.aproxK1 = aproxK1
        self.aproxK2 = aproxK2
        self.weightK1 = weightK1
        self.weightK2 = weightK2
        self.MSE = MSE