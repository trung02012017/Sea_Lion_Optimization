import numpy as np
import json
import os

class Fitness_Selector(object):

    def __init__(self):
        self.function_dict = None

    def f1(self, particle):
        return sum([particle[i]**2 for i in range(particle.shape[0])])

    def f2(self, particle):
        x = np.abs(particle)
        return np.sum(x) + np.prod(x)

    def f3(self, particle):
        fitness = 0
        for i in range(particle.shape[0]):
            for j in range(i+1):
              fitness += particle[j]
        return fitness

    def f4(self, particle):
        x = np.abs(particle)
        return np.max(x)

    def f5(self, particle):
        return sum([(100*((particle[i+1] - particle[i]**2)**2) + (particle[i] -1)**2) for i in range(particle.shape[0]-1)])

    def f6(self, particle):
        return np.sum([(particle[i] + 0.5)**2 for i in range(particle.shape[0])])

    def f7(self, particle):
        return np.sum([(i+1)*particle[i]**4 for i in range(particle.shape[0])]) + np.random.rand()

    def f8(self, particle):
        return particle[0]**2 + 10e6*np.sum([particle[i]**6 for i in range(1, particle.shape[0])])

    def f9(self, particle):
        return 10e6*particle[0]**2 + np.sum([particle[i]**6 for i in range(1, particle.shape[0])])

    def f10(self, particle):
        return (particle[0] - 1)**2 + np.sum([i*(2*particle[i]**2 - particle[i-1])**2 for i in range(1, particle.shape[0])])

    def f11(self, particle):
        return np.sum([((10e6)**((i)/(particle.shape[0]-1)))*particle[i]**2 for i in range(2, particle.shape[0])])

    def f12(self, particle):
        return np.sum([(i+1)*particle[i]**2 for i in range(particle.shape[0])])

    def f13(self, particle):
        return np.sum([particle[i]**2 for i in range(particle.shape[0])]) + \
               (np.sum([0.5*i*particle[i]**2 for i in range(particle.shape[0])]))**2 + \
               (np.sum([particle[i]**2 for i in range(particle.shape[0])]))**4

    def f14(self, particle):
        return np.sum([-1*particle[i]*np.sin(np.abs(particle[i])**0.5) for i in range(particle.shape[0])])

    def f15(self, particle):
        return np.sum([particle[i]**2 - 10*np.cos(2*np.pi*particle[i]) + 10 for i in range(particle.shape[0])])

    def f16(self, particle):
        return -20*np.exp(-0.2*(1/particle.shape[0]*np.sum([particle[i]**2 for i in range(particle.shape[0])]))**0.5) - \
               np.exp(1/particle.shape[0]*np.sum([np.cos(2*np.pi*particle[i]) for i in range(particle.shape[0])])) + 20 + \
               np.e
    def f17(self, particle):
        return 1/4000*np.sum([particle[i]**2 for i in range(particle.shape[0])]) - \
               np.prod([np.cos(particle[i]/((i+1)**0.5)) for i in range(particle.shape[0])]) + 1

    def f18(self, particle):
        x = particle
        res = 0
        dim = len(x)
        A = 0
        B = 0

        def g(x, y):
            return 0.5 + (np.square(np.sin(np.sqrt(x * x + y * y))) - 0.5) / \
                   np.square(1 + 0.001 * np.square((x * x + y * y)))

        for i in range(dim):
            res += g(x[i], x[(i + 1) % dim])
        return res

    def f19(self, particle):
        return 0

    def f20(self, particle):
        return np.sum([((np.sum([(0.5**k)*np.cos(2*np.pi*(3**k)*(particle[i]+0.5)) for k in range(21)])) -
                       (particle.shape[0]*np.sum([(0.5**j)*np.cos(np.pi*(3**j)) for j in range(21)])))
                       for i in range(particle.shape[0])])

    def f21(self, particle):
        return np.sum([np.abs(particle[i]*np.sin(particle[i]) + 0.1*particle[i]) for i in range(particle.shape[0])])

    def f22(self, particle):
        return 0.5 + ((np.sin(np.sum([particle[i]**2 for i in range(particle.shape[0])])))**2 - 0.5)*\
               (1+0.001*(np.sum([particle[i]**2 for i in range(particle.shape[0])])))**-2

    def f23(self, particle):
        return 1/particle.shape[0]* \
               np.sum([particle[i]**4 - 16*particle[i]**2 + 5*particle[i] for i in range(particle.shape[0])])

    def f24(self, particle):
        return np.sum([particle[i]**2 + 2*particle[i+1]**2 - 0.3*np.cos(3*np.pi*particle[i]) -
                       0.4*np.cos(4*np.pi*particle[i+1]) + 0.7 for i in range(particle.shape[0]-1)])

    def f25(self, particle):
        return -1*(-0.1*np.sum([np.cos(5*np.pi*particle[i]) for i in range(particle.shape[0])]) -
                   np.sum([particle[i]**2 for i in range(particle.shape[0])]))

    def shift(self, solution, shift_number):
        return np.array(solution) - shift_number

    def C1(self, solution, problem_size=None, shift_num=1, rate=1):
        x = self.shift(solution, shift_num)
        return self.CEC_1(x) + 100 * rate

    def CEC_2(self, solution=None, problem_size=None, shift=0):
        """
        Bent cigar function
        f(x*) =  200
        """
        res = 0
        constant = np.power(10, 6)
        dim = len(solution)
        res = np.square((solution[0] - shift))
        for i in range(1, dim):
            res += constant * np.square((solution[i] - shift))
        return res

    def CEC_3(self, solution=None, problem_size=None, shift=0):
        """
        Discus Function
        f(x*) = 300
        """
        x = solution - shift
        constant = np.power(10, 6)
        dim = len(solution)
        res = constant * np.square(x[0])
        for i in range(1, dim):
            res += np.square(x[i])
        return res

    def CEC_4(self, solution=None, problem_size=None, shift=0):
        """
        rosenbrock Function
        f(x*) = 400
        """
        x = solution - shift
        constant = np.power(10, 6)
        dim = len(solution)
        res = 0
        for i in range(dim - 1):
            res += 100 * np.square(x[i] ** 2 - x[i + 1]) + np.square(x[i] - 1)
        return res

    def CEC_5(self, solution=None, problem_size=None, shift=0):
        """
        Ackley’s Function
        """
        x = solution - shift
        dim = len(solution)
        res = 0
        A = 0
        B = 0
        A += -0.2 * np.sqrt(np.sum(np.square(x)) / dim)
        B += np.sum(np.cos(2 * np.pi * x)) / dim
        res = -20 * np.exp(A) - np.exp(B) + 20 + np.e
        # print("res", res)
        return res

    def CEC_6(self, solution=None, problem_size=None, shift=0):
        """
        Weierstrass Function
        """
        x = solution - shift
        dim = len(solution)
        res = 0
        kmax = 1
        a = 0.5
        b = 3
        A = 0
        B = 0
        for i in range(dim):
            for k in range(kmax + 1):
                A += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (x[i] + 0.5))
        for k in range(kmax + 1):
            B += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * 0.5)
        res = A - dim * B
        return res

    def CEC_7(self, solution=None, problem_size=None, shift=0):
        x = solution - shift
        res = 0
        A = np.sum(np.square(x)) / 4000
        B = 1
        if isinstance(x, np.ndarray):
            dim = len(x)
            for i in range(dim):
                B *= np.cos(x[i] / np.sqrt(i + 1))
        else:
            B = np.cos(x)
        res = A - B + 1
        return res

    def CEC_8(self, solution=None, problem_size=None, shift=0):
        x = solution - shift
        res = 0
        dim = len(x)
        res = np.sum(np.square(x)) - 10 * np.sum(np.cos(2 * np.pi * x)) + 10 * dim
        return res

    def g9(self, z, dim):
        if np.abs(z) <= 500:
            return z * np.sin(np.power(np.abs(z), 1 / 2))
        elif z > 500:
            return (500 - z % 500) * np.sin(np.sqrt(np.abs(500 - z % 500))) \
                   - np.square(z - 500) / (10000 * dim)
        else:
            return (z % 500 - 500) * np.sin(np.sqrt(np.abs(z % 500 - 500))) \
                   - np.square(z + 500) / (10000 * dim)

    def CEC_9(self, solution=None, problem_size=None, shift=0):
        x = solution - shift
        res = 0
        dim = len(x)
        A = 0
        B = 0
        A = 418.9829 * dim
        z = x + 4.209687462275036e+002
        for i in range(dim):
            B += g9(z[i], dim)
        res = A - B
        return res

    def CEC_10(self, solution=None, problem_size=None, shift=0):
        x = solution - shift
        res = 0
        dim = len(x)
        A = 1
        B = 0
        for i in range(dim):
            temp = 1
            for j in range(32):
                temp += i * (np.abs(np.power(2, j + 1) * x[i]
                                    - round(np.power(2, j + 1) * x[i]))) / np.power(2, j)
            A *= np.power(temp, 10 / np.power(dim, 1.2))
        B = 10 / np.square(dim)
        res = B * A - B
        return res

    def CEC_11(self, solution=None, problem_size=None, shift=0):
        x = solution - shift
        res = 0
        dim = len(x)
        A = 0
        B = 0
        A = np.power(np.abs(np.sum(np.square(x)) - dim), 1 / 4)
        B = (0.5 * np.sum(np.square(x)) + np.sum(x)) / dim
        res = A + B + 0.5
        return res

    def CEC_12(self, solution=None, problem_size=None, shift=0):
        x = solution - shift
        res = 0
        dim = len(x)
        A = 0
        B = 0
        A = np.power(np.abs(np.square(np.sum(np.square(x))) - np.square(np.sum(x))), 1 / 2)
        B = (0.5 * np.sum(np.square(x)) + np.sum(x)) / dim
        res = A + B + 0.5
        return res

    def CEC_13(self, solution=None, problem_size=None, shift=0):
        x = solution - shift
        res = 0
        dim = len(x)
        A = 0
        B = 0
        for i in range(dim):
            res += self.CEC_7(self.CEC_4(x[i: (i + 2) % dim], shift=0), shift=0)
        return res

    def CEC_14(self, solution=None, problem_size=None, shift=0):
        x = solution - shift
        res = 0
        dim = len(x)
        A = 0
        B = 0

        def g(x, y):
            return 0.5 + (np.square(np.sin(np.sqrt(x * x + y * y))) - 0.5) / \
                   np.square(1 + 0.001 * np.square((x * x + y * y)))

        for i in range(dim):
            res += g(x[i], x[(i + 1) % dim])
        return res

    def shift(self, solution, shift_number):
        return np.array(solution) - shift_number

    def rotate(self, solution, original_x, rotate_rate=1):
        return solution

    def C1(self, problem_size=None, shift_num=1, rate=1):
        x = self.shift(solution, shift_num)
        return self.CEC_1(x) + 100 * rate

    def C2(self, solution, prolem_size=None, shift_num=1, rate=1):
        x = self.shift(solution, shift_num)
        return self.CEC_2(x) + 200 * rate

    def C3(self, solution, prolem_size=None, shift_num=1, rate=1):
        x = self.shift(solution, shift_num)
        return self.CEC_3(x) + 300 * rate

    def C4(self, solution, prolem_size=None, shift_num=2, rate=1):
        x = 2.48 / 100 * self.shift(solution, shift_num)
        x = self.rotate(x, solution) + 1
        return self.CEC_4(x) + 400 * rate

    def C5(self, solution, prolem_size=None, shift_num=1, rate=1):
        x = self.shift(solution, shift_num)
        x = self.rotate(x, solution)
        return self.CEC_5(x) + 500 * rate

    def C6(self, solution, prolem_size=None, shift_num=1, rate=1):
        x = 0.5 / 100 * self, shift(solution, shift_num)
        return self.CEC_6(x) + 600 * rate

    def C7(self, solution, prolem_size=None, shift_num=1, rate=1):
        x = 600 / 100 * self.shift(solution, shift_num)
        return self.CEC_7(x) + 700 * rate

    def C8(solution, prolem_size=None, shift_num=1, rate=1):
        x = 5.12 / 100 * shift(solution, shift_num)
        return CEC_8(x) + 800 * rate

    def C9(solution, prolem_size=None, shift_num=1, rate=1):
        x = 5.12 / 100 * shift(solution, shift_num)
        x = rotate(x, solution)
        return CEC_8(x) + 900 * rate

    def C10(solution, prolem_size=None, shift_num=1, rate=1):
        x = 1000 / 100 * shift(solution, shift_num)
        return CEC_9(x) + 1000 * rate

    def C11(solution, prolem_size=None, shift_num=1, rate=1):
        x = 1000 / 100 * shift(solution, shift_num)
        x = rotate(x, solution)
        return CEC_9(x) + 1100 * rate

    def C12(solution, prolem_size=None, shift_num=1, rate=1):
        x = 5 / 100 * shift(solution, shift_num)
        x = rotate(x, solution)
        return CEC_10(x) + 1200 * rate

    def C13(solution, prolem_size=None, shift_num=1, rate=1):
        x = 5 / 100 * shift(solution, shift_num)
        x = rotate(x, solution)
        return CEC_11(x) + 1300 * rate

    def C14(solution, prolem_size=None, shift_num=1, rate=1):
        x = 5 / 100 * shift(solution, shift_num)
        x = rotate(x, solution)
        return CEC_12(x) + 1400 * rate

    def C15(solution, prolem_size=None, shift_num=2, rate=1):
        x = 5 / 100 * shift(solution, shift_num)
        x = rotate(x, solution) + 1
        return CEC_13(x) + 1500 * rate

    def C16(solution, prolem_size=None, shift_num=1, rate=1):
        x = 5 / 100 * shift(solution, shift_num)
        x = rotate(x, solution) + 1
        return CEC_14(x) + 1600 * rate

    def C17(solution, prolem_size=None, shift_num=1, rate=1):
        dim = len(solution)
        n1 = int(0.3 * dim)
        n2 = int(0.3 * dim) + n1
        D = np.arange(dim)

        # np.random.shuffle(D)
        x = shift(solution, shift_num)
        return CEC_9(x[D[: n1]]) + CEC_8(x[D[n1: n2]]) + CEC_1(x[D[n2:]]) + 1700 * rate

    def C18(solution, prolem_size=None, shift_num=1, rate=1):
        dim = len(solution)
        n1 = int(0.3 * dim)
        n2 = int(0.3 * dim) + n1
        D = np.arange(dim)
        # np.random.shuffle(D)
        x = shift(solution, shift_num)
        return CEC_2(x[D[: n1]]) + CEC_12(x[D[n1: n2]]) + CEC_8(x[D[n2:]]) + 1800 * rate

    def C19(solution, prolem_size=None, shift_num=1, rate=1):
        dim = len(solution)
        n1 = int(0.2 * dim)
        n2 = int(0.2 * dim) + n1
        n3 = int(0.3 * dim) + n2
        D = np.arange(dim)
        # np.random.shuffle(D)
        x = shift(solution, shift_num)
        return CEC_7(x[D[: n1]]) + CEC_6(x[D[n1: n2]]) + CEC_4(x[D[n2: n3]]) + CEC_14(x[D[n3:]]) + 1900 * rate

    def C20(solution, prolem_size=None, shift_num=1, rate=1):
        dim = len(solution)
        n1 = int(0.2 * dim)
        n2 = int(0.2 * dim) + n1
        n3 = int(0.3 * dim) + n2
        D = np.arange(dim)
        # np.random.shuffle(D)
        x = shift(solution, shift_num)
        return CEC_12(x[D[: n1]]) + CEC_3(x[D[n1: n2]]) + CEC_13(x[D[n2: n3]]) + CEC_8(x[D[n3:]]) + 2000 * rate

    def C21(solution, prolem_size=None, shift_num=1, rate=1):
        dim = len(solution)
        n1 = int(0.1 * dim)
        n2 = int(0.2 * dim) + n1
        n3 = int(0.2 * dim) + n2
        n4 = int(0.2 * dim) + n3
        D = np.arange(dim)
        # np.random.shuffle(D)
        x = shift(solution, shift_num)
        return CEC_14(x[D[: n1]]) + CEC_12(x[D[n1: n2]]) + CEC_4(x[D[n2: n3]]) + CEC_9(x[D[n3: n4]]) + CEC_1(
            x[D[n4:]]) + 2100 * rate

    def C22(solution, prolem_size=None, shift_num=1, rate=1):
        dim = len(solution)
        n1 = int(0.1 * dim)
        n2 = int(0.2 * dim) + n1
        n3 = int(0.2 * dim) + n2
        n4 = int(0.2 * dim) + n3
        D = np.arange(dim)
        # np.random.shuffle(D)
        x = shift(solution, shift_num)
        return CEC_10(x[D[: n1]]) + CEC_11(x[D[n1: n2]]) + CEC_13(x[D[n2: n3]]) + CEC_9(x[D[n3: n4]]) + \
               CEC_5(x[D[n4:]]) + 2200 * rate

    def C23(solution, prolem_size=None, shift_num=1, rate=1):
        shift_arr = [1, 2, 3, 4, 5]
        sigma = [10, 20, 30, 40, 50]
        lamda = [1, 1.0e-6, 1.0e-26, 1.0e-6, 1.0e-6]
        bias = [0, 100, 200, 300, 400]
        fun = [C4, C1, C2, C3, C1]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = shift(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2300 * rate

    def C24(solution, prolem_size=None, shift_num=1, rate=1):
        shift_arr = [1, 2, 3]
        sigma = [20, 20, 20]
        lamda = [1, 1, 1]
        bias = [0, 100, 200]
        fun = [C10, C9, C14]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = shift(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
                   * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2400 * rate

    def C25(solution, prolem_size=None, shift_num=1, rate=1):
        shift_arr = [1, 2, 3]
        sigma = [10, 30, 50]
        lamda = [0.25, 1, 1.0e-7]
        bias = [0, 100, 200]
        fun = [C11, C9, C1]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = shift(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
                   * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2500 * rate

    def C26(solution, prolem_size=None, shift_num=1, rate=1):
        shift_arr = [1, 2, 3, 4, 5]
        sigma = [10, 10, 10, 10, 10]
        lamda = [0.25, 1.0, 1.0e-7, 2.5, 10.0]
        bias = [0, 100, 200, 300, 400]
        fun = [C11, C13, C1, C6, C7]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = shift(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
                   * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2600 * rate

    def C27(solution, prolem_size=None, shift_num=1, rate=1):
        shift_arr = [1, 2, 3, 4, 5]
        sigma = [10, 10, 10, 20, 20]
        lamda = [10, 10, 2.5, 25, 1.0e-6]
        bias = [0, 100, 200, 300, 400]
        fun = [C14, C9, C11, C6, C1]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = shift(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
                   * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2700 * rate

    def C28(solution, prolem_size=None, shift_num=1, rate=1):
        shift_arr = [1, 2, 3, 4, 5]
        sigma = [10, 20, 30, 40, 50]
        lamda = [2.5, 10, 2.5, 5.0e-4, 1.0e-6]
        bias = [0, 100, 200, 300, 400]
        fun = [C15, C13, C11, C16, C1]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = shift(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
                   * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2800 * rate

    def C29(solution, prolem_size=None, shift_num=1, rate=1):
        shift_arr = [4, 5, 6]
        sigma = [10, 30, 50]
        lamda = [1, 1, 1]
        bias = [0, 100, 200]
        fun = [C17, C18, C19]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = shift(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
                   * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 2900 * rate

    def C30(solution, prolem_size=None, shift_num=1, rate=1):
        shift_arr = [1, 2, 3]
        sigma = [10, 30, 50]
        lamda = [1, 1, 1]
        bias = [0, 100, 200]
        fun = [C20, C21, C22]
        dim = len(solution)
        res = 0
        w = np.zeros(len(shift_arr))
        for i in range(len(shift_arr)):
            x = shift(solution, shift_arr[i])
            w[i] = 1 / np.sqrt(np.sum(np.square(x))) \
                   * np.exp(- np.sum(np.square(x)) / (2 * dim * np.square(sigma[i])))
        for i in range(len(shift_arr)):
            res += w[i] / np.sum(w) * (lamda[i] * fun[i](solution, rate=0) + bias[i])
        return res + 3000 * rate

    def chose_function(self, function_name):
        function_dict = {
            'f1': self.f1,
            'f2': self.f2,
            'f3': self.f3,
            'f4': self.f4,
            'f5': self.f5,
            'f6': self.f6,
            'f7': self.f7,
            'f8': self.f8,
            'f9': self.f9,
            'f10': self.f10,
            'f11': self.f11,
            'f12': self.f12,
            'f13': self.f13,
            'f14': self.f14,
            'f15': self.f15,
            'f16': self.f16,
            'f17': self.f17,
            'f18': self.f18,
            'f19': self.f19,
            'f20': self.f20,
            'f21': self.f21,
            'f22': self.f22,
            'f23': self.f23,
            'f24': self.f24,
            'f25': self.f25,
            'C1': self.C1,
            'C2': self.C2,
        }
        return function_dict[function_name]


if __name__ == '__main__':

    # a = np.array([2, 3, 4])
    # fitness_selector = Fitness_Selector()
    # fitness_function = fitness_selector.chose_function('f1')
    # print(fitness_function(a))
    # print(round(1.1) - 1 )
    path = os.path.dirname(os.path.realpath(__file__))
    params_path = os.path.join(os.path.dirname(path), 'parameter_setup')
    print(params_path)
    for file in os.listdir(params_path):
        print(file)


#
# igso_parameter_set = data["parameters"]["IGSO"]
# m_s = igso_parameter_set['m']
# n_s = igso_parameter_set['n']
# l1_s = igso_parameter_set['l1']
# l2_s = igso_parameter_set['l2']
# max_ep_s = igso_parameter_set['max_ep']
#
# combinations = []
# for m in m_s:
#     for n in n_s:
#         for l1 in l1_s:
#             for l2 in l2_s:
#                 for max_ep in max_ep_s:
#                     combination = [m, n, l1, l2, max_ep]
#                     function_evaluation = (m*n*l1 + m*l2)*max_ep
#                     if function_evaluation >= 50000 and function_evaluation <= 60000:
#                         print("combination: {} and function evaluation: {}".format(str(combination),
#                                                                                    function_evaluation))
#                         combinations.append(combination)
# print(len(combinations))