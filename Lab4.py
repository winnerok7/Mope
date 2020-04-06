import math
import numpy as np
from numpy.linalg import solve
from scipy.stats import f, t
from functools import partial
from random import randint
from prettytable import PrettyTable


# testing functions
def cochrane(g_prac, g_teor):
    return g_prac < g_teor


def student(t_teor, t_pr):
    return t_pr < t_teor


def fischer(f_teor, f_prac):
    return f_teor > f_prac


def cochrane_teor(f1, f2, q=0.05):
    q1 = q / f1
    fischer_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
    return fischer_value / (fischer_value + f1 - 1)


fischer_teor = partial(f.ppf, q=1 - 0.05)
student_teor = partial(t.ppf, q=1 - 0.025)

X1min = -5
X1max = 15
X2min = -15
X2max = 35
X3min = 15
X3max = 30

Xmin_average = (X1min + X2min + X3min) / 3  # Xcp(min)
Xmax_average = (X1max + X2max + X3max) / 3  # Xcp(max)

y_max = round(200 + Xmax_average)
y_min = round(200 + Xmin_average)

while True:
    # matrix
    x0_factor = [1, 1, 1, 1, 1, 1, 1, 1]
    x1_factor = [-1, -1, 1, 1, -1, -1, 1, 1]
    x2_factor = [-1, 1, -1, 1, -1, 1, -1, 1]
    x3_factor = [-1, 1, 1, -1, 1, -1, -1, 1]
    x1x2_factor = [a * b for a, b in zip(x1_factor, x2_factor)]
    x1x3_factor = [a * b for a, b in zip(x1_factor, x3_factor)]
    x2x3_factor = [a * b for a, b in zip(x2_factor, x3_factor)]
    x1x2x3_factor = [a * b * c for a, b, c in zip(x1_factor, x2_factor, x3_factor)]
    factors = [x0_factor, x1_factor, x2_factor, x3_factor, x1x2_factor, x1x3_factor, x2x3_factor, x1x2x3_factor]

    m = 3  # repeat qty

    y1, y2, y3 = [randint(y_min, y_max) for i in range(8)], [randint(y_min, y_max) for i in range(8)], [
        randint(y_min, y_max) for i in range(8)]  # filling y

    # table rows (y values)
    y_dict = {}
    for x in range(1, 9):
        y_dict["y_row{0}".format(x)] = [y1[x - 1], y2[x - 1], y3[x - 1]]

    # averages for each y-row
    y_avg_dict = {}
    for x in range(1, 9):
        y_avg_dict["y_avg{0}".format(x)] = np.average(y_dict[f'y_row{x}'])
    Y_average = [round(val, 3) for val in y_avg_dict.values()]

    # reinitializing arrs + swapping to min/max values
    x0 = [1, 1, 1, 1, 1, 1, 1, 1]
    x1 = [X1min, X1min, X1max, X1max, X1min, X1min, X1max, X1max]
    x2 = [X2min, X2max, X2min, X2max, X2min, X2max, X2min, X2max]
    x3 = [X3min, X3max, X3max, X3min, X3max, X3min, X3min, X3max]
    x1x2 = [a * b for a, b in zip(x1, x2)]
    x1x3 = [a * b for a, b in zip(x1, x3)]
    x2x3 = [a * b for a, b in zip(x2, x3)]
    x1x2x3 = [a * b * c for a, b, c in zip(x1, x2, x3)]
    x_arr = [x0, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3]

    b_list = factors
    a_list = list(zip(x0, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3))

    N = 8  # test repeat quantity
    list_bi = []  # b(i)
    for k in range(N):
        S = 0
        for i in range(N):
            S += (b_list[k][i] * Y_average[i]) / N
        list_bi.append(round(S, 5))

    disp_dict = {}  # dispersions
    for x in range(1, 9):
        disp_dict["disp{0}".format(x)] = 0
    for i in range(m):
        ctr = 1
        for key, value in disp_dict.items():
            row = y_dict[f'y_row{ctr}']
            disp_dict[key] += ((row[i] - np.average(row)) ** 2) / m
            ctr += 1
    disp_sum = sum(disp_dict.values())
    disp_list = [round(disp, 3) for disp in disp_dict.values()]

    column_names = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "Y1", "Y2", "Y3", "Y", "S^2"]  # назви

    pt = PrettyTable()
    factors.extend([y1, y2, y3, Y_average, disp_list])
    for k in range(len(factors)):
        pt.add_column(column_names[k], factors[k])
    print(pt, "\n")
    # Regression eq with interaction effect
    print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3 \n".format(list_bi[0], list_bi[1],
                                                                                               list_bi[2], list_bi[3],
                                                                                               list_bi[4], list_bi[5],
                                                                                               list_bi[6], list_bi[7]))

    pt = PrettyTable()
    x_arr.extend([y1, y2, y3, Y_average, disp_list])
    for k in range(len(factors)):
        pt.add_column(column_names[k], x_arr[k])
    print(pt, "\n")

    list_ai = [round(i, 5) for i in solve(a_list, Y_average)]
    print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3".format(list_ai[0], list_ai[1],
                                                                                            list_ai[2], list_ai[3],
                                                                                            list_ai[4], list_ai[5],
                                                                                            list_ai[6], list_ai[7]))

    Gp = max(disp_dict.values()) / disp_sum  # experimental
    F1 = m - 1
    N = len(y1)
    F2 = N
    Gt = cochrane_teor(F1, F2)  # theoretical

    print("\nGp = ", Gp, " Gt = ", Gt)
    if cochrane(Gp, Gt):
        print("Dispresion is homogeneous!\n")

        Dispersion_B = disp_sum / N
        Dispersion_beta = Dispersion_B / (m * N)
        S_beta = math.sqrt(abs(Dispersion_beta))

        beta_dict = {}  # beta values
        for x in range(8):
            beta_dict["beta{0}".format(x)] = 0
        for i in range(len(x0_factor)):
            ctr = 0
            for key, value in beta_dict.items():
                beta_dict[key] += (Y_average[i] * factors[ctr][i]) / N
                ctr += 1

        beta_list = list(beta_dict.values())
        t_list = [abs(beta) / S_beta for beta in beta_list]

        F3 = F1 * F2
        d = 0
        T = student_teor(df=F3)
        print("t table = ", T)
        for i in range(len(t_list)):
            if student(t_list[i], T):
                beta_list[i] = 0
                print("Hypothesis has been confirmed, beta{} = 0".format(i))
            else:
                print("Hypothesis was not confirmed.\nbeta{} = {}".format(i, beta_list[i]))
                d += 1

        # y_student = [beta_list[0] + beta_list[1] * x1[i] + beta_list[2] * x2[i] + beta_list[3] * x3[i] + beta_list[4] * x1x2[i] + beta_list[5] * x1x3[i] + beta_list[6] * x2x3[i] + beta_list[7] * x1x2x3[i] for i in range(8)]
        # Optimizing the value list generation above
        factors[0] = None
        y_student = [sum([a * b[x_idx] if b else a for a, b in zip(beta_list, x_arr)]) for x_idx in range(8)]

        F4 = N - d
        Dispersion_ad = 0
        for i in range(len(y_student)):
            Dispersion_ad += ((y_student[i] - Y_average[i]) ** 2) * m / (N - d)
        Fp = Dispersion_ad / Dispersion_beta
        Ft = fischer_teor(dfn=F4, dfd=F3)
        if fischer(Ft, Fp):
            print("Regression equation is adequate.")
            break
        else:
            print("Regression equation is non-adequate.")
            break

    else:
        print("Dispersion is not homogeneous.")
        m += 1
