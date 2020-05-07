from _pydecimal import Decimal
from scipy.stats import f, t
from random import randrange
from math import sqrt, fabs as fab
from numpy.linalg import solve

dependent, independent = 0, 0

min_x1, max_x1 = -30, 0
min_x2, max_x2 = -35, 10
min_x3, max_x3 = 0, 20

x01 = (max_x1 + min_x1) / 2
x02 = (max_x2 + min_x2) / 2
x03 = (max_x3 + min_x3) / 2

delta_x1 = max_x1 - x01
delta_x2 = max_x2 - x02
delta_x3 = max_x3 - x03

N = 15

matrix_pe = [
    # planning matrix
    [-1, -1, -1, +1, +1, +1, -1, +1, +1, +1],
    [-1, -1, +1, +1, -1, -1, +1, +1, +1, +1],
    [-1, +1, -1, -1, +1, -1, +1, +1, +1, +1],
    [-1, +1, +1, -1, -1, +1, -1, +1, +1, +1],
    [+1, -1, -1, -1, -1, +1, +1, +1, +1, +1],
    [+1, -1, +1, -1, +1, -1, -1, +1, +1, +1],
    [+1, +1, -1, +1, -1, -1, -1, +1, +1, +1],
    [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1],
    [-1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0, 0],
    [+1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0, 0],
    [0, -1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0],
    [0, +1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0],
    [0, 0, -1.73, 0, 0, 0, 0, 0, 0, 2.9929],
    [0, 0, +1.73, 0, 0, 0, 0, 0, 0, 2.9929],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]


def generate_matrix():
    def f(X1, X2, X3):
        # my function
        y = 5.4 + 3.6 * X1 + 6.6 * X2 + 7.7 * X3 + 8.0 * X1 * X1 + 0.3 * X2 * X2 + 2.5 * X3 * X3 + 5.9 * X1 * X2 + \
            0.3 * X1 * X3 + 7.2 * X2 * X3 + 5.3 * X1 * X2 * X3 + randrange(0, 10) - 5
        return y

    matrix_with_y = [[f(matrix_x[j][0], matrix_x[j][1], matrix_x[j][2]) for i in range(m)] for j in range(N)]
    return matrix_with_y


def average_value(arr, orientation):
    # Average values
    average = []
    if orientation == 1:  # Row
        for rows in range(len(arr)):
            average.append(sum(arr[rows]) / len(arr[rows]))
    else:  # Column
        for column in range(len(arr[0])):
            number_arr = []
            for rows in range(len(arr)):
                number_arr.append(arr[rows][column])
            average.append(sum(number_arr) / len(number_arr))
    return average


def cochrane(selectionSize, qty_of_selections, significance):
    selectionSize += 1
    partResult1 = significance / (selectionSize - 1)
    params = [partResult1, qty_of_selections, (selectionSize - 1 - 1) * qty_of_selections]
    fischer = f.isf(*params)
    result = fischer / (fischer + (selectionSize - 1 - 1))
    return Decimal(result).quantize(Decimal('.0001')).__float__()


def student(f3, significance):
    return Decimal(abs(t.ppf(significance / 2, f3))).quantize(Decimal('.0001')).__float__()


def fischer(f3, f4, significance):
    return Decimal(abs(f.isf(significance, f4, f3))).quantize(Decimal('.0001')).__float__()


def x(l1, l2, l3):
    x_1 = l1 * delta_x1 + x01
    x_2 = l2 * delta_x2 + x02
    x_3 = l3 * delta_x3 + x03
    return [x_1, x_2, x_3]


def a(first, second):
    necessary_a = 0
    for j in range(N):
        necessary_a += matrix_x[j][first - 1] * matrix_x[j][second - 1] / N
    return necessary_a


def find_known(number):
    necessary_a = 0
    for j in range(N):
        necessary_a += middle_y[j] * matrix_x[j][number - 1] / 15
    return necessary_a


def check_result(arr_b, k):
    y_i = arr_b[0] + arr_b[1] * matrix[k][0] + arr_b[2] * matrix[k][1] + arr_b[3] * matrix[k][2] + \
          arr_b[4] * matrix[k][3] + arr_b[5] * matrix[k][4] + arr_b[6] * matrix[k][5] + arr_b[7] * matrix[k][6] + \
          arr_b[8] * matrix[k][7] + arr_b[9] * matrix[k][8] + arr_b[10] * matrix[k][9]
    return y_i


def student_test(arr_b, number_x=10):
    dispersion_b = sqrt(dispersion_b2)
    for column in range(number_x + 1):
        t_practice = 0
        t_theoretical = student(f3, q)
        for row in range(N):
            if column == 0:
                t_practice += middle_y[row] / N
            else:
                t_practice += middle_y[row] * matrix_pe[row][column - 1]
        if fab(t_practice / dispersion_b) < t_theoretical:
            arr_b[column] = 0
    return arr_b


def fischer_test():
    dispersion_ad = 0
    f4 = N - d
    for row in range(len(middle_y)):
        dispersion_ad += (m * (middle_y[row] - check_result(student_arr, row))) / (N - d)
    f_practice = dispersion_ad / dispersion_b2
    f_theoretical = fischer(f3, f4, q)
    return f_practice < f_theoretical


m = 3
p = 0.95


matrix_x = [[] for x_val in range(N)]
for i in range(len(matrix_x)):
    if i < 8:
        x_1 = min_x1 if matrix_pe[i][0] == -1 else max_x1
        x_2 = min_x2 if matrix_pe[i][1] == -1 else max_x2
        x_3 = min_x3 if matrix_pe[i][2] == -1 else max_x3
    else:
        arr_x = x(matrix_pe[i][0], matrix_pe[i][1], matrix_pe[i][2])
        x_1, x_2, x_3 = arr_x
    matrix_x[i] = [x_1, x_2, x_3, x_1 * x_2, x_1 * x_3, x_2 * x_3, x_1 * x_2 * x_3, x_1 ** 2, x_2 ** 2, x_3 ** 2]

homogeneity, dispersion_y = False, None
matrix_y = generate_matrix()
middle_x = average_value(matrix_x, 0)
middle_y = average_value(matrix_y, 1)
matrix = [(matrix_x[i] + matrix_y[i]) for i in range(N)]
mx_i = middle_x
my = sum(middle_y) / 15

values = [
    [1, mx_i[0], mx_i[1], mx_i[2], mx_i[3], mx_i[4], mx_i[5], mx_i[6], mx_i[7], mx_i[8], mx_i[9]],
    [mx_i[0], a(1, 1), a(1, 2), a(1, 3), a(1, 4), a(1, 5), a(1, 6), a(1, 7), a(1, 8), a(1, 9), a(1, 10)],
    [mx_i[1], a(2, 1), a(2, 2), a(2, 3), a(2, 4), a(2, 5), a(2, 6), a(2, 7), a(2, 8), a(2, 9), a(2, 10)],
    [mx_i[2], a(3, 1), a(3, 2), a(3, 3), a(3, 4), a(3, 5), a(3, 6), a(3, 7), a(3, 8), a(3, 9), a(3, 10)],
    [mx_i[3], a(4, 1), a(4, 2), a(4, 3), a(4, 4), a(4, 5), a(4, 6), a(4, 7), a(4, 8), a(4, 9), a(4, 10)],
    [mx_i[4], a(5, 1), a(5, 2), a(5, 3), a(5, 4), a(5, 5), a(5, 6), a(5, 7), a(5, 8), a(5, 9), a(5, 10)],
    [mx_i[5], a(6, 1), a(6, 2), a(6, 3), a(6, 4), a(6, 5), a(6, 6), a(6, 7), a(6, 8), a(6, 9), a(6, 10)],
    [mx_i[6], a(7, 1), a(7, 2), a(7, 3), a(7, 4), a(7, 5), a(7, 6), a(7, 7), a(7, 8), a(7, 9), a(7, 10)],
    [mx_i[7], a(8, 1), a(8, 2), a(8, 3), a(8, 4), a(8, 5), a(8, 6), a(8, 7), a(8, 8), a(8, 9), a(8, 10)],
    [mx_i[8], a(9, 1), a(9, 2), a(9, 3), a(9, 4), a(9, 5), a(9, 6), a(9, 7), a(9, 8), a(9, 9), a(9, 10)],
    [mx_i[9], a(10, 1), a(10, 2), a(10, 3), a(10, 4), a(10, 5), a(10, 6), a(10, 7), a(10, 8), a(10, 9), a(10, 10)]
]
known_values = [my, find_known(1), find_known(2), find_known(3), find_known(4), find_known(5), find_known(6),
                find_known(7),
                find_known(8), find_known(9), find_known(10)]

beta = solve(values, known_values)
print("\nRegression equation:")
print("{:.3f} + {:.3f} * X1 + {:.3f} * X2 + {:.3f} * X3 + {:.3f} * Х1X2 + {:.3f} * Х1X3 + {:.3f} * Х2X3"
      "+ {:.3f} * Х1Х2X3 + {:.3f} * X11^2 + {:.3f} * X22^2 + {:.3f} * X33^2 = ŷ\n\nПеревірка"
      .format(beta[0], beta[1], beta[2], beta[3], beta[4], beta[5], beta[6], beta[7], beta[8], beta[9], beta[10]))
for i in range(N):
    print("ŷ{} = {:.3f} ≈ {:.3f}".format((i + 1), check_result(beta, i), middle_y[i]))

while not homogeneity:
    print("\n")
    print(" " * 65 + "Planning matrix" + " " * 65)
    print("      X1           X2           X3          X1X2        X1X3         X2X3         X1X2X3       X1X1"
          "         X2X2         X3X3          Yi ...")
    for row in range(N):
        print(" ", end=' ')
        for column in range(len(matrix[0])):
            print("{:^12.3f}".format(matrix[row][column]), end=' ')
        print(" ")
    print("\n")
    dispersion_y = [0.0 for x in range(N)]
    for i in range(N):
        dispersion_i = 0
        for j in range(m):
            dispersion_i += (matrix_y[i][j] - middle_y[i]) ** 2
        dispersion_y.append(dispersion_i / (m - 1))
    f1 = m - 1
    f2 = N
    f3 = f1 * f2
    q = 1 - p
    Gp = max(dispersion_y) / sum(dispersion_y)
    print("Cochrane criterion")
    Gt = cochrane(f2, f1, q)
    if Gt > Gp and m < 25:
        print("Dispersion is homogeneous with q = {:.2f}!\n".format(q))
        homogeneity = True
    else:
        print("Dispersion is not homogeneous. Retrying with m += 1...")
        m += 1
    if m == 25:
        exit()


dispersion_b2 = sum(dispersion_y) / (N * N * m)
student_arr = list(student_test(beta))
print("Regression equation according to Student criterion")
print("{:.3f} + {:.3f} * X1 + {:.3f} * X2 + {:.3f} * X3 + {:.3f} * Х1X2 + {:.3f} * Х1X3 + {:.3f} * Х2X3"
      "+ {:.3f} * Х1Х2X3 + {:.3f} * X11^2 + {:.3f} * X22^2 + {:.3f} * X33^2 = ŷ\n\nПеревірка"
      .format(student_arr[0], student_arr[1], student_arr[2], student_arr[3], student_arr[4], student_arr[5],
              student_arr[6], student_arr[7], student_arr[8], student_arr[9], student_arr[10]))
for i in range(N):
    print("ŷ{} = {:.3f} ≈ {:.3f}".format((i + 1), check_result(student_arr, i), middle_y[i]))

d = 11 - student_arr.count(0)
dependent += N - d
independent += d
if fischer_test():
    print('\n\033[93mMath model is adequate to experimental data.')
else:
    print('\n\033[93mMath model is non-adequate to experimental data.')