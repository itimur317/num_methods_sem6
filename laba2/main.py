import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go

# func and const
alpha1 = 1
beta1 = 0
alpha2 = 1
beta2 = 0
a_square = 1


def h():
    return 1 / h_step


def tau():
    return 1 / tau_step


def f(n, k):
    return 0
    # x = k * h()
    # t = n * tau()
    # return np.exp(x) / 2 * (
    #     1 / np.power(np.cos(t), 2) - np.tan(t)
    # )
    # return 2 * x * t + (1 + np.tanh(x - t) - 2 * np.tanh(x - t) * np.tanh(x - t)) / np.cosh(x - t)


def u0(n, k):
    x = k * h()
    t = n * tau()
    return (np.exp(x) * np.tan(t) - x) / 2
    # return 1 / np.cosh(x - t) + x * t**2


def phi(k):
    if k >= h_step / 2:
        return 1
    return 0
    # x = k * h()
    # return -x / 2
    # return 1 / np.cosh(x)


def gamma1(n):
    return 0
    # t = n * tau()
    # return np.tan(t) - 0.5
    # return t**2 + (1 + np.tanh(t)) / np.cosh(t)


def gamma2(n):
    return 0
    # t = n * tau()
    # return (np.exp(1) * np.tan(t) - 1) / 2
    # return t**2 + 1 / np.cosh(1 - t)


def get_next_line(cur_line, n, h_step, tau_step):
    next_line = np.zeros(h_step + 1)

    a = np.zeros(h_step + 1)
    b = np.zeros(h_step + 1)
    c = np.zeros(h_step + 1)
    d = np.zeros(h_step + 1)

    # a[0] = 0
    b[0] = -1
    c[0] = 1
    d[0] = 0

    a[h_step] = -1
    b[h_step] = 1
    # c[h_step] = 0
    d[h_step] = 0

    for k in range(1, h_step):
        if k >= h_step / 2:
            a_square = 1
        elif k <= h_step / 2:
            a_square = 1.2
        a[k] = a_square * tau() / h()**2 / 2
        b[k] = -a_square * tau() / h()**2 - 1
        c[k] = a_square * tau() / h()**2 / 2
        d[k] = (-a_square * tau() / 2 / h()**2 * (cur_line[k + 1] - 2 * cur_line[k] + cur_line[k - 1]) -
                cur_line[k] - tau() * f(n, k + 0.5))

    A = np.zeros(h_step + 1)
    B = np.zeros(h_step + 1)

    A[0] = -c[0] / b[0]
    B[0] = d[0] / b[0]

    for k in range(1, h_step + 1):
        A[k] = -c[k] / (b[k] + a[k] * A[k - 1])
        B[k] = (d[k] - a[k] * B[k - 1]) / (b[k] + a[k] * A[k - 1])

    next_line[h_step] = B[h_step]

    for k in range(h_step - 1, -1, -1):
        next_line[k] = B[k] + A[k] * next_line[k + 1]

    return next_line


def error_graphic(time_in_n, initial_conditions,  h_step, tau_step):
    next_line = initial_conditions
    for n in range(time_in_n):
        if n == 0:
            next_line = get_next_line(initial_conditions, 0, h_step, tau_step)
        else:
            tmp = np.copy(next_line)
            next_line = get_next_line(tmp, n, h_step, tau_step)

    array_for_graphic = np.zeros((2, h_step + 1))

    for k in range(h_step + 1):
        array_for_graphic[0][k] = k * h()
        array_for_graphic[1][k] = next_line[k]

    plt.title("Погр " + " t=" + str(time_in_n * tau())[:5] + " мин h=" + str(h()) + ", " + "tau =" + str(tau()))
    plt.xlabel("x")
    plt.ylabel("delta(u)")
    plt.plot(array_for_graphic[0], array_for_graphic[1], alpha=0.7, label="first", lw=5, mec='b', mew=2, ms=10)
    plt.show()


# def error_3d(initial_conditions,  h_step, tau_step):
#     next_line = initial_conditions
#
#     z = np.zeros((tau_step + 1, h_step + 1))
#     for n in range(tau_step + 1):
#         if n == 0:
#             next_line = get_next_line(initial_conditions, 0, h_step, tau_step)
#             z[n] = initial_conditions
#         else:
#             z[n] = next_line
#             tmp = np.copy(next_line)
#             next_line = get_next_line(tmp, n, h_step, tau_step)
#
#     x = np.linspace(0, 1, h_step + 1)
#     y = np.linspace(0, 1, tau_step + 1)
#     X, Y = np.meshgrid(x, y, indexing='ij')
#
#     fig = go.Figure([go.Surface(x = X, y = Y, z = z )])
#     fig.update_layout(title='3D модель Галимов', autosize=True,
#                   width=700, height=700,
#                    margin=dict(l=100, r=100, b=100, t=100))
#     fig.write_html('empty.html', auto_open=True)


# h = 0.05,   h_step <=> M

h_step = 100
# tau = 0.05,   tau_step <=> K
tau_step = 1000
# начальные условия
# initial_conditions = phi(np.arange(0, h_step + 1, 1))
# error_graphic(15, initial_conditions, h_step, tau_step)


# h_step = 200
# tau_step = 200
# начальные условия
# initial_conditions = phi(np.arange(0, h_step + 1, 1))
a = np.arange(0, h_step + 1, 1)
initial_conditions = np.where(a >= h_step / 2, 1, 0)
print(initial_conditions)

error_graphic(50, initial_conditions, h_step, tau_step)


# вызывая функцию error_graphic(4, initial_conditions, h_step, tau_step) рассматривается момент времени
# t = 4 / 20 минуты = 0.2 минуты
# аналогично при шаге h_step = 200
# t = 40 / 200 = 0.2

# точность схемы О(h + tau)
# погрешность на правом равна нулю в силу того, что рассматривается граничное условие первого рода
