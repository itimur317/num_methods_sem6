import matplotlib.pyplot as plt
import numpy as np

h_step = 200
tau_step = 200

N = 1
alpha = 1
K = 1

U0 = 1
chi = 1


def V0():
    return np.power(U0, N)


def h():
    return 1 / h_step


def tau():
    return 1 / tau_step


def lambda_N(arr):
    return np.power(arr, N)


def mu(n):
    t = n * tau()
    return U0 * np.power(t, K)


def f(z):
    return alpha * N * (alpha - z)


def U(n, k):
    t = n * tau()
    x = k * h()
    m = N * K
    z = x / np.power(chi * V0() * np.power(t, m + 1), 0.5)
    if x >= 0 and x <= alpha * np.power(chi * V0() * np.power(t, m + 1), 0.5):
        return U0 * np.power(t, K) * np.power(f(z), 1 / N)
    else:
        return 0


def get_iterated_next_line(cur_line, n, h_step):
    next_line = np.zeros(h_step + 1)
    for i in range(0, 4):
        if i == 0:
            next_line = get_next_line(cur_line, cur_line, n, h_step)
        else:
            tmp = next_line
            next_line = get_next_line(cur_line, tmp, n, h_step)
    return next_line


def get_next_line(cur_line, next_line_info, n, h_step):
    next_line = np.zeros(h_step + 1)
    cur_lambda_N = lambda_N(next_line_info)

    a = np.zeros(h_step + 1)
    b = np.zeros(h_step + 1)
    c = np.zeros(h_step + 1)
    d = np.zeros(h_step + 1)

    # a[0] = 0
    b[0] = 1
    # c[0] = 0
    d[0] = mu(n + 1)

    # a[h_step] = 0
    b[h_step] = 1
    # c[h_step] = 0
    d[h_step] = 0

    for k in range(1, h_step):
        factor1 = cur_lambda_N[k + 1] + cur_lambda_N[k]
        factor2 = cur_lambda_N[k] + cur_lambda_N[k - 1]

        a[k] = factor2
        b[k] = - factor1 - factor2 - 2 * h() * h() / tau()
        c[k] = factor1
        d[k] = - 2 * h() * h() / tau() * cur_line[k]

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


def error_graphic(time_in_n, initial_cond, h_step):
    next_line = initial_cond
    for n in range(time_in_n):
        if n == 0:
            next_line = get_iterated_next_line(initial_cond, 0, h_step)
        else:
            tmp = np.copy(next_line)
            next_line = get_iterated_next_line(tmp, n, h_step)

    delta_arr = np.zeros(h_step + 1)

    for k in range(h_step + 1):
        delta_arr[k] = abs(next_line[k] - U(time_in_n, k))

    points = np.linspace(0, 1, h_step + 1)

    plt.title("Погр " + " t=" + str(time_in_n * tau())[:5] + " мин h=" + str(h()) + ", " + "tau =" + str(tau()))
    plt.xlabel("x")
    plt.ylabel("delta(u)")
    plt.plot(points, delta_arr, alpha=0.7, label="first", lw=5, mec='b', mew=2, ms=10)
    plt.show()


h_step = 20
tau_step = 20

initial_cond = np.zeros(h_step + 1)


# вызывая функцию error_graphic(4, initial_conditions, h_step) рассматривается момент времени
# t = 4 / 20 минуты = 0.2 минуты
# аналогично при шаге tau_step = 200
# t = 40 / 200 = 0.2 минуты

error_graphic(4, initial_cond, h_step)
error_graphic(8, initial_cond, h_step)
error_graphic(12, initial_cond, h_step)
error_graphic(16, initial_cond, h_step)


