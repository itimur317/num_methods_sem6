import matplotlib.pyplot as plt
import numpy as np


def h():
    return 10 / h_step


def tau():
    return 5 / tau_step


def f(arr, k, var):
    x = k * h()
    if var == 1:
        return arr[k]
    elif var == 2:
        return arr[k] * arr[k]
    elif var == 3:
        return arr[k]**2 * 0.5


def a1(u):
    a = np.zeros(len(u))
    for i in range(len(u)):
        a[i] = 1
    return a


def a2(u):
    a = np.copy(u)
    return a


def a3(u):
    return 0.5 * u


def check_curant(next_line, n, var):
    if var == 1:
        maximum = max(a1(next_line))
    elif var == 2:
        maximum = max(a2(next_line))
    elif var == 3:
        maximum = max(a3(next_line))

    C = np.power(tau(), n) * maximum / h()

    if C <= 1 and n != 0:
        return 1
    return 0


def get_next_line(cur_line, h_step, var):
    next_line = np.zeros(h_step + 1)

    next_line[0] = 0
    next_line[-1] = 0

    for k in range(1, h_step):
        term1 = f(cur_line, k + 1, var)
        term2 = f(cur_line, k - 1, var)
        next_line[k] = 1 / 2 * (cur_line[k - 1] + cur_line[k + 1] -
                        tau() / h() * (term1 - term2))

    return next_line


def graphic(time_in_n, initial_cond, h_step, var):
    next_line = initial_cond
    for n in range(time_in_n):
        if n == 0:
            next_line = get_next_line(initial_cond, h_step, var)
        else:
            tmp = np.copy(next_line)
            next_line = get_next_line(tmp, h_step, var)

    if not check_curant(next_line, time_in_n, var) and time_in_n != 1:
        print("неустойчива", time_in_n)

    points = np.linspace(0, 10, h_step + 1)

    plt.title("Погр " + " t=" + str(time_in_n * tau())[:5] + " мин h=" + str(h()) + ", " + "tau =" + str(tau()))
    plt.xlabel("x")
    plt.ylabel("delta(u)")
    plt.plot(points, next_line, alpha=0.7, label="first", lw=5, mec='b', mew=2, ms=10)
    plt.show()


def phi1():
    arr = np.linspace(0, 10, h_step + 1)
    for k in range(h_step + 1):
        if arr[k] <= 2 and arr[k] >= 1:
            arr[k] = 1
        else:
            arr[k] = 0
    return arr


def phi2():
    arr = np.linspace(0, 10, h_step + 1)
    for k in range(h_step + 1):
        if k * h() > 2:
            arr[k] = 0
        else:
            sin = np.sin(arr[k] * np.pi / 2)
            arr[k] = np.power(sin, 2)
    return arr


def phi3():
    arr = np.linspace(0, 10, h_step + 1)
    for k in range(h_step + 1):
        if k * h() > 4 or k * h() < 2:
            arr[k] = 0
        else:
            cos = np.cos(arr[k] * np.pi / 2)
            arr[k] = abs(cos)
    return arr


# вызывая функцию graphic(100, initial_cond1, h_step, 1)
# рассматривается момент времени
# t = 100 / 200 минуты = 0.5 минуты

tau_step = 1000
h_step = 2000

# для линейного переноса прямоуг профиля
initial_cond1 = list(phi1())
# graphic(100, initial_cond1, h_step, 1)
# graphic(250, initial_cond1, h_step, 1)

tau_step = 1000
h_step = 1000

# для нелинейного переноса с разрывным решением
initial_cond2 = list(phi2())
# graphic(1, initial_cond2, h_step, 2)
# graphic(20, initial_cond2, h_step, 2)
# graphic(50, initial_cond2, h_step, 2)
# graphic(100, initial_cond2, h_step, 2)
# graphic(200, initial_cond2, h_step, 2)
# graphic(500, initial_cond2, h_step, 2)
# graphic(1000, initial_cond2, h_step, 2)
# graphic(2000, initial_cond2, h_step, 2)


# tau_step = 200
# h_step = 50
# phi(x) = cos(pi * x / 2) * heaviside(x - 2) * heaviside(4 - x)
# a(u) = 0.5 * u
initial_cond3 = list(phi3())
graphic(1, initial_cond3, h_step, 3)
graphic(20, initial_cond3, h_step, 3)
graphic(50, initial_cond3, h_step, 3)
graphic(100, initial_cond3, h_step, 3)
graphic(200, initial_cond3, h_step, 3)
graphic(500, initial_cond3, h_step, 3)
graphic(1000, initial_cond3, h_step, 3)
graphic(2000, initial_cond3, h_step, 3)
graphic(3000, initial_cond3, h_step, 3)


