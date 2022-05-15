from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np

hx_step = 20
hy_step = 20
tau_step = 500


def hx():
    return 2 / hx_step


def hy():
    return 2 / hy_step


def tau():
    return 1 / tau_step


def stability():
    if tau() > 1 / 2 * min(hy() * hy(), hx() * hx()):
        print("неуст")


def Q(i, j, var):
    # Фигура плюс из 5 точек по центру
    if var == 1:
        half_y = hy_step // 2
        half_x = hx_step // 2
        if i == half_x and (j == half_y or j == half_y - 1 or j == half_y + 1):
            return 0.2
        if j == half_y and (i == half_x + 1 or i == half_x - 1):
            return 0.2
    # Полоса по центру параллельно Y
    elif var == 2:
        if j == 10:
            return 1 / 21
    # Полоса по центру параллельно X
    elif var == 3:
        if i == 10:
            return 1 / 21
    # Точечный источник
    elif var == 4:
        half_y = hy_step // 2
        half_x = hx_step // 2
        if i == half_x and j == half_y:
            return 1
    # Фигура плюс(большая)
    elif var == 5:
        half_y = hy_step // 2
        half_x = hx_step // 2
        if i == half_x or j == half_y:
            return 1 / (hy_step + hx_step - 1)
    # Напротив двери(поближе)
    elif var == 6:
        half_x = hx_step // 2
        if j == 5 and (i == half_x + 4 or i == half_x + 3 or i == half_x + 1 or i == half_x or
        i == half_x - 4 or i == half_x - 3 or i == half_x - 2 or i == half_x - 1):
            return 1 / 9
    # Напротив двери(подальше)
    elif var == 7:
        half_x = hx_step // 2
        half_y = hy_step // 2
        if j == half_y and (i == half_x + 4 or i == half_x + 3 or i == half_x + 1 or i == half_x or
                       i == half_x - 4 or i == half_x - 3 or i == half_x - 2 or i == half_x - 1):
            return 1 / 9
    # Напротив двери(у стены)
    elif var == 8:
        half_x = hx_step // 2
        if j == hy_step - 1 and (i == half_x + 7 or i == half_x + 6 or i == half_x + 5 or i == half_x + 4 or i == half_x + 3 or
        i == half_x + 2 or i == half_x + 1 or i == half_x or i == half_x - 1 or i == half_x - 2 or i == half_x - 3
        or i == half_x - 4 or i == half_x - 5 or i == half_x - 6 or i == half_x - 7):
            return 1 / 15
    # 2 батареи в углу(угол с окном, дальний от двери)
    elif var == 9:
        y_bool = (j == hy_step - 1 or j == hy_step - 2 or j == hy_step - 3 or j == hy_step - 4) and i == hx_step - 1
        x_bool = (i == hx_step - 1 or i == hx_step - 2 or i == hx_step - 3 or i == hx_step - 4) and j == hy_step - 1
        if x_bool or y_bool:
            return 1 / 8
    # Батарея под окном
    elif var == 10:
        if (i == hx_step - 1 and (j == hy_step * 3 // 4 - 4 or j == hy_step * 3 // 4 - 3 or j == hy_step * 3 // 4 - 2
                                 or j == hy_step * 3 // 4 - 1 or j == hy_step * 3 // 4 or j == hy_step * 3 // 4 + 1
                                  or j == hy_step * 3 // 4 + 2 or j == hy_step * 3 // 4 + 3 or j == hy_step * 3 // 4 + 4)):
            return 1 / 9
    # 2 батареи рядом с дверью
    elif var == 11:
        if j == 1 and (i == hx_step // 4 - 2 or i == hx_step // 4 - 3
                       or i == hx_step * 3 // 4 + 2 or i == hx_step * 3 // 4 + 3):
            return 1 / 4
    # 2 батареи в углу(дальний угол от батареи и двери)
    elif var == 12:
        y_bool = i == 1 and (j == hy_step - 4 or j == hy_step - 3 or j == hy_step - 2 or j == hy_step - 1)
        x_bool = j == hy_step and (i == 1 or i == 2 or i == 3 or i == 4)
        if x_bool or y_bool:
            return 1 / 8
    return 0


# get next element
def get_next(cur_layer, i, j, var):
    term1 = (cur_layer[i + 1][j] - 2 * cur_layer[i][j] + cur_layer[i - 1][j]) / hx() / hx()
    term2 = (cur_layer[i][j + 1] - 2 * cur_layer[i][j] + cur_layer[i][j - 1]) / hy() / hy()
    return tau() * (term1 + term2 + Q(i, j, var)) + cur_layer[i][j]


def get_next_layer(cur_layer, var):
    next_layer = np.zeros((hx_step + 1, hy_step + 1))

    for i in range(1, hx_step):
        for j in range(1, hy_step):
            next_layer[i][j] = get_next(cur_layer, i, j, var)

    # boundary conditions x
    for i in range(hx_step + 1):
        # top:
        next_layer[i][hy_step] = next_layer[i][hy_step - 1]

        # bottom:
        x = i * hx()
        if x < 0.5 or x > 1.5:
            next_layer[i][0] = next_layer[i][1]
        else:
            next_layer[i][0] = 0

    # boundary conditions y
    for j in range(hy_step + 1):
        # left
        next_layer[0][j] = next_layer[1][j]

        # right
        y = j * hy()
        if y < 1.2 or y > 1.8:
            next_layer[hx_step][j] = next_layer[hx_step - 1][j]
        else:
            next_layer[hx_step][j] = 1 / (1 + hy()) * next_layer[hx_step - 1][j]
    return next_layer


def get_static_layer(initial_cond, var):
    layer = get_next_layer(initial_cond, var)

    for i in range(10000):
        tmp = layer
        layer = get_next_layer(tmp, var)

    return layer


def graphic(layer, var, name_):
    # create 3d axes
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = np.linspace(0, 2, hx_step + 1)
    y = np.linspace(0, 2, hy_step + 1)

    X, Y = np.meshgrid(x, y)
    Z = layer

    fig = go.Figure([go.Surface(x=X, y=Y, z=Z)])
    name = "Q" + str(var) + " " + name_
    name_html = name + ".html"
    fig.update_layout(title= name, autosize=True,
                  width=700, height=700,
                   margin=dict(l=100, r=100, b=100, t=100))
    fig.write_html(name_html, auto_open=True)


def maximum(layer):
    return np.amax(layer)


def average(layer):
    return np.average(layer)


# u(x, y, 0) = 0
initial_cond = np.zeros((hx_step + 1, hy_step + 1))
stability()

best_max = [0, 0]
best_average = [0, 0]

for i in range(1, 13):
    var = i
    if var == 1:
        name = "Фигура плюс из 5 точек по центру"
    elif var == 2:
        name = "Полоса по центру параллельно Y"
    elif var == 3:
        name = "Полоса по центру параллельно X"
    elif var == 4:
        name = "Точечный источник"
    elif var == 5:
        name = "Фигура плюс(большая)"
    elif var == 6:
        name = "Напротив двери(поближе)"
    elif var == 7:
        name = "Напротив двери(подальше)"
    elif var == 8:
        name = "Напротив двери(у стены)"
    elif var == 9:
        name = "2 батареи в углу(угол с окном, дальний от двери)"
    elif var == 10:
        name = "Батарея под окном"
    elif var == 11:
        name = "2 батареи рядом с дверью"
    elif var == 12:
        name = "2 батареи в углу(дальний угол от батареи и двери)"
    else:
        name = "ошибка"
    static_layer = get_static_layer(initial_cond, var)
    graphic(static_layer, var, name)
    print("Q вариант: ", var)
    print(name, "max", maximum(static_layer))
    print(name, "average", average(static_layer), "\n")

    if maximum(static_layer) > best_max[0]:
        best_max[0] = maximum(static_layer)
        best_max[1] = var

    if average(static_layer) > best_average[0]:
        best_average[0] = average(static_layer)
        best_average[1] = var


print("Наибольшее среднее: ", best_average[0], "Q вариант: ", best_average[1])
print("Наибольший максимум: ", best_max[0], "Q вариант: ", best_max[1])
