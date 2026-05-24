import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return -y + x + 1


def y_exact(x):
    return x + np.exp(-x)


def rk4_step(x, y, h):
    """Один крок методу Рунге-Кутта 4-го порядку"""
    k1 = f(x, y)
    k2 = f(x + h / 2.0, y + h * k1 / 2.0)
    k3 = f(x + h / 2.0, y + h * k2 / 2.0)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_constant_step(a, b, y0, h):
    x_vals = np.arange(a, b + h, h)
    y_vals = np.zeros(len(x_vals))
    y_vals[0] = y0

    for i in range(len(x_vals) - 1):
        y_vals[i + 1] = rk4_step(x_vals[i], y_vals[i], h)

    return x_vals, y_vals


def rk4_auto_step(a, b, y0, eps, h0):
    """Метод Рунге-Кутта 4 з автоматичним вибором кроку"""
    x_vals = [a]
    y_vals = [y0]
    h_vals = [h0]

    x = a
    y = y0
    h = h0

    while x < b:
        if x + h > b:
            h = b - x

        y_full_step = rk4_step(x, y, h)
        y_half_step_1 = rk4_step(x, y, h / 2.0)
        y_half_step_2 = rk4_step(x + h / 2.0, y_half_step_1, h / 2.0)

        runge_err = (16.0 / 15.0) * abs(y_full_step - y_half_step_2)

        if runge_err <= eps:
            x += h
            y = y_half_step_2
            x_vals.append(x)
            y_vals.append(y)
            h_vals.append(h)

            if runge_err <= eps / 32.0:
                h *= 2.0
        else:
            h /= 2.0

    return np.array(x_vals), np.array(y_vals), np.array(h_vals)


def adams_pc2_constant_step(a, b, y0, h, eps_iter=1e-6):
    x_vals = np.arange(a, b + h, h)
    y_vals = np.zeros(len(x_vals))
    pred_diffs = np.zeros(len(x_vals))

    y_vals[0] = y0

    if len(x_vals) > 1:
        y_vals[1] = rk4_step(x_vals[0], y_vals[0], h)

    for i in range(1, len(x_vals) - 1):
        x_n = x_vals[i]
        x_next = x_vals[i + 1]

        f_n = f(x_n, y_vals[i])
        f_prev = f(x_vals[i - 1], y_vals[i - 1])

        y_pred = y_vals[i] + (h / 2.0) * (3.0 * f_n - f_prev)

        y_cor_old = y_pred
        y_cor_new = y_pred

        iters = 0
        while iters < 10:
            iters += 1
            y_cor_new = y_vals[i] + (h / 2.0) * (f(x_next, y_cor_old) + f_n)
            if abs(y_cor_new - y_cor_old) <= eps_iter:
                break
            y_cor_old = y_cor_new

        y_vals[i + 1] = y_cor_new
        pred_diffs[i + 1] = abs(y_cor_new - y_pred)

    return x_vals, y_vals, pred_diffs


def adams_auto_step(a, b, y0, eps, h0):
    """Метод Адамса з автоматичним вибором кроку (безпечний алгоритм)"""
    x_vals = [a]
    y_vals = [y0]
    h_vals = [h0]

    x = a
    y = y0
    h = h0

    while x < b:
        if x + h > b:
            h = b - x

        f_n = f(x, y)

        if len(x_vals) >= 2 and abs((x - x_vals[-2]) - h) < 1e-9:
            f_prev = f(x_vals[-2], y_vals[-2])
        else:
            y_prev = rk4_step(x, y, -h)
            f_prev = f(x - h, y_prev)

        y_pred = y + (h / 2.0) * (3.0 * f_n - f_prev)

        y_cor_old = y_pred
        y_cor_new = y_pred

        for _ in range(2):
            y_cor_new = y + (h / 2.0) * (f(x + h, y_cor_old) + f_n)
            y_cor_old = y_cor_new

        err_est = (1.0 / 6.0) * abs(y_cor_new - y_pred)

        if err_est <= eps:
            x += h
            y = y_cor_new
            x_vals.append(x)
            y_vals.append(y)
            h_vals.append(h)

            if err_est <= eps / 10.0:
                h *= 2.0
        else:
            h /= 2.0

    return np.array(x_vals), np.array(y_vals), np.array(h_vals)


def print_results_table(method_name, x_arr, y_arr, h_arr):
    print(f"\n--- {method_name} ---")
    print(f"{'x':>8} | {'y (чисельно)':>15} | {'y (точно)':>15} | {'Крок h':>10}")
    print("-" * 57)

    n_points = len(x_arr)
    for i in range(n_points):
        if i < 5 or i >= n_points - 5:
            y_ex = y_exact(x_arr[i])
            print(f"{x_arr[i]:8.4f} | {y_arr[i]:15.6f} | {y_ex:15.6f} | {h_arr[i]:10.5f}")
        elif i == 5:
            print(f"{'':>8} | {'... (дані на графіках) ...':^33} | {'':>10}")

    print("-" * 57)
    print(f"Загальна кількість точок: {n_points}\n")


def main():
    a_val, b_val = 0.0, 2.0
    y0_val = 1.0
    h_const_val = 0.05
    eps_auto_val = 1e-5

    print("Розпочато розрахунки... Будь ласка, зачекайте.")

    x_adams, y_adams, diffs_adams = adams_pc2_constant_step(a_val, b_val, y0_val, h_const_val)
    err_exact_adams = np.abs(y_adams - y_exact(x_adams))

    x_adams_auto, y_adams_auto, h_adams_auto = adams_auto_step(a_val, b_val, y0_val, eps_auto_val, 0.1)

    x_rk, y_rk = rk4_constant_step(a_val, b_val, y0_val, h_const_val)
    err_exact_rk = np.abs(y_rk - y_exact(x_rk))

    x_rk_auto, y_rk_auto, h_rk_auto = rk4_auto_step(a_val, b_val, y0_val, eps_auto_val, 0.1)

    err_runge = np.zeros(len(x_rk))
    for i in range(1, len(x_rk)):
        y_h = y_rk[i]
        y_h2_1 = rk4_step(x_rk[i - 1], y_rk[i - 1], h_const_val / 2.0)
        y_h2_2 = rk4_step(x_rk[i - 1] + h_const_val / 2.0, y_h2_1, h_const_val / 2.0)
        err_runge[i] = (16.0 / 15.0) * abs(y_h - y_h2_2)

    print("\n" + "=" * 60)
    print(" РЕЗУЛЬТАТИ ОБЧИСЛЕНЬ (АВТОМАТИЧНИЙ КРОК)")
    print("=" * 60)
    print_results_table("Метод Рунге-Кутта 4-го порядку", x_rk_auto, y_rk_auto, h_rk_auto)
    print_results_table("Метод Адамса 2-го порядку", x_adams_auto, y_adams_auto, h_adams_auto)
    print("Графіки зараз відкриються у новому вікні або на панелі SciView...")

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(x_adams, err_exact_adams, label='Справжня похибка', color='blue')
    plt.plot(x_adams, diffs_adams, label='Оцінка похибки |y_cor - y_pred|', linestyle='--', color='red')
    plt.title("Метод Адамса: Локальна похибка")
    plt.xlabel("x")
    plt.ylabel("Похибка")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(x_rk, err_exact_rk, label='Справжня похибка', color='blue')
    plt.plot(x_rk, err_runge, label='Оцінка Рунге', linestyle='--', color='green')
    plt.title("Метод Рунге-Кутта 4: Локальна похибка")
    plt.xlabel("x")
    plt.ylabel("Похибка")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.step(x_adams_auto, h_adams_auto, where='post', label='Метод Адамса', color='red')
    plt.step(x_rk_auto, h_rk_auto, where='post', label='Метод Рунге-Кутта', color='green')
    plt.title(f"Зміна кроку h(x) (eps={eps_auto_val})")
    plt.xlabel("x")
    plt.ylabel("Крок h")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(x_rk, y_exact(x_rk), label='Точний розв\'язок', color='black', linewidth=2)
    plt.plot(x_rk_auto, y_rk_auto, label='РК4 (Авто)', linestyle='--', marker='.')
    plt.plot(x_adams_auto, y_adams_auto, label='Адамс (Авто)', linestyle=':', marker='x')
    plt.title("Отримані розв'язки")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()