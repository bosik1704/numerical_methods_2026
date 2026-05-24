import math
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# ЧАСТИНА 1: Цільові функції
# ==========================================

# 1. Функція Розенброка
def rosenbrock(X):
    x1, x2 = X[0], X[1]
    return 100 * (x1 ** 2 - x2) ** 2 + (x1 - 1) ** 2


# 2. Система нелінійних рівнянь
def system_equations(X):
    x1, x2 = X[0], X[1]
    f1 = x1 ** 2 + x2 ** 2 - 4
    f2 = x2 - x1 ** 2
    return f1 ** 2 + f2 ** 2


# ==========================================
# ЧАСТИНА 2: Метод Хука-Дживса
# ==========================================

def hooke_jeeves(func, start_point, start_steps, q=2.0, p=2.0, eps1=1e-6, eps2=1e-6, filename="trajectory.txt"):
    n = len(start_point)
    X_base = list(start_point)
    delta_X = list(start_steps)
    trajectory = [list(X_base)]

    def explore(current_base, current_deltas, allow_reduction):
        X_new = list(current_base)
        for i in range(n):
            while True:
                # Крок вперед
                X_plus = list(X_new)
                X_plus[i] += current_deltas[i]
                if func(X_plus) < func(X_new):
                    X_new = list(X_plus)
                    break

                # Крок назад
                X_minus = list(X_new)
                X_minus[i] -= current_deltas[i]
                if func(X_minus) < func(X_new):
                    X_new = list(X_minus)
                    break

                # Зменшення кроку
                if allow_reduction:
                    current_deltas[i] /= q
                    if current_deltas[i] < eps1:
                        break
                else:
                    break
        return X_new

    iters = 0
    while iters < 1000:
        iters += 1

        # Досліджуючий пошук
        X1 = explore(X_base, delta_X, allow_reduction=True)
        trajectory.append(list(X1))

        if X1 == X_base:
            break

        max_dx = max(delta_X)
        diff_phi = abs(func(X1) - func(X_base))
        if max_dx < eps1 and diff_phi < eps2:
            break

        # Пошук по зразку
        while True:
            X_p = [X1[i] + p * (X1[i] - X_base[i]) for i in range(n)]
            X2 = explore(X_p, delta_X, allow_reduction=False)

            if func(X2) < func(X1):
                X_base = list(X1)
                X1 = list(X2)
                trajectory.append(list(X1))
            else:
                X_base = list(X1)
                break

                # Запис траєкторії в файл
    with open(filename, 'w') as f:
        f.write("Крок\tx1\t\tx2\t\tPhi(X)\n")
        for idx, pt in enumerate(trajectory):
            f.write(f"{idx}\t{pt[0]:.6f}\t{pt[1]:.6f}\t{func(pt):.10f}\n")

    return X1, len(trajectory) - 1, np.array(trajectory)


# ==========================================
# ЧАСТИНА 3: Функція для побудови графіків
# ==========================================
def plot_trajectory(func, trajectory, x_bounds, y_bounds, title):
    x = np.linspace(x_bounds[0], x_bounds[1], 400)
    y = np.linspace(y_bounds[0], y_bounds[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])

    plt.figure(figsize=(8, 6))

    # Створюємо логарифмічні лінії рівня, щоб краще було видно "ямку"
    if np.max(Z) > 10:
        levels = np.logspace(-2, np.log10(np.max(Z)), 30)
    else:
        levels = 30

    plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5)

    # Малюємо саму траєкторію
    traj_x = trajectory[:, 0]
    traj_y = trajectory[:, 1]

    plt.plot(traj_x, traj_y, 'ro-', markersize=4, label='Траєкторія спуску')
    plt.plot(traj_x[0], traj_y[0], 'go', markersize=8, label='Старт (Початкове наближення)')
    plt.plot(traj_x[-1], traj_y[-1], 'b*', markersize=12, label="Мінімум (Розв'язок)")

    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()


# ==========================================
# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
# ==========================================
if __name__ == "__main__":
    eps1 = 1e-5
    eps2 = 1e-5

    # ТЕСТ 1: Функція Розенброка
    print("=== ТЕСТ: Функція Розенброка ===")
    start_rosen = [-1.2, 0.0]
    steps_rosen = [0.5, 0.5]

    res_rosen, steps_r, traj_rosen = hooke_jeeves(
        func=rosenbrock,
        start_point=start_rosen,
        start_steps=steps_rosen,
        eps1=eps1, eps2=eps2,
        filename="trajectory_rosenbrock.txt"
    )
    print(f"Початкова точка: {start_rosen}")
    print(f"Знайдений мінімум: x1 = {res_rosen[0]:.6f}, x2 = {res_rosen[1]:.6f}")
    print(f"Значення функції: {rosenbrock(res_rosen):.10f}")
    print(f"Кількість кроків траєкторії: {steps_r}\n")

    # Малюємо графік для функції Розенброка
    plot_trajectory(rosenbrock, traj_rosen, [-1.5, 1.5], [-0.5, 1.5], "Метод Хука-Дживса: Функція Розенброка")

    # ТЕСТ 2: Розв'язок системи рівнянь
    print("=== СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ ===")
    start_sys = [1.0, 1.0]
    steps_sys = [0.5, 0.5]

    res_sys, steps_s, traj_sys = hooke_jeeves(
        func=system_equations,
        start_point=start_sys,
        start_steps=steps_sys,
        eps1=eps1, eps2=eps2,
        filename="trajectory_system.txt"
    )
    print(f"Початкова точка: {start_sys}")
    print(f"Знайдений розв'язок: x1 = {res_sys[0]:.6f}, x2 = {res_sys[1]:.6f}")
    print(f"Значення цільової функції Phi: {system_equations(res_sys):.10f}")
    print(f"Кількість кроків: {steps_s}")

    # Малюємо графік для системи рівнянь
    plot_trajectory(system_equations, traj_sys, [0.0, 2.0], [0.0, 2.5], "Метод Хука-Дживса: Система рівнянь")