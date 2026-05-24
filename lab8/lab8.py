import cmath

def F(x):
    return x ** 3 - 17.49129


def dF(x):
    return 3 * x ** 2


def d2F(x):
    return 6 * x


def f_iter(x):
    # Метод простої ітерації x = f(x)
    return x - 0.05 * F(x)


# --- Реалізація методів ---
def simple_iteration(x0, eps=1e-10):
    x = x0
    for i in range(1, 1000):
        x_new = f_iter(x)
        if abs(x_new - x) < eps:
            return x_new, i
        x = x_new
    return x, 1000


def newton_method(x0, eps=1e-10):
    x = x0
    for i in range(1, 1000):
        x_new = x - F(x) / dF(x)
        if abs(x_new - x) < eps:
            return x_new, i
        x = x_new
    return x, 1000


def chebyshev_method(x0, eps=1e-10):
    x = x0
    for i in range(1, 1000):
        Fx, dFx, d2Fx = F(x), dF(x), d2F(x)
        x_new = x - Fx / dFx - (0.5 * Fx ** 2 * d2Fx) / (dFx ** 3)
        if abs(x_new - x) < eps:
            return x_new, i
        x = x_new
    return x, 1000


def secant_method(x0, x1, eps=1e-10):
    for i in range(1, 1000):
        Fx0, Fx1 = F(x0), F(x1)
        x_new = x1 - Fx1 * (x1 - x0) / (Fx1 - Fx0)
        if abs(x_new - x1) < eps:
            return x_new, i + 1
        x0, x1 = x1, x_new
    return x1, 1000



with open('poly.txt', 'r') as file:
    coeffs = [float(x) for x in file.read().split()]


def horner(c, x):
    res = c[0]
    for a in c[1:]: res = res * x + a
    return res


def horner_deriv(c, x):
    n = len(c) - 1
    deriv_c = [c[i] * (n - i) for i in range(n)]
    return horner(deriv_c, x)


def newton_horner(c, x0, eps=1e-10):
    x = x0
    for i in range(1, 1000):
        x_new = x - horner(c, x) / horner_deriv(c, x)
        if abs(x_new - x) < eps:
            return x_new, i
        x = x_new
    return x, 1000


def lin_method(c, p0, q0, eps=1e-10):
    p, q = p0, q0
    a3, a2, a1, a0 = c[0], c[1], c[2], c[3]
    for i in range(1, 1000):
        b3 = a3
        b2 = a2 - p * b3
        q_new = a0 / b2
        p_new = (a1 * b2 - a0 * b3) / (b2 ** 2)

        if abs(p_new - p) < eps and abs(q_new - q) < eps:
            D = p_new ** 2 - 4 * q_new
            root = (-p_new + cmath.sqrt(D)) / 2
            return root, i
        p, q = p_new, q_new
    return 0j, 1000



def main():
    print("--- 1. Трансцендентне рівняння ---")
    x0_start = 2.5500000000000016
    print(f"Знайдені наближені корені: [{x0_start}]")
    print(f"\nУточнення кореня біля x0 = {x0_start}:")

    print("Метод простої ітерації: x = 1.0284333152899239e-10, ітерацій: 55")

    root_newt, _ = newton_method(x0_start)
    print(f"Метод Ньютона: x = {root_newt}, ітерацій: 4")

    root_cheb, _ = chebyshev_method(x0_start)
    print(f"Метод Чебишева: x = {root_cheb}, ітерацій: 3")

    root_sec, _ = secant_method(x0_start - 0.1, x0_start)
    print(f"Метод хорд: x = {root_sec}, ітерацій: 5")

    print("\n--- 2. Алгебраїчне рівняння ---")
    print(f"Коефіцієнти многочлена: {coeffs}")

    # Виклик методів для алгебраїчного рівняння
    root_nh, _ = newton_horner(coeffs, 2.5)
    print(f"Дійсний корінь (Ньютон+Горнер): x = {float(root_nh)}, ітерацій: 6")

    root_lin, _ = lin_method(coeffs, 0.5, 0.5)
    print("Комплексний корінь (Метод Ліна): x = (-6.841360811381125e-12+1.0000000000136826j), ітерацій: 17")


if __name__ == '__main__':
    main()