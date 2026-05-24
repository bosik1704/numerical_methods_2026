import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


a, b = 0, 24
I0, _ = quad(f, a, b)
print(f"2. Точне значення інтегралу I0: {I0:.12f}\n")


def simpson(f, a, b, N):
    if N % 2 != 0:
        raise ValueError("N має бути парним для методу Сімпсона")

    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (h / 3) * S


N_values = np.arange(10, 1002, 2)
errors = []

N_opt = None
eps_opt = None
target_accuracy = 1e-12

for N in N_values:
    I_N = simpson(f, a, b, N)
    error = abs(I_N - I0)
    errors.append(error)

    if error <= target_accuracy and N_opt is None:
        N_opt = N
        eps_opt = error

print(f"4. Задана точність 1e-12 досягається при N_opt = {N_opt}")
print(f"   Похибка eps_opt = {eps_opt:.2e}\n")

N0_raw = N_opt // 10
N0 = N0_raw + (8 - N0_raw % 8) if N0_raw % 8 != 0 else N0_raw
if N0 < 8:
    N0 = 8

I_N0 = simpson(f, a, b, N0)
eps0 = abs(I_N0 - I0)
print(f"5. Вибрано N0 = {N0} (кратне 8)")
print(f"   Інтеграл I(N0) = {I_N0:.12f}")
print(f"   Похибка eps0 = {eps0:.2e}\n")

I_N0_half = simpson(f, a, b, N0 // 2)
I_R = I_N0 + (I_N0 - I_N0_half) / 15
eps_R = abs(I_R - I0)

print(f"6. Метод Рунге-Ромберга:")
print(f"   Уточнене значення I_R = {I_R:.12f}")
print(f"   Похибка epsR = {eps_R:.2e}")
print(f"   Зменшення похибки у {eps0 / eps_R:.1f} разів\n")


I_N0_quarter = simpson(f, a, b, N0 // 4)


I1, I2, I3 = I_N0, I_N0_half, I_N0_quarter

num_E = I2 ** 2 - I1 * I3
den_E = 2 * I2 - (I1 + I3)
I_E = num_E / den_E
eps_E = abs(I_E - I0)


val_p = abs((I3 - I2) / (I2 - I1))
p = (1 / np.log(2)) * np.log(val_p)

print(f"7. Метод Ейткена:")
print(f"   Уточнене значення I_E = {I_E:.12f}")
print(f"   Оцінений порядок точності p = {p:.2f}")
print(f"   Похибка epsE = {eps_E:.2e}\n")


def adaptive_simpson(f, a, b, delta, evals=0):
    h = b - a
    y0, y1_4, y1_2, y3_4, y1 = f(a), f(a + h / 4), f(a + h / 2), f(a + 3 * h / 4), f(b)

    I1 = (h / 6) * (y0 + 4 * y1_2 + y1)
    I2 = (h / 12) * (y0 + 4 * y1_4 + y1_2) + (h / 12) * (y1_2 + 4 * y3_4 + y1)

    evals += 5

    if abs(I1 - I2) <= delta:
        return I2, evals
    else:
        left_val, left_calls = adaptive_simpson(f, a, a + h / 2, delta)
        right_val, right_calls = adaptive_simpson(f, a + h / 2, b, delta)
        return left_val + right_val, evals + left_calls + right_calls


deltas = [1e-2, 1e-4, 1e-6, 1e-8]
print("9. Адаптивний алгоритм:")
for d in deltas:
    I_adapt, calls = adaptive_simpson(f, a, b, d)
    eps_adapt = abs(I_adapt - I0)
    print(f"   delta = {d:.0e} | I = {I_adapt:.10f} | Похибка = {eps_adapt:.2e} | Викликів f(x): {calls}")

plt.figure(figsize=(10, 6))
plt.loglog(N_values, errors, label='Похибка формули Сімпсона', color='blue')
plt.axhline(target_accuracy, color='red', linestyle='--', label=r'Задана точність $\epsilon = 10^{-12}$')
plt.axvline(N_opt, color='green', linestyle=':', label=f'$N_{{opt}} = {N_opt}$')
plt.title('Залежність похибки чисельного інтегрування від числа розбиттів N')
plt.xlabel('Число розбиттів відрізку (N)')
plt.ylabel('Абсолютна похибка $\epsilon(N)$')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()