import numpy as np
import matplotlib.pyplot as plt


# 1. Задаємо функцію та її точну похідну
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t) #Формула вологості


def exact_derivative(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t) #Аналітична (точна) похідна


# Функція для чисельного диференціювання (центральна різниця)
def central_difference(f, x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2 * h) #Реалізація формули центральної різниці


# Задаємо точку, в якій шукаємо похідну (як у прикладі з методички)
t0 = 1.0
exact_val = exact_derivative(t0)

print(f"1. Точне значення похідної в точці t0={t0}: {exact_val:.10f}\n")

# 2. Дослідження залежності похибки від кроку h (h = 10^-20 ... 10^3)
# Використовуємо логарифмічну шкалу для генерації масиву кроків
h_values = np.logspace(-20, 3, 200)
errors = []

best_h = None
min_error = float('inf')

for h in h_values:
    # Обходимо проблему ділення на нуль при екстремально малих h
    if h == 0:
        continue

    approx_val = central_difference(M, t0, h)
    error = abs(approx_val - exact_val)
    errors.append(error)

    # Шукаємо оптимальний крок
    if error < min_error:
        min_error = error
        best_h = h

print(f"2. Оптимальний крок (найменша похибка): h0 = {best_h:.2e}")
print(f"   Досягнута точність: R0 = {min_error:.10e}\n")

# 3-5. Обчислення з кроком h = 10^-3 та 2h
h_base = 1e-3
y_prime_h = central_difference(M, t0, h_base)
y_prime_2h = central_difference(M, t0, 2 * h_base)

R1 = abs(y_prime_h - exact_val)

print(f"3-5. Обчислення при h = {h_base}:")
print(f"     Похідна y'(h) = {y_prime_h:.10f}")
print(f"     Похідна y'(2h) = {y_prime_2h:.10f}")
print(f"     Похибка R1 = {R1:.10e}\n")

# 6. Метод Рунге-Ромберга
y_R = y_prime_h + (y_prime_h - y_prime_2h) / 3
R2 = abs(y_R - exact_val)

print(f"6. Метод Рунге-Ромберга:")
print(f"   Уточнене значення: y_R' = {y_R:.10f}")
print(f"   Похибка R2 = {R2:.10e}")
print(f"   Зменшення похибки у {R1 / R2:.2f} разів\n")

# 7. Метод Ейткена
y_prime_4h = central_difference(M, t0, 4 * h_base)

numerator_E = (y_prime_2h) ** 2 - y_prime_4h * y_prime_h #Чисельник
denominator_E = 2 * y_prime_2h - (y_prime_4h + y_prime_h) #Знаменник
y_E = numerator_E / denominator_E

# Оцінка порядку точності
val_p = abs((y_prime_4h - y_prime_2h) / (y_prime_2h - y_prime_h))
p = (1 / np.log(2)) * np.log(val_p)

R3 = abs(y_E - exact_val)

print(f"7. Метод Ейткена:")
print(f"   Уточнене значення: y_E' = {y_E:.10f}")
print(f"   Порядок точності p = {p:.4f}")
print(f"   Похибка R3 = {R3:.10e}")

# Будуємо графік залежності похибки від кроку h (лог-лог масштаб)
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors, label='Похибка чисельного диференціювання', color='blue')
plt.axvline(best_h, color='red', linestyle='--', label=rf'Оптимальне $h_0 \approx$ {best_h:.1e}')
plt.xlabel('Крок сітки h')
plt.ylabel('Абсолютна похибка R')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.show()