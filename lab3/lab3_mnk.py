import csv
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# 1. Підготовка вхідних даних
# ==============================================================================
def create_and_read_data():
    """
    Створює CSV файл з даними про середньомісячну температуру за 24 місяці
    [cite_start]та зчитує їх для подальшої роботи [cite: 580, 594, 601-626].
    """
    filename = "temperature_data.csv"

    # [cite_start]Вхідні дані згідно з методичкою: (Місяць x_i, Температура f_i) [cite: 601-626].
    data = [
        (1, -2), (2, 0), (3, 5), (4, 10), (5, 15), (6, 20), (7, 23), (8, 22),
        (9, 17), (10, 10), (11, 5), (12, 0), (13, -10), (14, 3), (15, 7), (16, 13),
        (17, 19), (18, 20), (19, 22), (20, 21), (21, 18), (22, 15), (23, 10), (24, 3)
    ]

    # Записуємо згенеровані дані у файл CSV
    with open(filename, 'w', newline='') as file_out:
        writer = csv.writer(file_out)
        writer.writerow(["Month", "Temp"])
        writer.writerows(data)

    # Зчитуємо дані з файлу у масиви NumPy.
    # x_arr - це наші вузли x_0, x_1, ... x_n
    # y_arr - це значення функції у вузлах f_0, f_1, ... f_n
    x_arr, y_arr = [], []
    with open(filename, 'r') as file_in:
        reader = csv.DictReader(file_in)
        for row in reader:
            x_arr.append(float(row['Month']))
            y_arr.append(float(row['Temp']))

    return np.array(x_arr), np.array(y_arr)


# ==============================================================================
# 2. Формування матриць для Методу найменших квадратів (МНК)
# ==============================================================================
def form_matrix(x_arr, m_deg):
    """
    Формує матрицю коефіцієнтів A (розміром m+1 на m+1) для системи
    нормальних рівнянь МНК.
    [cite_start]Згідно з формулою: b_{kl} = сума( p_i * x_i^(k+l) )[cite: 528].
    [cite_start]У нашому випадку вага p_i = 1 для всіх точок [cite: 514-515].
    """
    mat_a = np.zeros((m_deg + 1, m_deg + 1))

    for i in range(m_deg + 1):
        for j in range(m_deg + 1):
            # Підсумовуємо значення x_i у степені (i+j) по всіх вузлах
            mat_a[i, j] = np.sum(x_arr ** (i + j))

    return mat_a


def form_vector(x_arr, y_arr, m_deg):
    """
    Формує вектор вільних членів b (розміром m+1).
    [cite_start]Згідно з формулою: c_k = сума( p_i * f_i * x_i^k )[cite: 529].
    Вага p_i = 1.
    """
    vec_b = np.zeros(m_deg + 1)

    for i in range(m_deg + 1):
        # Підсумовуємо добутки f_i та x_i у степені i по всіх вузлах
        vec_b[i] = np.sum(y_arr * (x_arr ** i))

    return vec_b


# ==============================================================================
# [cite_start]3. Метод Гауса з вибором головного елемента [cite: 537-539]
# ==============================================================================
def gauss_solve(mat_a_in, vec_b_in):
    """
    Знаходить розв'язок системи лінійних алгебраїчних рівнянь A*x = b.
    Використовується метод Гауса з вибором найбільшого елемента по стовпцю
    [cite_start]для мінімізації похибки округлення[cite: 546].
    """
    mat_a = np.copy(mat_a_in)
    vec_b = np.copy(vec_b_in)
    size_n = len(vec_b)

    # [cite_start]--- Прямий хід методу Гауса (зведення матриці до трикутного вигляду) --- [cite: 571]
    for k in range(size_n - 1):
        # [cite_start]1. Вибір головного елемента [cite: 546-548]
        # Шукаємо індекс рядка з найбільшим за модулем елементом у поточному стовпці k
        max_row = k + np.argmax(np.abs(mat_a[k:size_n, k]))

        # [cite_start]2. Перестановка рядків [cite: 549-552]
        # Якщо найбільший елемент не лежить на головній діагоналі, міняємо рядки місцями
        if max_row != k:
            mat_a[[k, max_row]] = mat_a[[max_row, k]]
            vec_b[[k, max_row]] = vec_b[[max_row, k]]

        # [cite_start]3. Виключення невідомих [cite: 553-560]
        # Занулюємо елементи під головною діагоналлю у стовпці k
        for i in range(k + 1, size_n):
            # Обчислюємо множник (відношення поточного елемента до головного)
            factor = mat_a[i, k] / mat_a[k, k]
            # Віднімаємо від поточного рядка головний рядок, помножений на множник
            mat_a[i, k:] -= factor * mat_a[k, k:]
            vec_b[i] -= factor * vec_b[k]

    # [cite_start]--- Зворотний хід методу Гауса --- [cite: 571-572]
    # Знаходження самих коефіцієнтів a_i, рухаючись від останнього рівняння до першого
    x_sol = np.zeros(size_n)
    for i in range(size_n - 1, -1, -1):
        # [cite_start]Віднімаємо від вільного члена суму вже знайдених невідомих [cite: 572]
        sum_ax = np.sum(mat_a[i, i + 1:] * x_sol[i + 1:])
        x_sol[i] = (vec_b[i] - sum_ax) / mat_a[i, i]

    return x_sol


# ==============================================================================
# 4. Обчислення многочлена та дисперсії
# ==============================================================================
def evaluate_polynomial(x_val, coef):
    """
    Обчислює значення апроксимуючого многочлена в заданій точці x_val.
    [cite_start]Формула: phi(x) = a_0 + a_1*x + a_2*x^2 + ... + a_m*x^m[cite: 509].
    Тут coef - це масив знайдених коефіцієнтів [a_0, a_1, ..., a_m].
    """
    # Якщо передано одне число, перетворюємо його на масив NumPy для зручності
    if np.isscalar(x_val):
        x_val = np.array([x_val])

    y_poly = np.zeros_like(x_val, dtype=float)

    # Підсумовуємо всі члени многочлена
    for idx in range(len(coef)):
        y_poly += coef[idx] * (x_val ** idx)

    return y_poly


def calculate_variance(y_true, y_approx):
    """
    Обчислює дисперсію відхилень (міру того, наскільки добре многочлен описує дані).
    [cite_start]Формула: delta = sqrt( сума(phi(x_i) - f_i)^2 / (n+1) ) [cite: 531-532].
    """
    num_pts = len(y_true)
    # Знаходимо суму квадратів різниць між наближеними та фактичними значеннями
    sum_sq = np.sum((y_approx - y_true) ** 2)
    # Повертаємо корінь з середнього квадрату відхилення
    return np.sqrt(sum_sq / num_pts)


# ==============================================================================
# Головний блок програми
# ==============================================================================
if __name__ == "__main__":
    # [cite_start]1. Отримуємо дані [cite: 580-581]
    x_data, y_data = create_and_read_data()

    print("=== Лабораторна робота №3: Метод найменших квадратів ===")

    # [cite_start]Максимальний степінь многочлена m = 10, як вказано в завданні [cite: 584, 590]
    max_degree = 10
    variances = []  # Список для збереження значень дисперсії
    models = []  # Список для збереження коефіцієнтів многочленів

    print("\n--- Дисперсії для різних степенів m ---")

    # [cite_start]2. Знаходження апроксимуючого многочлена для m від 1 до 10 [cite: 584]
    for m in range(1, max_degree + 1):
        # Формуємо СЛАР
        mat_A = form_matrix(x_data, m)
        vec_B = form_vector(x_data, y_data, m)

        try:
            # Розв'язуємо СЛАР методом Гауса
            coefficients = gauss_solve(mat_A, vec_B)
            # Обчислюємо значення многочлена у вузлах
            y_approximation = evaluate_polynomial(x_data, coefficients)
            # [cite_start]Обчислюємо дисперсію [cite: 584]
            variance_val = calculate_variance(y_data, y_approximation)

            variances.append(variance_val)
            models.append(coefficients)
            print(f"Степінь m={m:<2}: дисперсія = {variance_val:.4f}")
        except Exception as e:
            # Обробка помилки (якщо матриця вироджена або числа занадто великі)
            print(f"Степінь m={m:<2}: Неможливо розв'язати ({e})")
            variances.append(float('inf'))
            models.append(None)

    # [cite_start]3. Вибір оптимального степеня (там, де дисперсія мінімальна) [cite: 585-587, 597]
    optimal_index = np.argmin(variances)
    optimal_m = optimal_index + 1
    optimal_coef = models[optimal_index]

    print(f"\n=> Оптимальний степінь многочлена (мінімум дисперсії): m = {optimal_m}")
    print("Коефіцієнти апроксимуючого полінома:")
    for idx, a_val in enumerate(optimal_coef):
        print(f"  a_{idx} = {a_val:.6f}")

    # [cite_start]4. Прогноз на наступні 3 місяці (Екстраполяція) [cite: 600, 681-684]
    x_future = np.array([25, 26, 27])
    y_future = evaluate_polynomial(x_future, optimal_coef)

    print("\n--- Прогноз температури на наступні 3 місяці ---")
    for mth, temp in zip(x_future, y_future):
        print(f"Місяць {mth}: {temp:.2f} °C")

    # ==========================================================================
    # [cite_start]Побудова графіків [cite: 584, 590, 598-599]
    # ==========================================================================
    # Створюємо вікно з 3 графіками, розташованими вертикально
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # [cite_start]--- Графік 1: Залежність дисперсії від степені [cite: 584] ---
    axs[0].plot(range(1, max_degree + 1), variances, marker='o', color='purple')
    axs[0].set_title("Залежність величини дисперсії від степені многочлена m")
    axs[0].set_xlabel("Степінь многочлена (m)")
    axs[0].set_ylabel("Дисперсія $\delta$")
    axs[0].axvline(optimal_m, color='r', linestyle='--', label=f'Оптимальне m={optimal_m}')
    axs[0].legend()
    axs[0].grid(True)

    # [cite_start]--- Графік 2: Апроксимація та прогноз [cite: 598] ---
    # Генеруємо густу сітку точок для плавного малювання кривої
    x_dense = np.linspace(min(x_data), max(x_future), 200)
    y_dense_approx = evaluate_polynomial(x_dense, optimal_coef)

    axs[1].scatter(x_data, y_data, color='blue', label='Фактичні дані (з CSV)', zorder=5)
    axs[1].plot(x_dense, y_dense_approx, color='red', label=f'Апроксимація (m={optimal_m})')
    axs[1].scatter(x_future, y_future, color='green', marker='*', s=150, zorder=10, label='Прогноз (екстраполяція)')
    axs[1].set_title("Апроксимація температурних даних МНК та Прогноз")
    axs[1].set_xlabel("Місяць")
    axs[1].set_ylabel("Температура (°C)")
    axs[1].legend()
    axs[1].grid(True)

    # [cite_start]--- Графік 3: Графік похибки [cite: 588-590, 599] ---
    # Знаходимо значення оптимального многочлена безпосередньо у вузлах сітки
    y_opt_approx = evaluate_polynomial(x_data, optimal_coef)
    # [cite_start]Обчислюємо абсолютну похибку e(x) = |f(x) - phi(x)| [cite: 583]
    error_vals = np.abs(y_data - y_opt_approx)

    axs[2].bar(x_data, error_vals, color='orange', alpha=0.7, label='Абсолютна похибка $\epsilon(x)$')
    axs[2].plot(x_data, error_vals, color='red', marker='o')
    axs[2].set_title(f"Похибка апроксимації для оптимального многочлена (m={optimal_m})")
    axs[2].set_xlabel("Місяць")
    axs[2].set_ylabel("Похибка $\epsilon$ (°C)")
    axs[2].legend()
    axs[2].grid(True)

    # Автоматичне вирівнювання відступів між графіками
    plt.tight_layout()

    # Відображення графіка у вікні (збереження у файл видалено за запитом)
    plt.show()