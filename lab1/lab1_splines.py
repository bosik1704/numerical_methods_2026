import requests
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# Завдання 1-2: Запит до Open-Elevation API та отримання даних
# ==============================================================================
def fetch_elevation_data():
    """Виконує запит до відкритого API висот."""
    # Точно за координатами з методички, з'єднані через '|' для API [cite: 93, 100-102]
    coords_str = "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={coords_str}"

    response = requests.get(url)  # Запасного плану немає, покладаємось на API
    data = response.json()
    return data["results"]  # Отримуємо значення широти, довготи, висоти [cite: 94]


# ==============================================================================
# Завдання 4: Кумулятивна відстань (Haversine)
# ==============================================================================
def haversine(lat1, lon1, lat2, lon2):
    """Обчислює відстань між двома точками на сфері [cite: 121-129]."""
    R = 6371000  # [cite: 121]
    phi1, phi2 = np.radians(lat1), np.radians(lat2)  # [cite: 122]
    dphi = np.radians(lat2 - lat1)  # [cite: 123, 124]
    dlambda = np.radians(lon2 - lon1)  # [cite: 125, 126]
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2  # [cite: 127, 128]
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))  # [cite: 129]


# ==============================================================================
# Завдання 7: Метод прогонки
# ==============================================================================
def solve_tridiagonal(alpha, beta, gamma, delta):
    """Розв'язок системи з трьохдіагональною матрицею методом прогонки[cite: 141]."""
    n = len(delta)
    A = np.zeros(n)
    B = np.zeros(n)
    x = np.zeros(n)

    # Пряма прогонка [cite: 53-64]
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]
    for i in range(1, n - 1):
        denominator = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denominator
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denominator

    # Зворотна прогонка [cite: 66-72]
    x[-1] = (delta[-1] - alpha[-1] * B[-2]) / (alpha[-1] * A[-2] + beta[-1])
    for i in range(n - 2, -1, -1):
        x[i] = A[i] * x[i + 1] + B[i]

    return x


# ==============================================================================
# Завдання 6, 8, 9: Кубічні сплайни та коефіцієнти
# ==============================================================================
def compute_splines(x_nodes, y_nodes, print_coeffs=False):
    """Знаходить коефіцієнти системи та коефіцієнти сплайнів [cite: 139-144]."""
    n = len(x_nodes) - 1
    h = np.diff(x_nodes)
    a = y_nodes[:-1]  # a_i = y_{i-1} [cite: 36]

    alpha = np.zeros(n + 1);
    beta = np.ones(n + 1)
    gamma = np.zeros(n + 1);
    delta = np.zeros(n + 1)

    # Задаємо нульову кривизну в крайніх вузлах (вільний сплайн) [cite: 29-31]
    beta[0] = 1.0;
    gamma[0] = 0.0;
    delta[0] = 0.0  # c_1 = 0 [cite: 45]

    for i in range(1, n):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])  # [cite: 43]
        gamma[i] = h[i]
        delta[i] = 3 * ((y_nodes[i + 1] - y_nodes[i]) / h[i] - (y_nodes[i] - y_nodes[i - 1]) / h[i - 1])  # [cite: 43]

    beta[n] = 1.0;
    alpha[n] = 0.0;
    delta[n] = 0.0

    if print_coeffs:
        # Завдання 6: Вивести коефіцієнти СЛАР
        print("\n--- Завдання 6: Коефіцієнти СЛАР ---")
        print(f"Alpha: {np.round(alpha, 3)}")
        print(f"Beta:  {np.round(beta, 3)}")
        print(f"Gamma: {np.round(gamma, 3)}")
        print(f"Delta: {np.round(delta, 3)}")

    # Завдання 7-8: Розв'язок і вивід C_i [cite: 141, 142]
    c = solve_tridiagonal(alpha, beta, gamma, delta)
    if print_coeffs:
        print("\n--- Завдання 7 та 8: Коефіцієнти C_i ---")
        print(f"C_i = {np.round(c, 4)}")

    # Завдання 9: Обчислення a_i, b_i, d_i [cite: 143]
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - (h[i] / 3) * (c[i + 1] + 2 * c[i])  # [cite: 38]
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])  # [cite: 37]

    if print_coeffs:
        print("\n--- Завдання 9: Коефіцієнти a_i, b_i, d_i ---")
        for i in range(n):
            print(f"Інтервал {i + 1}: a={a[i]:.2f}, b={b[i]:.4f}, d={d[i]:.6f}")

    return a, b, c[:-1], d


def evaluate_spline(x_val, x_nodes, a, b, c, d):
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_val <= x_nodes[i + 1]:
            dx = x_val - x_nodes[i]
            return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3  # [cite: 11]

    dx = x_val - x_nodes[-2]
    return a[-1] + b[-1] * dx + c[-1] * dx ** 2 + d[-1] * dx ** 3


# ==============================================================================
# Виконання головної програми
# ==============================================================================
if __name__ == "__main__":
    # 1-2. Отримуємо дані
    results = fetch_elevation_data()
    n_total = len(results)

    # 3. Запис табуляції у текстовий файл
    with open("tabulation_results.txt", "w", encoding="utf-8") as f:
        header = f"{'Index':<5} | {'Latitude':<10} | {'Longitude':<10} | {'Elevation (m)'}"
        print(header);
        f.write(header + "\n")

        coords = [];
        elevations = []
        for i, p in enumerate(results):
            lat, lon, elev = p["latitude"], p["longitude"], p["elevation"]
            coords.append((lat, lon))
            elevations.append(elev)
            row = f"{i:<5} | {lat:<10.6f} | {lon:<10.6f} | {elev:.2f}"
            print(row);
            f.write(row + "\n")

    # 4. Обчислення кумулятивної відстані [cite: 115-136]
    distances = [0.0]
    for i in range(1, n_total):
        d_val = haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        distances.append(distances[-1] + d_val)

    # 5. Дискретний набір точок [cite: 137, 138]
    x_arr = np.array(distances)
    y_arr = np.array(elevations)

    # Виконуємо завдання 6-9 для повного набору вузлів
    a_full, b_full, c_full, d_full = compute_splines(x_arr, y_arr, print_coeffs=True)

    # Додаткові завдання 1-3 [cite: 150-180]
    total_ascent = sum(max(y_arr[i] - y_arr[i - 1], 0) for i in range(1, n_total))  # [cite: 155]
    total_descent = sum(max(y_arr[i - 1] - y_arr[i], 0) for i in range(1, n_total))  # [cite: 157]
    print("\n--- Додаткові завдання ---")
    print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")  # [cite: 153]
    print(f"Сумарний набір висоти (м): {total_ascent:.2f}")  # [cite: 155]
    print(f"Сумарний спуск (м): {total_descent:.2f}")  # [cite: 157]

    # Градієнт та енергія [cite: 158-180]
    x_dense = np.linspace(x_arr[0], x_arr[-1], 500)
    y_full_spline = [evaluate_spline(xv, x_arr, a_full, b_full, c_full, d_full) for xv in x_dense]
    grad_full = np.gradient(y_full_spline, x_dense) * 100  # [cite: 164, 165]
    print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")  # [cite: 166, 167]
    print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")  # [cite: 168]
    print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")  # [cite: 169]

    mass = 80;
    g = 9.81  # [cite: 172, 173]
    energy = mass * g * total_ascent  # [cite: 174-177]
    print(f"Механічна робота (Дж): {energy:.2f}")  # [cite: 178]
    print(f"Механічна робота (кДж): {energy / 1000:.2f}")  # [cite: 178]
    print(f"Енергія (ккал): {energy / 4184:.2f}")  # [cite: 180]

    # ==============================================================================
    # Завдання 10: Побудова графіків (10, 15, 20 вузлів) [cite: 145]
    # Завдання 12: Графіки y=f(x), y_approx та похибки [cite: 147, 148]
    # ==============================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # --- Графік 1: Інтерполяція (Завдання 10 та 12 - наближені значення) ---
    ax1.plot(x_dense, y_full_spline, label='Сплайн y=f(x) (всі 21 вузлів)', color='black', linewidth=2)  # [cite: 147]

    node_counts = [10, 15]  # [cite: 145, 146]
    colors = ['r', 'g']

    # Для розрахунку похибки збережемо сплайн для 10 вузлів
    y_approx_10 = []

    for count, color in zip(node_counts, colors):
        idx = np.linspace(0, len(x_arr) - 1, count, dtype=int)
        x_sub = x_arr[idx];
        y_sub = y_arr[idx]

        a_sub, b_sub, c_sub, d_sub = compute_splines(x_sub, y_sub)
        y_spline = [evaluate_spline(xv, x_sub, a_sub, b_sub, c_sub, d_sub) for xv in x_dense]  #

        if count == 10:
            y_approx_10 = y_spline  # Зберігаємо для графіка похибки

        ax1.plot(x_dense, y_spline, label=f'Наближене значення ({count} вузлів)', color=color, alpha=0.8)  #
        ax1.scatter(x_sub, y_sub, color=color, s=20)

    ax1.set_title("Завдання 10 та 12: Графіки функції y=f(x) та її наближень [cite: 145, 147]")
    ax1.set_xlabel("Кумулятивна відстань (м)")
    ax1.set_ylabel("Висота (м)")
    ax1.legend()
    ax1.grid(True)

    # --- Графік 2: Похибка epsilon = |y - y_approx| (Завдання 12) ---
    # Похибка між еталонним сплайном (21 вузол) та наближеним (10 вузлів)
    epsilon = np.abs(np.array(y_full_spline) - np.array(y_approx_10))  #

    ax2.plot(x_dense, epsilon, color='purple', label='Похибка ε = |y - y_approx| (для 10 вузлів)')  #
    ax2.fill_between(x_dense, epsilon, color='purple', alpha=0.3)
    ax2.set_title("Завдання 12: Похибка інтерполяції на відрізку [x_0, x_n] ")
    ax2.set_xlabel("Кумулятивна відстань (м)")
    ax2.set_ylabel("Похибка ε (м)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
