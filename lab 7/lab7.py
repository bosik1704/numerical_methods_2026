import random

def generate_and_save_data(n=100, exact_x=2.5):
    A = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        A[i][i] = row_sum + random.uniform(1, 5)

    b = [0.0] * n
    for i in range(n):
        b[i] = sum(A[i][j] * exact_x for j in range(n))

    with open("matrix_A.txt", "w") as f_A:
        for row in A:
            f_A.write(" ".join(map(str, row)) + "\n")

    with open("vector_B.txt", "w") as f_B:
        for val in b:
            f_B.write(str(val) + "\n")
    print(f"Дані згенеровано та збережено (n={n}).")

def read_matrix(filename):
    with open(filename, "r") as f:
        return [[float(x) for x in line.split()] for line in f]


def read_vector(filename):
    with open(filename, "r") as f:
        return [float(line.strip()) for line in f]


def save_result_vector(filename, vector):
    with open(filename, "w") as f:
        for val in vector:
            f.write(str(val) + "\n")


def mat_vec_mult(A, x):
    n = len(A)
    res = [0.0] * n
    for i in range(n):
        res[i] = sum(A[i][j] * x[j] for j in range(n))
    return res


def vector_norm(v):
    return max(abs(x) for x in v)


def matrix_norm(A):
    return max(sum(abs(x) for x in row) for row in A)


def simple_iteration(A, b, eps=1e-14):
    n = len(A)
    x = [1.0] * n
    tau = 1.0 / matrix_norm(A)
    iterations = 0

    while True:
        Ax = mat_vec_mult(A, x)
        x_new = [0.0] * n
        for i in range(n):
            x_new[i] = x[i] - tau * (Ax[i] - b[i])

        diff = [x_new[i] - x[i] for i in range(n)]
        if vector_norm(diff) < eps:
            break
        x = x_new
        iterations += 1
    return x_new, iterations


def jacobi(A, b, eps=1e-14):
    n = len(A)
    x = [1.0] * n
    iterations = 0

    while True:
        x_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        diff = [x_new[i] - x[i] for i in range(n)]
        if vector_norm(diff) < eps:
            break
        x = x_new
        iterations += 1
    return x_new, iterations


def seidel(A, b, eps=1e-14):
    n = len(A)
    x = [1.0] * n
    iterations = 0

    while True:
        x_new = list(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        diff = [x_new[i] - x[i] for i in range(n)]
        if vector_norm(diff) < eps:
            break
        x = x_new
        iterations += 1
    return x_new, iterations


if __name__ == "__main__":
    generate_and_save_data(n=100)

    A = read_matrix("matrix_A.txt")
    b = read_vector("vector_B.txt")
    eps = 1e-14

    print("\nПочинаємо обчислення...")

    x_simple, iters_simple = simple_iteration(A, b, eps)
    print(f"Метод простої ітерації: {iters_simple} ітерацій")

    x_jacobi, iters_jacobi = jacobi(A, b, eps)
    print(f"Метод Якобі:           {iters_jacobi} ітерацій")

    x_seidel, iters_seidel = seidel(A, b, eps)
    print(f"Метод Гауса-Зейделя:   {iters_seidel} ітерацій")

    save_result_vector("result_vector_X.txt", x_seidel)
    print("\nУспіх! Результуючий вектор X збережено у файл 'result_vector_X.txt'.")