import numpy as np

def generate_and_save_data(n=100, x_val=2.5, file_A="matrix_A.txt", file_B="vector_B.txt"):
    A = np.random.rand(n, n) * 10

    X_exact = np.full(n, x_val)

    B = np.dot(A, X_exact)

    np.savetxt(file_A, A)
    np.savetxt(file_B, B)

    return A, B, X_exact


def read_matrix(filename):
    return np.loadtxt(filename)


def read_vector(filename):
    return np.loadtxt(filename)


def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        U[i][i] = 1.0

    for k in range(n):
        for i in range(k, n):
            sum_L = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - sum_L

        for i in range(k + 1, n):
            sum_U = sum(L[k][j] * U[j][i] for j in range(k))
            U[k][i] = (A[k][i] - sum_U) / L[k][k]

    return L, U


def write_lu(L, U, filename="matrix_LU.txt"):
    with open(filename, 'w') as f:
        f.write("Matrix L:\n")
        np.savetxt(f, L)
        f.write("\nMatrix U:\n")
        np.savetxt(f, U)


def solve_lu(L, U, B):
    n = len(B)
    Z = np.zeros(n)
    X = np.zeros(n)

    for i in range(n):
        sum_Z = sum(L[i][j] * Z[j] for j in range(i))
        Z[i] = (B[i] - sum_Z) / L[i][i]


    for i in range(n - 1, -1, -1):
        sum_X = sum(U[i][j] * X[j] for j in range(i + 1, n))
        X[i] = Z[i] - sum_X
    return X


def matrix_vector_mult(A, X):
    return np.dot(A, X)


def vector_norm(V):
    return np.max(np.abs(V))


def main():
    n = 100


    generate_and_save_data(n)


    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")


    L, U = lu_decomposition(A)
    write_lu(L, U)


    X_calc = solve_lu(L, U, B)


    B_calc = matrix_vector_mult(A, X_calc)
    residual_vector = B_calc - B
    eps = vector_norm(residual_vector)
    print(f"Початкова точність (максимальна похибка розв'язку): {eps}")


    eps_0 = 1e-14
    X_iter = X_calc.copy()
    iterations = 0

    print(f"Починаємо ітераційне уточнення (цільова точність: {eps_0})...")

    while True:

        B_curr = matrix_vector_mult(A, X_iter)
        R = B - B_curr
        current_eps = vector_norm(R)


        if current_eps <= eps_0:
            break


        dX = solve_lu(L, U, R)


        X_iter = X_iter + dX
        iterations += 1


        if iterations >= 50:
            print("Увага: Досягнуто машинної межі точності для типу float64.")
            break

    print("-" * 40)
    print(f"Кількість ітерацій для уточнення: {iterations}")
    print(f"Кінцева точність (норма нев'язки): {current_eps}")
    print(f"Фрагмент уточненого розв'язку: {X_iter[:5]} ...")


    np.savetxt("vector_X.txt", X_iter)


if __name__ == '__main__':
    main()