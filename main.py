import numpy as np

A = np.array([[1, 3], [5, 6]])
B = np.array([[1, 6], [5, 7]])

d = np.subtract(B, A)


def sherman_morrison_test(matrix):
    return np.linalg.matrix_rank(matrix) == 1


def define_uv(matrix, test):
    if test:
        n = matrix.shape[0]
        U = V = np.zeros(n, dtype=np.double)
        k = {
            "column": {"index": 0, "column_number": 0},
            "row": {"index": 0, "row_number": 0}
        }
        for i in range(n):
            if (matrix[:, i] != 0).all():
                k["column"]["column_number"] += 1
                k["column"]["index"] = i
            if (matrix[i, :] != 0).all():
                k["row"]["row_number"] += 1
                k["row"]["index"] = i
        if k["column"]["column_number"] != k["row"]["row_number"]:
            if k["column"]["column_number"] != 0:
                U = np.array(matrix[:, k["column"]["index"]])
                V[k["column"]["index"]] = 1
            if k["row"]["row_number"] != 0:
                U[k["row"]["index"]] = 1
                V = np.array(matrix[k["row"]["index"], :])
        else:
            for i in range(n):
                m = matrix[i, i] / matrix[0, i]
                U[i] = m
            V = np.array(matrix[0, :])
        return U, V
    else:
        print("conditions don't meet to be able to define u and v from A and B")
        return 0


print(A)
print(B)
u, v = define_uv(d, sherman_morrison_test(d))


def sherman_morrison(matrix_a, vector_u, vector_v):
    matrix_A_inv = np.linalg.inv(matrix_a)
    return np.subtract(matrix_A_inv, np.dot(np.outer(np.dot(matrix_A_inv, vector_u), vector_v), matrix_A_inv) / (1 + np.dot(np.dot(vector_v, matrix_A_inv), vector_u)))


B_inv = sherman_morrison(A, u, v)
print("sherman morrison method\n", B_inv)
print("in built function\n", np.linalg.inv(B))
