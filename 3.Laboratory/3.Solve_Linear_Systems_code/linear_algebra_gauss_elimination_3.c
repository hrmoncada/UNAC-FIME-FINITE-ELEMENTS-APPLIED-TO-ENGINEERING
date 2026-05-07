#include <stdio.h>
#include <math.h>

#define MAX 10

void printMatrix(int n, double A[MAX][MAX], double b[MAX], const char *title) {
    printf("\n%s\n", title);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.3f ", A[i][j]);
        }
        printf(" | %8.3f\n", b[i]);
    }
    printf("\n");
}

void gaussianElimination(int n, double A[MAX][MAX], double b[MAX], double x[MAX]) {
    printMatrix(n, A, b, "Initial System:");

    // Forward elimination
    for (int i = 0; i < n; i++) {
        printf("--- Eliminating column %d ---\n", i);

        // Pivoting
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(A[k][i]) > fabs(A[maxRow][i])) {
                maxRow = k;
            }
        }

        if (maxRow != i) {
            for (int j = 0; j < n; j++) {
                double temp = A[i][j];
                A[i][j] = A[maxRow][j];
                A[maxRow][j] = temp;
            }
            double tempb = b[i];
            b[i] = b[maxRow];
            b[maxRow] = tempb;

            printMatrix(n, A, b, "Row swap performed");
        }

        // Eliminate below pivot
        for (int k = i + 1; k < n; k++) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; j++) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];

            printMatrix(n, A, b, "Row elimination step");
        }
    }

    // Back substitution
    printf("--- Back Substitution ---\n");
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
        printf("x%d = %.4f\n", i, x[i]);
    }
}

int main() {
// Test system (3 equations)
    int n = 3;  // 🔹 Change this value for different system sizes

    double A[MAX][MAX] = {
    {1, -2, 6},
    {2,  2, 3},
    {-1, 0, 3}
    };

    double b[MAX] = {0, 3, 2};
/* // Test system (4 equations)
    int n = 4;  // 🔹 Change this value for different system sizes

    double A[MAX][MAX] = {
        {2, -1, 1, 2},
        {1,  1, -1, 1},
        {3, -1, 2, 3},
        {1,  2, 3, -1}
    };

    double b[MAX] = {8, 2, 13, 4};
*/
    double x[MAX];

    gaussianElimination(n, A, b, x);

    printf("\nFinal Solution:\n");
    for (int i = 0; i < n; i++) {
        printf("x%d = %.4f\n", i, x[i]);
    }

    return 0;
}
