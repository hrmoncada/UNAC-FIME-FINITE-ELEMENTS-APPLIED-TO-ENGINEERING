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

void gaussJordan(int n, double A[MAX][MAX], double b[MAX]) {
    printMatrix(n, A, b, "Initial System:");

    for (int i = 0; i < n; i++) {
        printf("--- Working on column %d ---\n", i);

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

        // Normalize pivot row
        double pivot = A[i][i];
        for (int j = 0; j < n; j++) {
            A[i][j] /= pivot;
        }
        b[i] /= pivot;

        printMatrix(n, A, b, "Normalize pivot row");

        // Eliminate all other rows
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = A[k][i];
                for (int j = 0; j < n; j++) {
                    A[k][j] -= factor * A[i][j];
                }
                b[k] -= factor * b[i];

                printMatrix(n, A, b, "Row elimination step");
            }
        }
    }

    printf("--- Final RREF ---\n");
    printMatrix(n, A, b, "");

    printf("Solution:\n");
    for (int i = 0; i < n; i++) {
        printf("x%d = %.4f\n", i, b[i]);
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
    gaussJordan(n, A, b);

    return 0;
}
