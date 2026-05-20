#include <stdio.h>
#include <math.h>

#define N 3
#define MAX_ITER 100
#define TOL 1e-6

int main() {

    /* Coefficient matrix A */
    double A[N][N] = {
        {10, -1,  2},
        {-1, 11, -1},
        { 2, -1, 10}
    };

    /* Right-hand side vector b */
    double b[N] = {6, 25, -11};

    /* Initial guess */
    double x[N] = {0.0, 0.0, 0.0};

    double x_old[N];
    double error;
    int iter, i, j;

    printf("Gauss-Seidel Method\n");
    printf("-------------------\n");

    for (iter = 1; iter <= MAX_ITER; iter++) {

        /* Save previous iteration values */
        for (i = 0; i < N; i++) {
            x_old[i] = x[i];
        }

        /* Gauss-Seidel iteration */
        for (i = 0; i < N; i++) {

            double sum = 0.0;

            for (j = 0; j < N; j++) {
                if (j != i) {
                    sum += A[i][j] * x[j];
                }
            }

            x[i] = (b[i] - sum) / A[i][i];
        }

        /* Compute maximum error */
        error = 0.0;

        for (i = 0; i < N; i++) {
            double current_error = fabs(x[i] - x_old[i]);

            if (current_error > error) {
                error = current_error;
            }
        }

        /* Print current iteration */
        printf("Iteration %2d: ", iter);

        for (i = 0; i < N; i++) {
            printf("x%d = %.6f  ", i + 1, x[i]);
        }

        printf("\n");

        /* Check convergence */
        if (error < TOL) {
            printf("\nConverged after %d iterations.\n", iter);
            break;
        }
    }

    /* Final solution */
    printf("\nApproximate Solution:\n");
    for (i = 0; i < N; i++) {
        printf("x%d = %.6f\n", i + 1, x[i]);
    }

    return 0;
}
