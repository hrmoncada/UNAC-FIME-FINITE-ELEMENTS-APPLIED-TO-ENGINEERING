#include <stdio.h>
#include <math.h>

#define N 3
#define MAX_ITER 100
#define TOL 1e-6

/*-------------------------------------------------
  Gauss-Seidel Method Subprogram
-------------------------------------------------*/
void gaussSeidel(double A[N][N], double b[N], double x[N])
{
    int i, j, k;
    double sum, error;

    /* Initial guess */
    for(i = 0; i < N; i++)
        x[i] = 0.0;

    printf("Iterations:\n");

    for(k = 0; k < MAX_ITER; k++)
    {
        error = 0.0;

        for(i = 0; i < N; i++)
        {
            double old = x[i];

            sum = 0.0;

            for(j = 0; j < N; j++)
            {
                if(j != i)
                    sum += A[i][j] * x[j];
            }

            x[i] = (b[i] - sum) / A[i][i];

            error += fabs(x[i] - old);
        }

        printf("Iter %2d: ", k + 1);

        for(i = 0; i < N; i++)
            printf("%10.6f ", x[i]);

        printf("\n");

        /* Convergence test */
        if(error < TOL)
        {
            printf("\nConverged after %d iterations.\n", k + 1);
            return;
        }
    }

    printf("\nMaximum iterations reached.\n");
}

/*-------------------------------------------------
  Main Program
-------------------------------------------------*/
int main()
{
    /* Coefficient matrix A */
    double A[N][N] = {
        {10, -1,  2},
        {-1, 11, -1},
        { 2, -1, 10}
    };

    /* Right-hand side vector b */
    double b[N] = {6, 25, -11};

    /* Solution vector */
    double x[N];

    /* Call Gauss-Seidel method */
    gaussSeidel(A, b, x);

    /* Final solution */
    printf("\nApproximate Solution:\n");

    for(int i = 0; i < N; i++)
    {
        printf("x[%d] = %.6f\n", i + 1, x[i]);
    }

    return 0;
}
