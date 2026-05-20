#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

const int N = 3;

/*-------------------------------------------------
  Gauss-Seidel Method Subprogram
--------------------------------------------------*/
void gaussSeidel(double A[N][N], double b[N],
                 double x[N], int maxIter, double tol)
{
    double x_old[N];

    for (int k = 0; k < maxIter; k++)
    {
        /* Store previous iteration */
        for (int i = 0; i < N; i++)
        {
            x_old[i] = x[i];
        }

        /* Gauss-Seidel iteration */
        for (int i = 0; i < N; i++)
        {
            double sum = 0.0;

            for (int j = 0; j < N; j++)
            {
                if (j != i)
                {
                    sum += A[i][j] * x[j];
                }
            }

            x[i] = (b[i] - sum) / A[i][i];
        }

        /* Compute error */
        double error = 0.0;

        for (int i = 0; i < N; i++)
        {
            error += pow(x[i] - x_old[i], 2);
        }

        error = sqrt(error);

        /* Print iteration results */
        cout << "Iteration " << k + 1 << ": ";

        for (int i = 0; i < N; i++)
        {
            cout << setw(10) << x[i] << " ";
        }

        cout << " Error = " << error << endl;

        /* Check convergence */
        if (error < tol)
        {
            cout << "\nConvergence reached.\n";
            return;
        }
    }

    cout << "\nMaximum iterations reached.\n";
}

/*-------------------------------------------------
  Main Program
--------------------------------------------------*/
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

    /* Initial guess */
    double x[N] = {0.0, 0.0, 0.0};

    /* Parameters */
    int maxIter = 100;
    double tol = 1e-6;

    cout << fixed << setprecision(6);

    /* Call Gauss-Seidel subprogram */
    gaussSeidel(A, b, x, maxIter, tol);

    /* Final solution */
    cout << "\nApproximate Solution:\n";

    for (int i = 0; i < N; i++)
    {
        cout << "x[" << i + 1 << "] = " << x[i] << endl;
    }

    return 0;
}
