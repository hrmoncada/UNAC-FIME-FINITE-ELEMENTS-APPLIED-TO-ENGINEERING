#include <iostream>
#include <fstream>
#include <cmath>

#define N 3
#define MAX_ITER 100
#define TOL 1e-8

using namespace std;

void matVec(double A[N][N], double x[N], double result[N]) {
    for(int i = 0; i < N; i++) {
        result[i] = 0;
        for(int j = 0; j < N; j++)
            result[i] += A[i][j] * x[j];
    }
}

double dot(double a[N], double b[N]) {
    double sum = 0;
    for(int i = 0; i < N; i++)
        sum += a[i] * b[i];
    return sum;
}

void transpose(double A[N][N], double AT[N][N]) {
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            AT[j][i] = A[i][j];
}

void matMul(double A[N][N], double B[N][N], double C[N][N]) {
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++) {
            C[i][j] = 0;
            for(int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

void matVecMul(double A[N][N], double b[N], double result[N]) {
    for(int i = 0; i < N; i++) {
        result[i] = 0;
        for(int j = 0; j < N; j++)
            result[i] += A[i][j] * b[j];
    }
}

void saveMatrix(const string& filename, double M[N][N]) {
    ofstream f(filename);
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++)
            f << M[i][j] << " ";
        f << "\n";
    }
    f.close();
}

void conjugateGradient(double A[N][N], double b[N], double x[N]) {
    double r[N], p[N], Ap[N];

    ofstream file("cg_output.txt");
    file << "iter error x0 x1 x2\n";

    for(int i = 0; i < N; i++) {
        x[i] = 0;
        r[i] = b[i];
        p[i] = r[i];
    }

    double rs_old = dot(r, r);

    for(int k = 0; k < MAX_ITER; k++) {
        matVec(A, p, Ap);

        double alpha = rs_old / dot(p, Ap);

        for(int i = 0; i < N; i++)
            x[i] += alpha * p[i];

        for(int i = 0; i < N; i++)
            r[i] -= alpha * Ap[i];

        double rs_new = dot(r, r);
        double error = sqrt(rs_new);

        cout << "Iter " << k+1 << " error = " << error << endl;

        file << k+1 << " " << error << " "
             << x[0] << " " << x[1] << " " << x[2] << "\n";

        if(error < TOL) break;

        double beta = rs_new / rs_old;

        for(int i = 0; i < N; i++)
            p[i] = r[i] + beta * p[i];

        rs_old = rs_new;
    }

    file.close();
}

int main() {

    double A[N][N] = {
        {1, -2, 6},
        {2,  2, 3},
        {-1, 0, 3}
    };

    double b[N] = {0, 3, 2};

    // Save original matrix
    saveMatrix("matrix.txt", A);

    // Build SPD system
    double AT[N][N], A_spd[N][N], b_spd[N];

    transpose(A, AT);
    matMul(AT, A, A_spd);
    matVecMul(AT, b, b_spd);

    // Save SPD matrix
    saveMatrix("matrix_spd.txt", A_spd);

    double x[N];
    conjugateGradient(A_spd, b_spd, x);

    cout << "\nSolution:\n";
    for(int i = 0; i < N; i++)
        cout << "x" << i << " = " << x[i] << endl;

    return 0;
}
