#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;

void printMatrix(const vector<vector<double>>& A, const vector<double>& b, const string& title) {
    cout << "\n" << title << endl;
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[i].size(); j++) {
            cout << setw(8) << fixed << setprecision(3) << A[i][j] << " ";
        }
        cout << " | " << setw(8) << b[i] << endl;
    }
    cout << endl;
}

vector<double> gaussJordan(vector<vector<double>> A, vector<double> b) {
    int n = A.size();

    printMatrix(A, b, "Initial System:");

    for (int i = 0; i < n; i++) {
        cout << "--- Working on column " << i << " ---" << endl;

        // Pivoting
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(A[k][i]) > fabs(A[maxRow][i])) {
                maxRow = k;
            }
        }

        if (maxRow != i) {
            swap(A[i], A[maxRow]);
            swap(b[i], b[maxRow]);
            printMatrix(A, b, "Swap R" + to_string(i) + " <-> R" + to_string(maxRow));
        }

        // Normalize pivot row
        double pivot = A[i][i];
        for (int j = 0; j < n; j++) {
            A[i][j] /= pivot;
        }
        b[i] /= pivot;

        printMatrix(A, b, "Normalize R" + to_string(i));

        // Eliminate ALL other rows
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = A[k][i];
                for (int j = 0; j < n; j++) {
                    A[k][j] -= factor * A[i][j];
                }
                b[k] -= factor * b[i];

                printMatrix(A, b,
                    "R" + to_string(k) + " = R" + to_string(k) +
                    " - (" + to_string(factor) + ")R" + to_string(i));
            }
        }
    }

    cout << "--- Final Reduced System (RREF) ---" << endl;
    printMatrix(A, b, "Result:");

    return b; // solution vector
}

int main() {

// Test system (3 equations)
    vector<vector<double>> A = {
    {1, -2, 6},
    {2,  2, 3},
    {-1, 0, 3}
    };

    vector<double> b = {0, 3, 2};
/*
    // 🔹 Test system (4 equations)
    vector<vector<double>> A = {
        {2, -1, 1, 2},
        {1,  1, -1, 1},
        {3, -1, 2, 3},
        {1,  2, 3, -1}
    };

    vector<double> b = {8, 2, 13, 4};
*/
    vector<double> solution = gaussJordan(A, b);

    cout << "Final Solution:" << endl;
    for (int i = 0; i < solution.size(); i++) {
        cout << "x" << i << " = " << fixed << setprecision(4) << solution[i] << endl;
    }

    return 0;
}
