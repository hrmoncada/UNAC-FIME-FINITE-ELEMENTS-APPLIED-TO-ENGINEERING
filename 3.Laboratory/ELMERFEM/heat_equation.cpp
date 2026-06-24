/*g++ heat_equation.cpp -o heat_equation
./heat_equation
*/
#include <iostream>
#include <fstream>
#include <vector>

const int nx = 50;  // Number of grid points in x-direction
const int ny = 50;  // Number of grid points in y-direction
const double Lx = 1.0;  // Length of domain in x-direction
const double Ly = 1.0;  // Length of domain in y-direction
const double dx = Lx / (nx - 1);  // Grid spacing in x-direction
const double dy = Ly / (ny - 1);  // Grid spacing in y-direction
const double dt = 0.001;  // Time step
const double alpha = 0.01;  // Thermal diffusivity
const double T_initial = 0.0;  // Initial temperature
const double T_left_boundary = 1.0;  // Temperature at left boundary
const double T_right_boundary = 0.0;  // Temperature at right boundary
const double T_top_boundary = 0.0;  // Temperature at top boundary
const double T_bottom_boundary = 0.0;  // Temperature at bottom boundary
const int num_steps = 10000;  // Number of time steps

void initialize(std::vector<std::vector<double>>& T) {
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            T[i][j] = T_initial;
        }
    }
    // Apply Dirichlet boundary conditions
    for (int i = 0; i < nx; ++i) {
        T[i][0] = T_bottom_boundary;
        T[i][ny - 1] = T_top_boundary;
    }
    for (int j = 0; j < ny; ++j) {
        T[0][j] = T_left_boundary;
        T[nx - 1][j] = T_right_boundary;
    }
}

void solve(std::vector<std::vector<double>>& T) {
    std::vector<std::vector<double>> T_new(nx, std::vector<double>(ny, 0.0));

    for (int step = 0; step < num_steps; ++step) {
        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                T_new[i][j] = T[i][j] + alpha * dt * (
                    (T[i + 1][j] - 2 * T[i][j] + T[i - 1][j]) / (dx * dx) +
                    (T[i][j + 1] - 2 * T[i][j] + T[i][j - 1]) / (dy * dy)
                );
            }
        }

        // Update T for the next time step
        T = T_new;
    }
}

int main() {
    // Create grid and initialize temperatures
    std::vector<std::vector<double>> T(nx, std::vector<double>(ny, 0.0));
    initialize(T);

    // Solve the heat equation
    solve(T);

    // Save results to a file
    std::ofstream output("temperature.csv");
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            output << i * dx << "," << j * dy << "," << T[i][j] << "\n";
        }
    }
    output.close();

    std::cout << "Simulation completed. Results saved to temperature.csv\n";

    return 0;
}

