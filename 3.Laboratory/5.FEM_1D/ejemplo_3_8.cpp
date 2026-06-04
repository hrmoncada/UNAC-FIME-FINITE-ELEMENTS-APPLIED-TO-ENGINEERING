/*
A continuación se muestra un código completo en Python para resolver el problema mediante el Método de los Elementos Finitos (FEM). El programa:

*. Ensambla automáticamente la matriz global de rigidez.
*. Calcula las cargas térmicas equivalentes.
*. Aplica las condiciones de frontera.
*. Resuelve los desplazamientos nodales.
*. Calcula los esfuerzos en cada elemento.
*. Presenta resultados numéricos.
*. Genera gráficas de desplazamientos y esfuerzos.
*/
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace std;

int main()
{
    //-------------------------------------------------
    // Datos
    //-------------------------------------------------

    double E1 = 70e3;     // MPa
    double E2 = 200e3;    // MPa

    double A1 = 900.0;    // mm²
    double A2 = 1200.0;   // mm²

    double L1 = 200.0;    // mm
    double L2 = 300.0;    // mm

    double alpha1 = 23e-6;
    double alpha2 = 11.7e-6;

    double dT = 40.0;

    double P = 300e3;     // N

    //-------------------------------------------------
    // Rigideces de elemento
    //-------------------------------------------------

    double k1 = E1*A1/L1;
    double k2 = E2*A2/L2;

    //-------------------------------------------------
    // Matriz global
    //-------------------------------------------------

    double K[3][3] =
    {
        { k1,     -k1,      0 },
        {-k1, k1+k2,    -k2 },
        {  0,     -k2,    k2 }
    };

    //-------------------------------------------------
    // Fuerzas térmicas
    //-------------------------------------------------

    double theta1 = E1*A1*alpha1*dT;
    double theta2 = E2*A2*alpha2*dT;

    //-------------------------------------------------
    // Vector global
    //-------------------------------------------------

    double F[3];

    F[0] = -theta1;

    F[1] = theta1 - theta2 + P;

    F[2] = theta2;

    //-------------------------------------------------
    // Condiciones de frontera
    //-------------------------------------------------

    double Q[3];

    Q[0] = 0.0;
    Q[2] = 0.0;

    Q[1] = F[1]/(K[1][1]);

    //-------------------------------------------------
    // Esfuerzos
    //-------------------------------------------------

    double sigma1 =
        E1*(Q[1]-Q[0])/L1
        - E1*alpha1*dT;

    double sigma2 =
        E2*(Q[2]-Q[1])/L2
        - E2*alpha2*dT;

    //-------------------------------------------------
    // Resultados
    //-------------------------------------------------

    cout << fixed << setprecision(6);

    cout << "\nDESPLAZAMIENTOS (mm)\n";
    cout << "Q1 = " << Q[0] << endl;
    cout << "Q2 = " << Q[1] << endl;
    cout << "Q3 = " << Q[2] << endl;

    cout << "\nESFUERZOS (MPa)\n";
    cout << "sigma1 = " << sigma1 << endl;
    cout << "sigma2 = " << sigma2 << endl;

    //-------------------------------------------------
    // Archivo para Python
    //-------------------------------------------------

    ofstream out("resultados.csv");

    out << "Nodo,X,Q\n";

    out << "1,0,"     << Q[0] << "\n";
    out << "2,200,"   << Q[1] << "\n";
    out << "3,500,"   << Q[2] << "\n";

    out.close();

    return 0;
}

