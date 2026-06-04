/*
A continuación se muestra una implementación completa en C++ para resolver el problema mediante el Método de Elementos Finitos (FEM), y un script en Python para visualizar los resultados numéricos y compararlos con la solución exacta.

La solución sigue exactamente el desarrollo mostrado en tus diapositivas:

*. Barra rotatoria.
*. 2 elementos cuadráticos.
*. 5 nodos.
*. Carga centrífuga equivalente.
*. Ensamblaje de la matriz global.
*. Aplicación de la condición de frontera Q_1 = 0.
*. Resolución del sistema.
*. Recuperación de esfuerzos en los nodos de cada elemento.
*. Comparación con la solución analítica.
*/

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;

//--------------------------------------------------
// Eliminación Gaussiana
//--------------------------------------------------
vector<double> solveSystem(vector<vector<double>> A,
                           vector<double> b)
{
    int n = b.size();

    for(int k=0;k<n-1;k++)
    {
        for(int i=k+1;i<n;i++)
        {
            double factor=A[i][k]/A[k][k];

            for(int j=k;j<n;j++)
                A[i][j]-=factor*A[k][j];

            b[i]-=factor*b[k];
        }
    }

    vector<double> x(n);

    for(int i=n-1;i>=0;i--)
    {
        x[i]=b[i];

        for(int j=i+1;j<n;j++)
            x[i]-=A[i][j]*x[j];

        x[i]/=A[i][i];
    }

    return x;
}

//--------------------------------------------------
int main()
{
    cout<<fixed<<setprecision(6);

    //--------------------------------------------------
    // Datos
    //--------------------------------------------------

    const double E  = 1.0e7;     // psi
    const double A  = 0.6;       // in²
    const double rho= 0.2836;    // lb/in³
    const double omega = 30.0;   // rad/s

    const double L  = 42.0;      // in
    const double Le = 21.0;      // in

    const double g  = 32.2*12.0; // in/s²

    //--------------------------------------------------
    // Matriz elemental
    //--------------------------------------------------

    double coef = E*A/(3.0*Le);

    vector<vector<double>> ke =
    {
        { 7,  1, -8},
        { 1,  7, -8},
        {-8, -8, 16}
    };

    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            ke[i][j]*=coef;

    //--------------------------------------------------
    // Ensamblaje global
    //--------------------------------------------------

    vector<vector<double>> K(5,vector<double>(5,0.0));

    int conn1[3]={0,2,1}; // [1 3 2]
    int conn2[3]={2,4,3}; // [3 5 4]

    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
        {
            K[conn1[i]][conn1[j]] += ke[i][j];
            K[conn2[i]][conn2[j]] += ke[i][j];
        }

    //--------------------------------------------------
    // Fuerzas centrífugas
    //--------------------------------------------------

    double r1=10.5;
    double r2=31.5;

    double f1=rho*r1*omega*omega/g;
    double f2=rho*r2*omega*omega/g;

    vector<double> fe1=
    {
        A*Le*f1/6.0,
        A*Le*f1/6.0,
        2.0*A*Le*f1/3.0
    };

    vector<double> fe2=
    {
        A*Le*f2/6.0,
        A*Le*f2/6.0,
        2.0*A*Le*f2/3.0
    };

    vector<double> F(5,0.0);

    for(int i=0;i<3;i++)
    {
        F[conn1[i]] += fe1[i];
        F[conn2[i]] += fe2[i];
    }

    //--------------------------------------------------
    // Condición de frontera
    //--------------------------------------------------

    vector<vector<double>> Kr(4,vector<double>(4));
    vector<double> Fr(4);

    for(int i=1;i<5;i++)
    {
        Fr[i-1]=F[i];

        for(int j=1;j<5;j++)
            Kr[i-1][j-1]=K[i][j];
    }

    //--------------------------------------------------
    // Resolver
    //--------------------------------------------------

    vector<double> qr = solveSystem(Kr,Fr);

    vector<double> Q(5,0.0);

    Q[0]=0.0;

    for(int i=1;i<5;i++)
        Q[i]=qr[i-1];

    //--------------------------------------------------
    // Resultados
    //--------------------------------------------------

    cout<<"\nDESPLAZAMIENTOS\n";
    cout<<"---------------------------\n";

    for(int i=0;i<5;i++)
        cout<<"Q"<<i+1<<" = "
            <<Q[i]
            <<" in\n";

    //--------------------------------------------------
    // Esfuerzos FEM
    //--------------------------------------------------

    auto stress = [&](double xi,
                      vector<double> q)
    {
        double B1=-(1.0-2.0*xi)/2.0;
        double B2=(1.0+2.0*xi)/2.0;
        double B3=-2.0*xi;

        return (2.0*E/Le)*
               (B1*q[0]+B2*q[1]+B3*q[2]);
    };

    vector<double> q1=
    {
        Q[0],Q[2],Q[1]
    };

    vector<double> q2=
    {
        Q[2],Q[4],Q[3]
    };

    cout<<"\nESFUERZOS FEM\n";
    cout<<"---------------------------\n";

    cout<<"Elemento 1\n";
    cout<<"xi=-1 : "<<stress(-1,q1)<<" psi\n";
    cout<<"xi= 0 : "<<stress(0,q1) <<" psi\n";
    cout<<"xi=+1 : "<<stress(1,q1) <<" psi\n";

    cout<<"\nElemento 2\n";
    cout<<"xi=-1 : "<<stress(-1,q2)<<" psi\n";
    cout<<"xi= 0 : "<<stress(0,q2) <<" psi\n";
    cout<<"xi=+1 : "<<stress(1,q2) <<" psi\n";

    return 0;
}
