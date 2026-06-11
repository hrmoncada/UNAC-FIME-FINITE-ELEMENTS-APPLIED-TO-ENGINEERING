#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

using namespace std;

struct Node
{
    double x,y;
};

void ShapeFunctions()
{
    Node n[3];

    n[0]={1.5,2.0};
    n[1]={7.0,3.5};
    n[2]={4.0,7.0};

    double xp=3.85;
    double yp=4.80;

    double A[3][4]={
        {n[0].x,n[1].x,n[2].x,xp},
        {n[0].y,n[1].y,n[2].y,yp},
        {1,1,1,1}
    };

    // Gaussian elimination

    for(int i=0;i<3;i++)
    {
        double pivot=A[i][i];

        for(int j=i;j<4;j++)
            A[i][j]/=pivot;

        for(int k=0;k<3;k++)
        {
            if(k==i) continue;

            double f=A[k][i];

            for(int j=i;j<4;j++)
                A[k][j]-=f*A[i][j];
        }
    }

    cout<<"\nShape functions\n";

    cout<<"N1 = "<<A[0][3]<<endl;
    cout<<"N2 = "<<A[1][3]<<endl;
    cout<<"N3 = "<<A[2][3]<<endl;

    ofstream out("triangle.dat");

    for(int i=0;i<3;i++)
        out<<n[i].x<<" "<<n[i].y<<endl;

    out<<n[0].x<<" "<<n[0].y<<endl;

    out.close();
}

void Jacobian()
{
    Node n[3];

    n[0]={1.5,2.0};
    n[1]={7.0,3.5};
    n[2]={4.0,7.0};

    double J[2][2];

    J[0][0]=n[0].x-n[2].x;
    J[0][1]=n[0].y-n[2].y;

    J[1][0]=n[1].x-n[2].x;
    J[1][1]=n[1].y-n[2].y;

    double det=
    J[0][0]*J[1][1]-
    J[0][1]*J[1][0];

    cout<<"\nJacobian\n";

    cout<<J[0][0]<<" "<<J[0][1]<<endl;
    cout<<J[1][0]<<" "<<J[1][1]<<endl;

    cout<<"\ndet(J)="<<det<<endl;
    cout<<"Area="<<0.5*fabs(det)<<endl;
}

void Mesh()
{
    ofstream out("mesh.dat");

    out<<0<<" "<<0<<endl;
    out<<3<<" "<<0<<endl;
    out<<3<<" "<<2<<endl;
    out<<0<<" "<<2<<endl;
    out<<0<<" "<<0<<endl;

    out.close();
}

int main()
{
    cout<<fixed<<setprecision(4);

    ShapeFunctions();

    Jacobian();

    Mesh();

    return 0;
}
