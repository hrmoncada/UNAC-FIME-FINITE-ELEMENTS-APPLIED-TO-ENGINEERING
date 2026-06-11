#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;

struct Node{
    double x,y;
};

void EdgeLoad(
double x1,double y1,
double x2,double y2,
double p1,double p2,
double t,
double T[4])
{
    double L=sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));

    double c=(y2-y1)/L;
    double s=(x1-x2)/L;

    double Tx1=-p1*c;
    double Ty1=-p1*s;

    double Tx2=-p2*c;
    double Ty2=-p2*s;

    double factor=t*L/6.0;

    T[0]=factor*(2*Tx1+Tx2);
    T[1]=factor*(2*Ty1+Ty2);

    T[2]=factor*(Tx1+2*Tx2);
    T[3]=factor*(Ty1+2*Ty2);
}

int main()
{
    cout<<fixed<<setprecision(3);

    double thickness=10.0;

    Node n7={100,20};
    Node n8={85,40};
    Node n9={70,60};

    double T1[4];
    double T2[4];

    EdgeLoad(
    n7.x,n7.y,
    n8.x,n8.y,
    1,2,
    thickness,
    T1);

    EdgeLoad(
    n8.x,n8.y,
    n9.x,n9.y,
    2,3,
    thickness,
    T2);

    double F[6]={0};

    F[0]+=T1[0];
    F[1]+=T1[1];

    F[2]+=T1[2];
    F[3]+=T1[3];

    F[2]+=T2[0];
    F[3]+=T2[1];

    F[4]+=T2[2];
    F[5]+=T2[3];

    cout<<"\nEquivalent nodal loads\n\n";

    for(int i=0;i<6;i++)
        cout<<"F"<<13+i<<" = "<<F[i]<<endl;

    ofstream out("forces.dat");

    out<<n7.x<<" "<<n7.y<<" "<<F[0]<<" "<<F[1]<<endl;
    out<<n8.x<<" "<<n8.y<<" "<<F[2]<<" "<<F[3]<<endl;
    out<<n9.x<<" "<<n9.y<<" "<<F[4]<<" "<<F[5]<<endl;

    out.close();

    ofstream geo("geometry.dat");

    geo<<n7.x<<" "<<n7.y<<endl;
    geo<<n8.x<<" "<<n8.y<<endl;
    geo<<n9.x<<" "<<n9.y<<endl;

    geo.close();

    return 0;
}
