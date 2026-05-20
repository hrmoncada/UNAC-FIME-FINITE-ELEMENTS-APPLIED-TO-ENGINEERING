#include <stdio.h>
#include <math.h>

#define N 3
#define MAX_ITER 100
#define TOL 1e-8

void matVec(double A[N][N], double x[N], double r[N]) {
    for(int i=0;i<N;i++){
        r[i]=0;
        for(int j=0;j<N;j++)
            r[i]+=A[i][j]*x[j];
    }
}

double dot(double a[N], double b[N]){
    double s=0;
    for(int i=0;i<N;i++) s+=a[i]*b[i];
    return s;
}

void transpose(double A[N][N], double AT[N][N]){
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            AT[j][i]=A[i][j];
}

void matMul(double A[N][N], double B[N][N], double C[N][N]){
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            C[i][j]=0;
            for(int k=0;k<N;k++)
                C[i][j]+=A[i][k]*B[k][j];
        }
}

void matVecMul(double A[N][N], double b[N], double r[N]){
    for(int i=0;i<N;i++){
        r[i]=0;
        for(int j=0;j<N;j++)
            r[i]+=A[i][j]*b[j];
    }
}

int main(){

    double A[N][N]={
        {1,-2,6},
        {2,2,3},
        {-1,0,3}
    };

    double b[N]={0,3,2};

    // Save original matrix
    FILE *fm=fopen("matrix.txt","w");
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++)
            fprintf(fm,"%lf ",A[i][j]);
        fprintf(fm,"\n");
    }
    fclose(fm);

    // Build SPD system
    double AT[N][N], A_spd[N][N], b_spd[N];
    transpose(A,AT);
    matMul(AT,A,A_spd);
    matVecMul(AT,b,b_spd);

    // CG variables
    double x[N]={0}, r[N], p[N], Ap[N];

    for(int i=0;i<N;i++){
        r[i]=b_spd[i];
        p[i]=r[i];
    }

    double rs_old=dot(r,r);

    FILE *fe=fopen("cg_error.txt","w");
    fprintf(fe,"iter error\n");

    for(int k=0;k<MAX_ITER;k++){

        matVec(A_spd,p,Ap);

        double alpha=rs_old/dot(p,Ap);

        for(int i=0;i<N;i++)
            x[i]+=alpha*p[i];

        for(int i=0;i<N;i++)
            r[i]-=alpha*Ap[i];

        double rs_new=dot(r,r);
        double error=sqrt(rs_new);

        printf("Iter %d error=%.6e\n",k+1,error);
        fprintf(fe,"%d %e\n",k+1,error);

        if(error<TOL) break;

        double beta=rs_new/rs_old;

        for(int i=0;i<N;i++)
            p[i]=r[i]+beta*p[i];

        rs_old=rs_new;
    }

    fclose(fe);

    printf("\nSolution:\n");
    for(int i=0;i<N;i++)
        printf("x%d = %lf\n",i,x[i]);

    return 0;
}
