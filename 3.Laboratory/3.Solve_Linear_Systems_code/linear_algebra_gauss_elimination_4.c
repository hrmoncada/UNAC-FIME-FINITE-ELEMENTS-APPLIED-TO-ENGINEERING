//https://juncotic.com/eliminacion-gaussiana-algoritmos-antiguos/
#include<stdio.h>
 
#define ORDEN 3
#define COLS 4
 
void mostrarMatriz(float matriz[ORDEN][COLS]);
 
int main(int argc, char** argv){
    //Función para mostrar la matriz
 
    float matriz[ORDEN][4]={
    {3,2,3,3},
    {1,3,1,-6},
    {5,1,3,12}
    };
 
    float x[ORDEN],suma;
 
    int j,i,k;
 
    //Orden de la matriz
    float division;
     
    mostrarMatriz(matriz);
     
    //Recorrer la matriz 
    for(j=0;j<=ORDEN;j++){
        for(i=0; i<=(ORDEN-1); i++){
            if(i>j){
                //Divir los elementos de la matriz
                division=matriz[i][j]/matriz[j][j];
                for(k=0;k<=ORDEN;k++){
                    //Obterner el nuevo valor para los elementos en la diagonal inferior
                    matriz[i][k]=matriz[i][k]-division*matriz[j][k];
                }
            }
        }
    }
   
    //Recorrer la matriz
    for(i=ORDEN-1;i>=0;i--){
        suma=0;
        for(j=i+1;j<=ORDEN-1;j++){
            suma=suma+matriz[i][j]*x[j];
        }
        //Obtener los valores de las variables
        x[i]=(matriz[i][ORDEN]-suma)/matriz[i][i];
    }
     
    //Mostrar los valores de las variables    
    for(i=0; i<=ORDEN-1; i++){
        printf("x%d =\t%.2f\n",i+1,x[i]);
    }
     
    return 0;
}
 
void mostrarMatriz(float matriz[ORDEN][COLS]){
    int i,j;
    printf("\n");
    for(i=0;i<=(ORDEN-1);i++){
        printf("|\t");
        for(j=0;j<=ORDEN;j++){
            printf("%.2f\t",matriz[i][j]);
        }
        printf("|\t\n");
    }
    printf("\n\n");
}
