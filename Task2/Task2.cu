#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cmath>
clock_t begin1, begin2, end1, end2;

// Macierze są pamiętane wierszami, a więc:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
        int width;
        int height;
        float *elements;
} Matrix;

#define BLOCK_SIZE 32

// prototyp funkcji mnożącej (kernela)
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Zakładamy (dla uproszczenia rozważań), że wymiary macierzy są
// całkowitymi wielokrotnościami wartości BLOCK_SIZE
// Funkcja mnożąca

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
        // kopiujemy macierze A i B to globalnej pamięci urządzenia
        // najpierw A
        Matrix d_A;
        d_A.width = A.width;
        d_A.height = A.height;
        size_t size = A.width * A.height * sizeof(float);
        cudaMalloc((void **)&d_A.elements, size);
        cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
        // potem B
        Matrix d_B;
        d_B.width = B.width;
        d_B.height = B.height;
        size = B.width * B.height * sizeof(float);
        cudaMalloc((void **)&d_B.elements, size);
        cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
        // przydzielamy macierz C w globalnej pamięci urządzenia
        Matrix d_C;
        d_C.width = C.width;
        d_C.height = C.height;
        size = C.width * C.height * sizeof(float);
        cudaMalloc((void **)&d_C.elements, size);
        // preparujemy środowisko i wywołujemy kernel
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        //dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
        dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
        begin1 = clock();
        MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C);
        cudaThreadSynchronize();
        end1 = clock();
        // odbieramy obliczoną macierz C z pamięci globalnej urządzenia
        cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
        // zwalniamy pamięć
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
}


// kernel odpowiedzialny za wymnożenie macierzy
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
        // każdy wątek oblicza jeden element macierzy C
        // akumulując wynik w zmiennej Cvalue
        float Cvalue = 0;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        for (int e = 0; e < A.width; ++e)
                Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
        //Cvalue = powf(Cvalue, 10);
        C.elements[row * C.width + col] = Cvalue;
}

Matrix newMatrix(int row, int col)
{
        Matrix newM;
        newM.width = row;
        newM.height = col;
        newM.elements = (float*)malloc(row * col * sizeof(float));
        return newM;
}


int main(int argc, char** argv)
{
        int N = 960;

        Matrix A = newMatrix(N, N);
        Matrix B = newMatrix(N, N);
        Matrix C = newMatrix(N, N);
        Matrix D = newMatrix(N, N);

        for (int i = 0; i < N; i++)
        {
                for (int j = 0; j < N; j++)
                {
                        A.elements[i*N + j] = 2.45317621124123;
                        B.elements[i*N + j] = 2.54493874134242;
                }
        }

        //begin1 = clock();
        MatMul(A, B, C);
        //end1 = clock();

        //Mnożenie CPU
        begin2 = clock();
        float suma;
        for (int row = 0; row < N; row++) {
                for (int col = 0; col < N; col++) {
                        suma = 0.f;
                        for (int n = 0; n < N; n++) {
                                suma += A.elements[row*N + n] * B.elements[n*N + col];
                        }
                        //suma = powf(suma, 10);
                        D.elements[row*N + col] = suma;
                }
        }
        end2 = clock();

        double time_spent1 = (double)(end1 - begin1) / CLOCKS_PER_SEC;
        double time_spent2 = (double)(end2 - begin2) / CLOCKS_PER_SEC;

        printf("Wynik na GPU: %.24f \n", C.elements[0]);
        printf("Wynik na CPU: %.24f \n", D.elements[0]);
        //Full range 8
        printf("Roznica GPU-CPU: %.24f \n", C.elements[0] - D.elements[0]);
        printf("Czas GPU: %.16f \n", time_spent1);
        printf("Czas CPU: %.16f \n", time_spent2);

        free(A.elements);
        free(B.elements);
        free(C.elements);
        free(D.elements);

        return 0;
}
