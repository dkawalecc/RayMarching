#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

__global__ void vecadd(int n, float *x, float *y, float *z)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
  	if (i < n) z[i] = x[i] + y[i];
}

int main(void){
	srand((unsigned int)time(NULL));
	int N=100;
	
	float *x, *y, *z, *d_x, *d_y, *d_z;
	x = new float[N];
	y = new float[N];
	z = new float[N];
	for (int i = 0; i < N; i++){
		x[i] = (float)rand() / RAND_MAX;
		y[i] = (float)rand() / RAND_MAX;
	}

	cudaMalloc(&d_x, N*sizeof(float));
	cudaMalloc(&d_y, N*sizeof(float));
	cudaMalloc(&d_z, N*sizeof(float));
	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
	
	vecadd <<<(N + 1023) / 1024, 1024>>> (N, d_x, d_y, d_z);
	
	cudaMemcpy(z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
		cout << x[i] << " + " << y[i] << " = " << z[i] << "\t(" << x[i] + y[i] << ")" << endl;

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	delete[] x;
	delete[] y;
	delete[] z;
	return 0;
}