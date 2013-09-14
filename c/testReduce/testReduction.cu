#include <time.h>
#include <stdio.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaUtilities.h"

__global__ void reduceArray(float* arrayIn, float* arrayOut, unsigned int n)
{
	//This algorithm was used from a this website:
	// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
	// accessed 4/28/2013
	
	extern __shared__ float sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x*2 + tid;
	unsigned int gridSize = blockDim.x*2*gridDim.x;
	sdata[tid] = 0;

	if ( i < n )
		sdata[tid] = arrayIn[i];

	if ( i+blockDim.x < n )
		sdata[tid] += arrayIn[i+blockDim.x];

	__syncthreads();

	for ( unsigned int step=(blockDim.x/2); step > 0; step = step >> 1 )
	{
		if ( tid < step )
			sdata[tid] += sdata[tid + step];
		__syncthreads();
	}

	if (tid==0)
		arrayOut[blockIdx.x] = sdata[0];
}


void calcBlockThread(unsigned int n, cudaDeviceProp& prop, dim3& blocks, dim3& threads)
{
	if (n<prop.maxThreadsPerBlock)
	{
		threads.x = n;
		threads.y = 1;
		threads.z = 1;
		blocks.x = 1;
		blocks.y = 1;
		blocks.z = 1;
	} 
	else
	{
		threads.x = prop.maxThreadsPerBlock;
		threads.y = 1;
		threads.z = 1;
		blocks.x = ceil((float)n/prop.maxThreadsPerBlock);
		blocks.y = 1;
		blocks.z = 1;
	}
}

void calcBlockThread(unsigned int width, unsigned int height, cudaDeviceProp& prop, dim3& blocks, dim3& threads)
{
	if (width*height<prop.maxThreadsPerBlock)
	{
		threads.x = width;
		threads.y = height;
		threads.z = 1;
		blocks.x = 1;
		blocks.y = 1;
		blocks.z = 1;
	} 
	else
	{
		int dim = sqrt((float)prop.maxThreadsPerBlock);
		threads.x = dim;
		threads.y = dim;
		threads.z = 1;
		blocks.x = ceil((float)width/dim);
		blocks.y = ceil((float)height/dim);
		blocks.z = 1;
	}
}

void calcBlockThread(unsigned int width, unsigned int height, unsigned int depth, const cudaDeviceProp &prop, dim3 &blocks,
	dim3 &threads )
{
	if(width*height*depth < prop.maxThreadsPerBlock)
	{
		blocks.x = 1;
		blocks.y = 1;
		blocks.z = 1;
		threads.x = width;
		threads.y = height;
		threads.z = depth;
	}
	else
	{
		int dim = pow((float)prop.maxThreadsPerBlock,1/3.0f);
		int eqDim = dim*dim*dim;
		int extra = prop.maxThreadsPerBlock/eqDim;

		threads.x = dim * (prop.maxThreadsPerBlock/eqDim);
		threads.y = dim;
		threads.z = dim;

		blocks.x = ceil((float)width/threads.x);
		blocks.y = ceil((float)height/threads.y);
		blocks.z = ceil((float)depth/threads.z);
	}
}

float cpuReduce(float* input, unsigned int length, dim3 blocks, dim3 threads)
{
	float* tempMem = new float[2*threads.x*blocks.x];

	// Initial copy/add
	for ( int i=0; i < blocks.x; ++i )
	{
		unsigned int inBlockStart = 2*i*threads.x;
		unsigned int outBlockStart = i*threads.x;
		for ( int j=0; j < threads.x; ++j )
		{
			tempMem[outBlockStart + j] = 0.0;

			if ( (inBlockStart + j) < length )
				tempMem[outBlockStart + j] = input[inBlockStart + j];

			if ( (inBlockStart + threads.x + j) < length )
				tempMem[outBlockStart + j] += input[inBlockStart + threads.x + j];
		}
	}

	// Binary reduction
	for ( int binStep=threads.x/2; binStep > 0; binStep = binStep >> 1 )
	{
		for ( int i=0; i < blocks.x; ++i )
		{
			unsigned int blockStart = i*threads.x;
			for ( int j=0; j < binStep; ++j )
			{
				tempMem[blockStart + j] += tempMem[blockStart + binStep + j];
			}
		}
	}

	float finalSum = 0;
	// Final Reduction using straight sum
	for ( int i=0; i < blocks.x; ++i )
		finalSum += tempMem[i*threads.x];

	delete tempMem;

	return finalSum;
}

int main(void)
{
	cudaDeviceProp prop;
	dim3 maxBlocks;
	dim3 maxThreads;

	const unsigned int maxArraySizes = 100000;


	HANDLE_ERROR(cudaSetDevice(0));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));
	calcBlockThread(maxArraySizes, prop, maxBlocks, maxThreads);


	float* deviceReduceTestInput;
	float* deviceReduceTestOutput;

	float* startArray = new float[maxArraySizes];
	float* sumArray = new float[maxBlocks.x];

	HANDLE_ERROR(cudaMalloc((void**)&deviceReduceTestInput,sizeof(float)*maxArraySizes));
	HANDLE_ERROR(cudaMalloc((void**)&deviceReduceTestOutput,sizeof(float)*maxBlocks.x));

	for ( int n=500; n < maxArraySizes; ++n )
	{
		dim3 blocks;
		dim3 threads;

		for ( int i=0; i < n; ++i )
			startArray[i] = (((float)(rand() % 1000) / 500) - 1.0f);


		calcBlockThread(n, prop, blocks, threads);

		blocks.x = (blocks.x+1) / 2;

		float cpuSum = 0.0;
		for ( int i=0; i < n; ++i )
			cpuSum += startArray[i];

		// CPU Reduction
		float cpuReductionSum = cpuReduce(startArray, n, blocks, threads);


		// CUDA Reduction
		HANDLE_ERROR(cudaMemcpy(deviceReduceTestInput, startArray, sizeof(float)*n, cudaMemcpyHostToDevice));

		reduceArray<<<blocks.x,threads.x,sizeof(float)*threads.x>>>(deviceReduceTestInput, deviceReduceTestOutput, n);

		HANDLE_ERROR(cudaMemcpy(sumArray, deviceReduceTestOutput, sizeof(float)*blocks.x, cudaMemcpyDeviceToHost));

		float cudaSum = 0.0;
		for ( int i=0; i < blocks.x; ++i )
			cudaSum += sumArray[i];

		float err = abs(cpuSum - cudaSum);
		float redErr = abs(cpuReductionSum - cudaSum);

		printf("%f : %f : (%f, %f, %f)\n", err, redErr, cpuSum, cudaSum, cpuReductionSum);
	}

	HANDLE_ERROR(cudaFree(deviceReduceTestInput));
	HANDLE_ERROR(cudaFree(deviceReduceTestOutput));

	delete startArray;
	delete sumArray;
}