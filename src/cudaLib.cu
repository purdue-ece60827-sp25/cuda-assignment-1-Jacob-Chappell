
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < size) y[i] = scale * x[i] + y[i];
}

int runGpuSaxpy(int vectorSize) {
	std::cout << "Hello GPU Saxpy!\n";

	int i;

	float scale = 6.3f;
	float * x = (float *)malloc(vectorSize * sizeof(float));
	float * y = (float *)malloc(vectorSize * sizeof(float));
	float * result = (float *)malloc(vectorSize * sizeof(float));

	if(!x || !y) {
		std::cout << "Malloc failed";
		return -1;
	}

	// generate random vectors
	for(i = 0; i < vectorSize; i ++) {
		x[i] = (float)(rand() % 1000);
		y[i] = (float)(rand() % 1000);
	}

	// assemble and launch gpu kernel
	float * gpu_x;
	cudaMalloc(&gpu_x, vectorSize * sizeof(float));
	float * gpu_y;
	cudaMalloc(&gpu_y, vectorSize * sizeof(float));

	cudaMemcpy(gpu_x, x, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_y, y, vectorSize * sizeof(float), cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocks = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;
	saxpy_gpu<<<blocks, threadsPerBlock>>>(gpu_x, gpu_y, scale, vectorSize);

	cudaMemcpy(result, gpu_y, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(gpu_x);
	cudaFree(gpu_y);

	// check result
	float fudgeFactor = 0.001;
	int error_count = 0;
	for(i = 0; i < vectorSize; i ++) {
		float exp_result = scale * x[i] + y[i];
		if(
			result[i] < exp_result - fudgeFactor ||
			result[i] > exp_result + fudgeFactor
		) {
			if(error_count < 20) std::cout << "Got: " << result[i] << " Expected: " << exp_result << "\n";
			error_count ++;
		}
	}

	std::cout << "Found " << error_count << " / " << vectorSize << " errors \n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
