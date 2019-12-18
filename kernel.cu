
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>



typedef struct {
    int id1;
    int id2;
    float similarity;
} Pair;



__global__ void histDupeKernel(const float*, int, Pair*, int*, int, float*);



int main(int argc, char* argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Too few arguments. Expected 1\nUsage: %s DATA_PATH\n", argv[0]);
        return 1;
    }


    int max_results = 1000000;
    float confidence = 0.99f;
    float color_variance = 0.25f;
    int N = 50000;


    std::cout << "Datafile Path: " << argv[1] << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "Max Results: " << max_results << std::endl;
    std::cout << "Confidence: " << confidence << std::endl;
    std::cout << "Color Variance: " << color_variance << std::endl;


    int* ids = new int[N];
    float* data = new float[128 * N];
    float* conf = new float[N];
    Pair* pairs = new Pair[max_results];


    // Read test data from file
    std::ifstream data_file(argv[1]);
    int c = 0;
    for (std::string line; std::getline(data_file, line);) {
        std::istringstream in(line);

        in >> ids[c];

        for (int i = 0; i < 128; i++) {
            in >> data[i];
        }
    }


    // Build confidence array
    float confidence_square = 1 - (1 - confidence) * (1 - confidence);
    for (int i = 0; i < N; i++) {
        float d = 0;

        for (int k = 0; k < 32; k++) {
            float r = data[i * 128 + k + 32];
            float g = data[i * 128 + k + 64];
            float b = data[i * 128 + k + 96];
            d += __max(__max(r, g), b) - __min(__min(r, g), b);
        }

        if (d > color_variance) {
            conf[i] = confidence;
        }
        else {
            conf[i] = confidence_square;
        }
    }


    // Find duplicates with CUDA
    int result_count;
    cudaError_t cudaStatus = findDupes(data, N, pairs, &result_count, max_results, conf);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel failed!");
        return 1;
    }


    // Print some results
    std::cout << "Found pairs: " << result_count << std::endl;
    std::cout << "Example results:" << std::endl;
    for (int i = 0; i < __min(result_count, 10); i++) {
        std::cout << "\t" << ids[pairs[i].id1] << " - " << ids[pairs[i].id2] << ":\t\t" << pairs[i].similarity << std::endl;
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    delete[] data;
    delete[] conf;
    delete[] pairs;
    delete[] ids;

    return 0;
}



cudaError_t findDupes(const float* data, unsigned int N, const Pair* pairs, int* result_count, int max_results, float* confidence) {

    float* d_data;
    Pair* d_pairs;
    float* d_confidence;
    float* d_result_count;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**) &d_data, sizeof(float) * 128 * N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**) &d_pairs, sizeof(Pair) * max_results);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**) &d_result_count, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**) &d_confidence, sizeof(float) * N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_data, data, sizeof(int) * 128 * N, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_confidence, confidence, sizeof(float) * N, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    // Launch a kernel on the GPU with one thread for each element.
    histDupeKernel<<<(int) ceil((double) N / 128), 128>>>(d_data, N, d_pairs, d_result_count, max_results, d_confidence);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy((void*) result_count, d_result_count, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy((void*) pairs, d_pairs, sizeof(Pair) * __min(result_count[0], max_results), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(d_data);
    cudaFree(d_pairs);
    cudaFree(d_result_count);
    cudaFree(d_confidence);

    return cudaStatus;
}



__global__ void histDupeKernel(const float* data, int N, Pair* results, int* result_count, int max_results, float* confidence) {

    int thread = threadIdx.x;
    int block = blockIdx.x;
    int block_size = blockDim.x;

    int index = block_size * block + thread;

    if (index < N) {
        __shared__ float conf[128];
        conf[thread] = confidence[index];

        float hist[128];
        for (int i = 0; i < 128; i++) {
            hist[i] = data[index * 128 + i];
        }

        __shared__ float other[128];

        for (int i = 0; i < N && *result_count < max_results; i++) {

            float other_conf = confidence[i];

            other[thread] = data[i * 128 + thread];
            __syncthreads();

            float d = 0;
            for (int k = 0; k < 128; k++) {
                d += std::fabsf(hist[k] - other[k]);
            }
            d = 1 - (d / 8);
            if (i != index && d > fmaxf(conf[thread], other_conf)) {
                int result_index = atomicAdd(result_count, 1);
                if (result_index < max_results) {
                    results[result_index].similarity = d;
                    results[result_index].id1 = index;
                    results[result_index].id2 = i;
                }
            }

            __syncthreads();
        }
    }

}
