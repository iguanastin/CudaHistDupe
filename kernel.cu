
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

#ifdef __INTELLISENSE__
#include "intellisense_cuda_intrinsics.h"
#define KERNEL_ARGS(grid, block)
#else
#define KERNEL_ARGS(grid, block) <<< grid, block >>>
#endif



typedef struct {
    int id1;
    int id2;
    float similarity;
} Pair;



// Method signatures
__global__ void histDupeKernel(const float*, int, Pair*, int*, int, float*);
cudaError_t findDupes(const float*, unsigned int, const std::vector<Pair>&, int*, int, float*);



int main(int argc, char* argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Too few arguments. Expected 1\n\nUsage: %s DATA_PATH\n", argv[0]);
        return 1;
    }


    // Initialize variables
    int max_results = 1000000;
    float confidence = 0.99f;
    float color_variance = 0.25f;
    int N = 50000;


    // Print some diagnostics
    std::cout << "Datafile Path: " << argv[1] << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "Max Results: " << max_results << std::endl;
    std::cout << "Confidence: " << confidence << std::endl;
    std::cout << "Color Variance: " << color_variance << std::endl;


    // Allocate some arrays
    int* ids = new int[N]; // Mapping of actual index to ID of histogram
    float* data = new float[128 * N]; // End-to-end array of all histograms. Each histogram consists of 128 floats
    float* conf = new float[N]; // Confidence array; allows using stricter confidence for black and white images
    std::vector<Pair> pairs; // Vector of similar pairs (to be populated)


    // Read test data from file
    std::ifstream data_file(argv[1]);
    int c = 0;
    for (std::string line; std::getline(data_file, line);) {
        std::istringstream in(line);

        // Read first element of line as ID of histogram
        in >> ids[c];

        // Read 128 histogram bins
        for (int i = 0; i < 128; i++) {
            in >> data[i];
        }
    }


    // Build confidence array
    float confidence_square = 1 - (1 - confidence) * (1 - confidence);
    for (int i = 0; i < N; i++) {
        float d = 0;

        // Compute sum of color variance across histogram
        for (int k = 0; k < 32; k++) {
            float r = data[i * 128 + k + 32];
            float g = data[i * 128 + k + 64];
            float b = data[i * 128 + k + 96];
            d += __max(__max(r, g), b) - __min(__min(r, g), b);
        }

        if (d > color_variance) {
            conf[i] = confidence; // Image is colorful, use normal confidence
        } else {
            conf[i] = confidence_square; // Image is not colorful, use squared confidence
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

    // Delete arrays
    delete[] data;
    delete[] conf;
    delete[] ids;

    return 0;
}



cudaError_t findDupes(const float* data, unsigned int N, const std::vector<Pair> &pairs, int* result_count, int max_results, float* confidence) {

    float* d_data;
    Pair* d_pairs;
    float* d_confidence;
    int* d_result_count;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    // Allocate GPU buffers
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

    // Copy input data from host memory to GPU buffers
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


    // Launch a kernel on the GPU
    histDupeKernel KERNEL_ARGS((int) ceil((double) N / 128), 128) (d_data, N, d_pairs, d_result_count, max_results, d_confidence);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output from GPU buffer to host memory.
    cudaStatus = cudaMemcpy((void*) result_count, d_result_count, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    result_count[0] = __min(result_count[0], max_results); // Clamp result_count
    Pair* temp_pairs = new Pair[result_count[0]];
    cudaStatus = cudaMemcpy((void*) temp_pairs, d_pairs, sizeof(Pair) * result_count[0], cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    std::copy(temp_pairs, temp_pairs + result_count[0], std::back_inserter(pairs)); // Insert all retrieved results into vector
    delete[] temp_pairs;

Error:
    // Free cuda memory
    cudaFree(d_data);
    cudaFree(d_pairs);
    cudaFree(d_result_count);
    cudaFree(d_confidence);

    return cudaStatus;
}



__global__ void histDupeKernel(const float* data, int N, Pair* results, int* result_count, int max_results, float* confidence) {

    int thread = threadIdx.x; // Thread index within block
    int block = blockIdx.x; // Block index
    int block_size = blockDim.x; // Size of each block

    int index = block_size * block + thread; // Index of histogram for this thread

    if (index < N) {
        __shared__ float conf[128]; // Shared array of confidence values for all histograms owned by this block
        conf[thread] = confidence[index]; // Coalesced read of confidence values

        float hist[128]; // Histogram owned by this block, stored in registers
        for (int i = 0; i < 128; i++) {
            hist[i] = data[index * 128 + i];
        }

        __shared__ float other[128]; // Histogram to compare all owned histograms against parallely

        for (int i = 0; i < N && *result_count < max_results; i++) {

            float other_conf = confidence[i]; // All threads read confidence for other histogram into register

            other[thread] = data[i * 128 + thread]; // Coalesced read of other histogram into shared memory

            __syncthreads(); // Ensure all values read

            float d = 0;
            for (int k = 0; k < 128; k++) { // Compute sum of distances between thread-owned histogram and shared histogram
                d += std::fabsf(hist[k] - other[k]);
            }
            d = 1 - (d / 8); // Massage the difference into a nice % similarity number, between 0 and 1

            if (i != index && d > fmaxf(conf[thread], other_conf)) { // Don't compare against self, only compare using highest confidence
                int result_index = atomicAdd(result_count, 1); // Increment result count by one atomically
                if (result_index < max_results) {
                    // Store resulting pair
                    results[result_index].similarity = d;
                    results[result_index].id1 = index;
                    results[result_index].id2 = i;
                }
            }

            __syncthreads(); // Ensure all threads have finished before looping and reading new shared histogram
        }
    }

}
