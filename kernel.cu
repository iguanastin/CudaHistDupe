
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <chrono>

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
cudaError_t findDupes(const float*, unsigned int, std::vector<Pair>&, int*, int, float*);



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
    bool cuda = true;

    std::chrono::steady_clock::time_point time;


    // Print some diagnostics
    std::cout << "Datafile Path: " << argv[1] << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "Max Results: " << max_results << std::endl;
    std::cout << "Confidence: " << confidence << std::endl;
    std::cout << "Color Variance: " << color_variance << std::endl;


    // Allocate some arrays
    std::cout << "Allocating memory..." << std::endl;
    time = std::chrono::steady_clock::now();
    int* ids = new int[N]; // Mapping of actual index to ID of histogram
    float* data = new float[128 * N]; // End-to-end array of all histograms. Each histogram consists of 128 floats
    float* conf = new float[N]; // Confidence array; allows using stricter confidence for black and white images
    std::vector<Pair> pairs; // Vector of similar pairs (to be populated)
    std::cout << "Allocated memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;



    // Read test data from file
    std::cout << "Reading data from file: " << argv[1] << "..." << std::endl;
    time = std::chrono::steady_clock::now();

    FILE* file;
    file = fopen(argv[1], "r");

    for (int i = 0; i < N; i++) {
        fscanf(file, "%d", &ids[i]);
        for (int j = 0; j < 128; j++) {
            fscanf(file, "%f", &data[i * 128 + j]);
        }
    }

    fclose(file);
    std::cout << "Read data in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;



    // Build confidence array
    std::cout << "Building confidence array..." << std::endl;
    time = std::chrono::steady_clock::now();
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
    std::cout << "Built confidence array in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


    // Find duplicates
    std::cout << "Finding duplicates..." << std::endl;
    cudaError_t cudaStatus;
    int result_count = 0;
    time = std::chrono::steady_clock::now();
    if (cuda) {
        // With CUDA
        cudaStatus = findDupes(data, N, pairs, &result_count, max_results, conf);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel failed!");
            return 1;
        }
        for (int i = 0; i < result_count; i++) {
            pairs[i].id1 = ids[pairs[i].id1];
            pairs[i].id2 = ids[pairs[i].id2];
        }
    } else {
        // Sequentially
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                double d = 0;
                for (int k = 0; k < 128; k++) {
                    d += fabs(data[i * 128 + k] - data[j * 128 + k]);
                }
                d = 1 - (d / 8);
                if (d > fmaxf(conf[i], conf[j])) {
                    Pair p;
                    p.similarity = (float) d;
                    p.id1 = ids[i];
                    p.id2 = ids[j];
                    pairs.push_back(p);
                    result_count++;
                }
            }
        }
    }
    std::cout << "Found duplicates in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


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



cudaError_t findDupes(const float* data, unsigned int N, std::vector<Pair>& pairs, int* result_count, int max_results, float* confidence) {

    float* d_data; // Data device pointer
    Pair* d_pairs; // Pairs device pointer
    float* d_confidence; // Confidence device pointer
    int* d_result_count; // Result count device pointer

    cudaError_t cudaStatus; // CUDA error

    std::chrono::steady_clock::time_point time; // Time tracking

    unsigned int dN = N; // Padded device N to match block size
    if (N % 64 != 0) {
        dN = (unsigned int) ceil((double) N / 64) * 64;
    }
    std::cout << "Adjusted N: " << dN << std::endl;

    // Choose which GPU to run on, change this on a multi-GPU system
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    // Allocate GPU buffers
    std::cout << "Allocating GPU memory..." << std::endl;
    time = std::chrono::steady_clock::now();
    cudaStatus = cudaMalloc((void**) &d_data, sizeof(float) * 128 * dN);
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
    cudaStatus = cudaMalloc((void**) &d_confidence, sizeof(float) * dN);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    std::cout << "Allocated GPU memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;

    // Copy input data from host memory to GPU buffers
    std::cout << "Copying data to device..." << std::endl;
    time = std::chrono::steady_clock::now();
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
    if (dN > N) {
        float* temp_conf = new float[dN - N];
        for (int i = 0; i < dN - N; i++) temp_conf[i] = 2; // Impossible confidence
        cudaStatus = cudaMemcpy(d_confidence + N, temp_conf, sizeof(float) * (dN - N), cudaMemcpyHostToDevice);
        delete[] temp_conf;
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    }
    std::cout << "Copied data to GPU memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


    // Launch a kernel on the GPU
    std::cout << "Launching kernel..." << std::endl;
    time = std::chrono::steady_clock::now();
    histDupeKernel KERNEL_ARGS((int) ceil((double) N / 64), 64) (d_data, N, d_pairs, d_result_count, max_results, d_confidence);

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
    std::cout << "Ran GPU kernel in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;

    // Copy output from GPU buffer to host memory.
    std::cout << "Copying results from device..." << std::endl;
    time = std::chrono::steady_clock::now();
    cudaStatus = cudaMemcpy((void*) result_count, d_result_count, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    result_count[0] = __min(result_count[0], max_results); // Clamp result_count
    // Read result pairs into buffer
    Pair* temp_pairs = new Pair[result_count[0]];
    cudaStatus = cudaMemcpy((void*) temp_pairs, d_pairs, sizeof(Pair) * result_count[0], cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // Only keep pairs that are unique (pairs are commutative)
    for (int i = 0; i < result_count[0]; i++) {
        int p1id1 = temp_pairs[i].id1;
        int p1id2 = temp_pairs[i].id2;
        bool found = false;
        for each (const Pair p2 in pairs) {
            if ((p1id1 == p2.id1 && p1id2 == p2.id2) || (p1id1 == p2.id2 && p1id2 == p2.id1)) {
                found = true;
                break;
            }
        }

        if (!found) {
            pairs.push_back(temp_pairs[i]);
        }
    }
    delete[] temp_pairs;
    result_count[0] = (int) pairs.size();
    std::cout << "Retrieved results from GPU memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;

Error:

    // Free cuda memory
    std::cout << "Freeing GPU memory..." << std::endl;
    time = std::chrono::steady_clock::now();
    cudaFree(d_data);
    cudaFree(d_pairs);
    cudaFree(d_result_count);
    cudaFree(d_confidence);
    std::cout << "Freed GPU memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;

    return cudaStatus;
}



__global__ void histDupeKernel(const float* data, int N, Pair* results, int* result_count, int max_results, float* confidence) {

    int thread = threadIdx.x; // Thread index within block
    int block = blockIdx.x; // Block index
    int block_size = blockDim.x; // Size of each block

    int block_start = block_size * block;
    int index = block_start + thread; // Index of histogram for this thread

    __shared__ float conf[64]; // Shared array of confidence values for all histograms owned by this block
    conf[thread] = confidence[index]; // Coalesced read of confidence values

    __shared__ float hists[128 * 64];
    for (int i = 0; i < 64; i++) {
        hists[i * 128 + thread] = data[(block_start + i) * 128 + thread];
        hists[i * 128 + thread + 64] = data[(block_start + i) * 128 + 64 + thread];
    }

    __shared__ float other[128]; // Histogram to compare all owned histograms against parallely
    for (int i = 0; i < N && *result_count < max_results; i++) {

        float other_conf = confidence[i]; // All threads read confidence for other histogram into register

        other[thread] = data[i * 128 + thread]; // Coalesced read of other histogram into shared memory
        other[thread + 64] = data[i * 128 + thread + 64];

        __syncthreads(); // Ensure all values read

        if (index < N) {
            float d = 0;
            for (int k = 0; k < 128; k++) { // Compute sum of distances between thread-owned histogram and shared histogram
                d += std::fabsf(hists[thread * 128 + k] - other[k]);
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
        }

        __syncthreads(); // Ensure all threads have finished before looping and reading new shared histogram
    }

}
