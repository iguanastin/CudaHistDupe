
#include "histdupe.h"
#include <algorithm>
#include <iterator>




__global__ void histDupeKernel(const float* data, float* confidence, Pair* results, int* result_count, int N, int max_results) {

    int thread = threadIdx.x; // Thread index within block
    int block = blockIdx.x; // Block index
    int block_size = blockDim.x; // Size of each block

    int block_start = block_size * block; // Index of the start of the block
    int index = block_start + thread; // Index of this thread

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

        other[thread] = data[i * 128 + thread]; // Coalesced read of first half of other histogram
        other[thread + 64] = data[i * 128 + thread + 64]; // Second half

        __syncthreads(); // Ensure all values read

        if (index < N) {
            float d = 0;
            for (int k = 0; k < 128; k++) { // Compute sum of distances between thread-owned histogram and shared histogram
                d += std::fabsf(hists[thread * 128 + k] - other[k]);
            }
            d = 1 - (d / 8); // Massage the difference into a nice % similarity number, between 0 and 1

            if (i != index && d > fmaxf(conf[thread], other_conf)) { // Don't compare against self, only compare using highest confidence
                int result_index = atomicAdd(result_count, 1); // Increment result count by one atomically (returns value before increment)
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