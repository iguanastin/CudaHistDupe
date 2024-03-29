
#include "histdupe.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include <chrono>



/*
Utility method for launching a CUDA kernel. Performes all the nasty checks, allocation, and copying of data.
*/
cudaError_t findDupes(const float*, const float*, const float*, const float*, const int*, const int*, std::vector<Pair>&, int*, const int, const int, const int);



int main(int argc, char* argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Too few arguments. Expected 1\n\nUsage: %s DATA_PATH\n", argv[0]);
        return 1;
    }


    // Initialize variables
    // Maximum number of results to return. Only applies to CUDA launches
    int max_results = 1000000;
    // Base confidence value for similar pairs
    float confidence = 0.95f;
    // Maximum color variance per histogram. Used to determine if an image is black-and-white or colorful
    float color_variance = 0.25f;
    // Number of histograms in the dataset
    int N = 50000;
    // N subset
    int subN = 25000;
    // Use CUDA to find similar pair
    bool cuda = true;

    // Clock used for timing
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

    int* ids1 = new int[subN]; // Mapping of actual index to ID of histogram
    int* ids2 = new int[N];
    float* data1 = new float[128 * subN];
    float* data2 = new float[128 * N]; // End-to-end array of all histograms. Each histogram consists of 128 floats
    float* conf1 = new float[subN];
    float* conf2 = new float[N]; // Confidence array; allows using stricter confidence for black and white images
    std::vector<Pair> pairs; // Vector of similar pairs (to be populated)

    std::cout << "Allocated memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;



    // Read test data from file
    std::cout << "Reading data from file: " << argv[1] << "..." << std::endl;
    time = std::chrono::steady_clock::now();

    FILE* file; // Data file
    file = fopen(argv[1], "r"); // Open data file to read
    for (int i = 0; i < N; i++) {
        fscanf(file, "%d", &ids2[i]); // Read first int as id of histogram
        for (int j = 0; j < 128; j++) { // Read 128 floats as histogram elements
            fscanf(file, "%f", &data2[i * 128 + j]);
        }
    }
    fclose(file); // Close data file

    // Copy data and ids for subset
    for (int i = 0; i < subN; i++) {
        ids1[i] = ids2[i];
        for (int j = 0; j < 128; j++) {
            data1[i * 128 + j] = data2[i * 128 + j];
        }
    }

    std::cout << "Read data in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;



    // Build confidence array
    std::cout << "Building confidence array..." << std::endl;
    time = std::chrono::steady_clock::now();

    float confidence_square = 1 - (1 - confidence) * (1 - confidence); // Squared confidence for comparing black and white images.
    // Generate confidence array
    for (int i = 0; i < N; i++) {
        float d = 0;

        // Compute sum of color variance across histogram
        for (int k = 0; k < 32; k++) {
            // Ignore alpha values (first 32 floats)
            float r = data2[i * 128 + k + 32];
            float g = data2[i * 128 + k + 64];
            float b = data2[i * 128 + k + 96];
            d += __max(__max(r, g), b) - __min(__min(r, g), b);
        }

        if (d > color_variance) {
            conf2[i] = confidence; // Image is colorful, use normal confidence
        } else {
            conf2[i] = confidence_square; // Image is not colorful, use squared confidence
        }
    }

    // Copy confidences to subset
    for (int i = 0; i < subN; i++) {
        conf1[i] = conf2[i];
    }

    std::cout << "Built confidence array in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


    // Find duplicates
    std::cout << "Finding duplicates..." << std::endl;
    time = std::chrono::steady_clock::now();

    cudaError_t cudaStatus; // CUDA Status variable
    int result_count = 0; // Track number of results
    if (cuda) {
        // With CUDA
        cudaStatus = findDupes(data1, data2, conf1, conf2, ids1, ids2, pairs, &result_count, subN, N, max_results);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel failed!");
            return 1;
        }
    } else {
        // Sequentially
        for (int i = 0; i < subN; i++) {
            for (int j = 0; j < N; j++) {
                double d = 0;
                for (int k = 0; k < 128; k++) {
                    d += fabs(data1[i * 128 + k] - data2[j * 128 + k]);
                }
                d = 1 - (d / 8);
                if (d > fmaxf(conf1[i], conf2[j])) { // Use highest confidence value of the two histograms
                    Pair p;
                    p.similarity = (float) d;
                    p.id1 = ids1[i];
                    p.id2 = ids2[j];
                    if (p.id1 != p.id2) {
                        pairs.push_back(p);
                        result_count++;
                    }
                }
            }
        }
    }

    std::cout << "Found duplicates in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


    // Print some results
    std::cout << "Found pairs: " << result_count << std::endl;
    std::cout << "Example results:" << std::endl;
    for (int i = 0; i < __min(result_count, 10); i++) {
        std::cout << "\t" << pairs[i].id1 << " - " << pairs[i].id2 << ":\t\t" << pairs[i].similarity << std::endl;
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    // Delete arrays
    delete[] data1;
    delete[] data2;
    delete[] conf1;
    delete[] conf2;
    delete[] ids1;
    delete[] ids2;

    return 0;
}



cudaError_t findDupes(const float* data1, const float* data2, const float* conf1, const float* conf2, const int* ids1, const int* ids2, std::vector<Pair>& pairs, int* result_count, const int N1, const int N2, const int max_results) {

    float* d_data1; // Data device pointer
    float* d_data2;
    float* d_confidence1; // Confidence device pointer
    float* d_confidence2;
    int* d_ids1;
    int* d_ids2;
    int* d_results_id1;
    int* d_results_id2;
    float* d_results_similarity;
    int* d_result_count; // Result count device pointer
    cudaError_t cudaStatus; // CUDA error

    std::chrono::steady_clock::time_point time; // Time tracking


    int dN = N1; // Padded device N to match block size
    if (N1 % 64 != 0) {
        dN = (int) ceil((double) N1 / 64) * 64;
    }
    std::cout << "Adjusted N1: " << dN << std::endl;


    // Choose which GPU to run on, change this on a multi-GPU system
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }


    // Allocate GPU buffers
    std::cout << "Allocating GPU memory..." << std::endl;
    time = std::chrono::steady_clock::now();
    cudaStatus = cudaMalloc((void**) &d_data1, sizeof(float) * 128 * dN); // Allocate memory for histogram data
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_data2, sizeof(float) * 128 * N2); // Allocate memory for histogram data
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_confidence1, sizeof(float) * dN); // Allocate memory for confidence array
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_confidence2, sizeof(float) * N2); // Allocate memory for confidence array
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_ids1, sizeof(int) * dN); // Allocate memory for ids array
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_ids2, sizeof(int) * N2); // Allocate memory for ids array
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**) &d_results_id1, sizeof(int) * max_results); // Allocate memory for results
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_results_id2, sizeof(int) * max_results); // Allocate memory for results
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_results_similarity, sizeof(float) * max_results); // Allocate memory for results
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**) &d_result_count, sizeof(int)); // Allocate single int for result count
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    std::cout << "Allocated GPU memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


    // Copy input data from host memory to GPU buffers
    std::cout << "Copying data to device..." << std::endl;
    time = std::chrono::steady_clock::now();
    cudaStatus = cudaMemcpy(d_data1, data1, sizeof(int) * 128 * N1, cudaMemcpyHostToDevice); // Copy histogram data to device
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_data2, data2, sizeof(int) * 128 * N2, cudaMemcpyHostToDevice); // Copy histogram data to device
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_confidence1, conf1, sizeof(float) * N1, cudaMemcpyHostToDevice); // Copy confidence array to device
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_confidence2, conf2, sizeof(float) * N2, cudaMemcpyHostToDevice); // Copy confidence array to device
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    if (dN > N1) {
        // Copy padded data to device at end of confidence array
        float* temp_conf = new float[dN - N1]; // Temp array of padded confidence values
        for (int i = 0; i < dN - N1; i++) temp_conf[i] = 2; // Impossible confidence
        cudaStatus = cudaMemcpy(d_confidence1 + N1, temp_conf, sizeof(float) * (dN - N1), cudaMemcpyHostToDevice); // Copy padded confidence values to device
        delete[] temp_conf; // Delete temp array
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    }
    cudaStatus = cudaMemcpy(d_ids1, ids1, sizeof(int) * N1, cudaMemcpyHostToDevice); // Copy ids array to device
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(d_ids2, ids2, sizeof(int) * N2, cudaMemcpyHostToDevice); // Copy ids array to device
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    std::cout << "Copied data to GPU memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


    // Launch a kernel on the GPU
    std::cout << "Launching kernel..." << std::endl;
    time = std::chrono::steady_clock::now();
    histDupeKernel KERNEL_ARGS((int) ceil((double) N1 / 64), 64) (d_data1, d_data2, d_confidence1, d_confidence2, d_ids1, d_ids2, d_results_id1, d_results_id2, d_results_similarity, d_result_count, N1, N2, max_results); // Launch CUDA kernel


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
    cudaStatus = cudaMemcpy((void*) result_count, d_result_count, sizeof(float), cudaMemcpyDeviceToHost); // Copy result count from device
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    result_count[0] = __min(result_count[0], max_results); // Clamp result_count to max_results
    // Read result pairs into buffer
    {
        int* temp_id1 = new int[result_count[0]];
        int* temp_id2 = new int[result_count[0]];
        float* temp_similarity = new float[result_count[0]];
        cudaStatus = cudaMemcpy((void*) temp_id1, d_results_id1, sizeof(int) * result_count[0], cudaMemcpyDeviceToHost); // Copy results from device
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
        cudaStatus = cudaMemcpy((void*)temp_id2, d_results_id2, sizeof(int) * result_count[0], cudaMemcpyDeviceToHost); // Copy results from device
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
        cudaStatus = cudaMemcpy((void*)temp_similarity, d_results_similarity, sizeof(float) * result_count[0], cudaMemcpyDeviceToHost); // Copy results from device
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
        // Only keep pairs that are unique (pairs are commutative)
        for (int i = 0; i < result_count[0]; i++) {
            bool found = false;
            for (const Pair p2 : pairs) {
                if ((temp_id1[i] == p2.id1 && temp_id2[i] == p2.id2) || (temp_id1[i] == p2.id2 && temp_id2[i] == p2.id1)) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                Pair pair;
                pair.id1 = temp_id1[i];
                pair.id2 = temp_id2[i];
                pair.similarity = temp_similarity[i];
                pairs.push_back(pair); // Only keep pair if it is unique
            }
        }
        delete[] temp_id1; // Delete temp results buffer
        delete[] temp_id2; // Delete temp results buffer
        delete[] temp_similarity; // Delete temp results buffer
    }
    result_count[0] = (int) pairs.size(); // Reset result_count to count of final result set
    std::cout << "Retrieved results from GPU memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;


Error:
    // Free cuda memory
    std::cout << "Freeing GPU memory..." << std::endl;
    time = std::chrono::steady_clock::now();
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_confidence1);
    cudaFree(d_confidence2);
    cudaFree(d_ids1);
    cudaFree(d_ids2);
    cudaFree(d_results_id1);
    cudaFree(d_results_id2);
    cudaFree(d_results_similarity);
    cudaFree(d_result_count);
    std::cout << "Freed GPU memory in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() << " ms" << std::endl;

    return cudaStatus;
}
